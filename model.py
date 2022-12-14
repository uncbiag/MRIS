import os
import torch
import torch.nn.init
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils import cosine_sim, l2_norm


class Encoder(nn.Module):
    def __init__(self, cnn_type='resnet18', pretrained=True, gpu_ids=[]):
        super(Encoder, self).__init__()
        self.cnn = self.get_cnn(cnn_type, pretrained, gpu_ids)
        for name, p in self.cnn.named_parameters():
            p.requires_grad = True
        self.cnn.module.fc = nn.Sequential()

    def get_cnn(self, arch, pretrained, gpu_ids):
        # Load pretrained CNN and parallelize over GPUs
        if pretrained:
            print("using pre-trained model '{}' on ImageNet".format(arch))
        else:
            print("creating model '{}'".format(arch))
        os.environ['TORCH_HOME'] = '/playpen-raid/bqchen/code/OAIRetrieval/models'
        model = models.__dict__[arch](pretrained=pretrained)
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            model.to(gpu_ids[0])
            model = nn.DataParallel(model, gpu_ids)
        return model

    def forward(self, img):
        features = self.cnn(img)
        features = l2_norm(features)
        return features


class TripletLoss(nn.Module):
    # adaptive: use adaptive margin, pseudo_list != None: use pseudo positive labeling
    def __init__(self, margin=0, inverse=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.inverse = inverse
        self.sim = cosine_sim

    def forward(self, emb1, emb2):
        scores = self.sim(emb1, emb2)

        # row-wise negative - positive
        diagonal1 = scores.diag()
        d1 = diagonal1.view(emb1.size(0), 1).expand_as(scores)
        cost1 = (self.margin + scores - d1).clamp(min=0)

        # column-wise negative - positive
        diagonal2 = scores.diag()
        d2 = diagonal2.view(1, emb1.size(0)).expand_as(scores)
        cost2 = (self.margin + scores - d2).clamp(min=0)

        # if self.inverse, add up row-wise & column-wise
        cost = cost1 + cost2 if self.inverse else cost1
        return cost.sum()


class RetrievalModel(object):
    def __init__(self, opt, pretrained):
        self.grad_clip = opt.grad_clip

        # build models
        self.data1_enc = Encoder(opt.cnn_type, pretrained[0], opt.gpu_ids)
        self.data2_enc = Encoder(opt.cnn_type, pretrained[1], opt.gpu_ids)

        # criterion and optimizer
        self.criterion = TripletLoss(margin=opt.margin, inverse=opt.inverse)
        self.params1 = list(self.data1_enc.cnn.parameters())
        self.params2 = list(self.data2_enc.cnn.parameters())
        self.optimizer1 = torch.optim.AdamW(self.params1, lr=opt.learning_rate1, weight_decay=0.001)
        self.optimizer2 = torch.optim.AdamW(self.params2, lr=opt.learning_rate2, weight_decay=0.001)

    def state_dict(self):
        return [self.data1_enc.state_dict(), self.data2_enc.state_dict()]

    def load_state_dict(self, state_dict):
        self.data1_enc.load_state_dict(state_dict[0])
        self.data2_enc.load_state_dict(state_dict[1])

    def train_start(self):
        # train mode
        self.data1_enc.train()
        self.data2_enc.train()

    def val_start(self):
        # evaluate mode
        self.data1_enc.eval()
        self.data2_enc.eval()

    def forward_emb(self, data1, data2):
        # forward
        data1_emb = self.data1_enc(data1)
        data2_emb = self.data2_enc(data2)
        return data1_emb, data2_emb

    def forward_loss(self, data1_emb, data2_emb):
        # calculate loss
        loss = self.criterion(data1_emb, data2_emb)
        return loss

    def train_emb(self, data1_emb, data2_emb):
        # train
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        loss = self.forward_loss(data1_emb, data2_emb)
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params1, self.grad_clip)
            clip_grad_norm_(self.params2, self.grad_clip)
        self.optimizer1.step()
        self.optimizer2.step()
        return loss.item()



