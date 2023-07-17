import os
import torch
import torch.nn.init
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils import cosine_sim, l2_norm


class Encoder(nn.Module):
    def __init__(self, cnn_type='resnet18', pretrained=True):
        super(Encoder, self).__init__()
        self.cnn = self.get_cnn(cnn_type, pretrained)
        for name, p in self.cnn.named_parameters():
            p.requires_grad = True
        self.cnn.fc = nn.Sequential()

    def get_cnn(self, arch, pretrained):
        if pretrained:
            print("using pre-trained model '{}' on ImageNet".format(arch))
        else:
            print("creating model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=pretrained)
        return model

    def forward(self, img):
        features = self.cnn(img)
        features = l2_norm(features)
        return features


class TripletLoss(nn.Module):
    # adaptive: use adaptive margin, pseudo_list != None: use pseudo positive labeling
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, emb1, emb2):
        scores = self.sim(emb1, emb2)
        diagonal1 = scores.diag()
        d1 = diagonal1.view(emb1.size(0), 1).expand_as(scores)
        cost = (self.margin + scores - d1).clamp(min=0)
        return cost.sum()


class RetrievalModel(object):
    def __init__(self, opt, pretrained):
        self.grad_clip = opt.grad_clip
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] >= 0 else torch.device('cpu')

        # build models
        self.data1_enc = Encoder(opt.cnn_type, pretrained[0])
        self.data2_enc = Encoder(opt.cnn_type, pretrained[1])
        self.data1_enc.to(device)
        self.data2_enc.to(device)
        if len(opt.gpu_ids) > 1:
            self.data1_enc = nn.DataParallel(self.data1_enc, opt.gpu_ids)
            self.data2_enc = nn.DataParallel(self.data2_enc, opt.gpu_ids)

        # criterion and optimizer
        self.criterion = TripletLoss(margin=opt.margin)
        self.params1 = list(self.data1_enc.parameters())
        self.params2 = list(self.data2_enc.parameters())
        self.optimizer1 = torch.optim.AdamW(self.params1, lr=opt.learning_rate1, weight_decay=0.001)
        self.optimizer2 = torch.optim.AdamW(self.params2, lr=opt.learning_rate2, weight_decay=0.001)
        self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer1, step_size=opt.lr_update, gamma=0.2)
        self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=opt.lr_update, gamma=0.2)

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



