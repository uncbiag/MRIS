import os
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np


class Encoder(nn.Module):
    def __init__(self, task, num_classes, cnn_type='resnet18', pretrained=True):
        """Load pretrained resnet18 and replace top fc layer."""
        super(Encoder, self).__init__()
        self.task = task

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, pretrained)
        # for progression prediction
        finetune_layers = ['layer2', 'layer3', 'layer4']
        for name, p in self.cnn.named_parameters():
            p.requires_grad = False
            for l in finetune_layers:
                if l in name:
                    p.requires_grad = True
                    break
        # for KLG prediction
        # for name, p in self.cnn.named_parameters():
        #     p.requires_grad = True

        self.fc1 = nn.Linear(512, num_classes)
        if self.task == 2:
            self.fc_combine = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        self.cnn.fc = nn.Sequential()
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        # Load pretrained CNN and parallelize over GPUs
        os.environ['TORCH_HOME'] = '/playpen-raid/bqchen/code/OAIRetrieval/models'
        if pretrained:
            print("=> using pre-trained model '{}' on ImageNet".format(arch))
        else:
            print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=pretrained)
        # model = nn.DataParallel(model)
        return model

    def init_weights(self):
        # Xavier initialization for the fully connected layer
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features + self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)
        if self.task == 2:
            r = np.sqrt(6.) / np.sqrt(self.fc_combine.in_features + self.fc_combine.out_features)
            self.fc_combine.weight.data.uniform_(-r, r)
            self.fc_combine.bias.data.fill_(0)

    def forward(self, image):
        if self.task == 1:     # klg prediction
            feature = self.cnn(image)
        if self.task == 2:   # progression prediction
            feature1 = self.cnn(image[0])
            feature2 = self.cnn(image[1])
            feature = torch.cat([feature1, feature2], dim=1)
            feature = self.dropout(F.relu(self.fc_combine(feature)))
        cls1 = self.fc1(feature)
        return cls1


class PredModel(object):
    def __init__(self, opt, num_classes, class_weight=None, pretrained=True):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.num_classes = num_classes
        self.model = Encoder(opt.task, self.num_classes, opt.cnn_type,  pretrained=pretrained)
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.model.to(opt.gpu_ids[0])
            self.model = nn.DataParallel(self.model, opt.gpu_ids)

        # Loss and Optimizer
        self.criterion = torch.nn.CrossEntropyLoss() if class_weight is None else torch.nn.CrossEntropyLoss(weight=class_weight)
        params = list(self.model.module.fc1.parameters()) + list(self.model.module.cnn.parameters())
        if opt.task == 2:
            params += list(self.model.module.fc_combine.parameters())
        self.params = params
        self.optimizer = torch.optim.AdamW(params, lr=opt.learning_rate1, weight_decay=0.1)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train_start(self):
        # train mode
        self.model.train()

    def val_start(self):
        # evaluate mode
        self.model.eval()

    def forward_pred(self, data):
        # forward
        pred = self.model(data)
        return pred

    def forward_loss(self, pred, label, weight):
        # calculate loss
        return weight * self.criterion(pred, label)

    def train_pred(self, data, label, weight):
        # train
        pred = self.forward_pred(data)
        self.optimizer.zero_grad()
        pred = torch.nn.Softmax(dim=1)(pred)
        loss = self.forward_loss(pred, label, weight)
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        return loss, pred

