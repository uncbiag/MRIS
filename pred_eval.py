from __future__ import print_function
import torch
import numpy as np
import pandas as pd
import seaborn as sn
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

from setting import parse_args
from downstream_task.model import PredModel
from utils import data_augmentation
from data import ToTensor

from downstream_task.data import get_loader


def augmentation(opt, data, device):
    aug = transforms.Compose([
        iaa.Sequential([
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            iaa.Jigsaw(nb_rows=20, nb_cols=10)]).augment_image,
        ToTensor()
    ])
    if isinstance(data, list):
        data = torch.stack(data)
        data = data.transpose(0, 1)
        data = data.cpu().numpy()
        data = np.moveaxis(data, 1, -1)
        for i in range(len(data)):
            data[i] = aug(data[i])
        data = np.moveaxis(data, -1, 1)
        data = Variable(torch.from_numpy(data)).to(device).to(torch.float32)
        data = data.transpose(0, 1)
        data = [data[0], data[1]]
    else:
        if opt.modality == 1:
            data = data_augmentation(data)
        else:
            data = data.cpu().numpy()
            for i in range(len(data)):
                data[i] = np.squeeze(aug(np.expand_dims(data[i], -1)))
            data = Variable(torch.from_numpy(data)).to(device).to(torch.float32)
    return data


def prepare(opt, data, device, is_train):
    data = [Variable(data[i]).to(device).to(torch.float32) for i in range(len(data))] if isinstance(data, list) else \
        Variable(data).to(device).to(torch.float32)
    if opt.augmentation and is_train:
        data = augmentation(opt, data, device)
    # make the data to 3 channels to fit ResNet
    if isinstance(data, list):
        data = [data[i].unsqueeze(1) for i in range(len(data))]
        data = [data[i].repeat(1, 3, 1, 1) for i in range(len(data))]
    else:
        data = data.unsqueeze(1)
        data = data.repeat(1, 3, 1, 1)
    return data


def prepare_data(opt, loader, is_train=False):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    name = loader['name']
    if 'klg' in loader.keys():
        label = loader['klg']
        data = loader['img']
    if 'prog' in loader.keys():
        label = loader['prog']
        data = [loader['img'], loader['img_next']]
    data = prepare(opt, data, device, is_train)
    return data, name, Variable(label).to(device)


def predict(opt, data_loader, model, loss_weight):
    # switch to evaluate mode
    total_loss = 0
    y_true, y_pred = [], []
    model.val_start()
    with torch.no_grad():
        for loader in data_loader:
            data, name, label = prepare_data(opt, loader, is_train=False)
            pred = model.forward_pred(data)
            loss = model.forward_loss(pred, label, weight=loss_weight)
            total_loss += loss.item()
            pred = torch.nn.Softmax(dim=1)(pred)
            if opt.task == 2:
                label = torch.where(label == 2, 1, label)
                y_pred.extend(pred[:, 1:].sum(1).cpu().data.numpy())
            else:
                pred = pred.cpu().data.numpy()
                y_pred.extend(pred)
            y_true.extend(label.cpu().numpy())
    report = get_report(opt, y_true, y_pred)
    if opt.task == 2:
        y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)
    else:
        y_pred = np.argmax(y_pred, 1)
    cf_matrix = confusion_matrix(y_true, y_pred)
    return total_loss / len(data_loader.dataset), report, cf_matrix


def load_model():
    # Construct the model
    num_classes = 4 if opt.task == 1 else 2
    model = PredModel(opt, num_classes, pretrained=False)
    if opt.resume_pred is None:
        print('no checkpoint is being loaded')
        return
    checkpoint = torch.load(opt.resume_pred)
    start_epoch = checkpoint['epoch']
    best = checkpoint['best']
    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, best {})".format(opt.resume_pred, start_epoch, best))
    return model


def get_report(opt, y_true, y_pred):
    if opt.task == 1:
        roc = 0
        ap = 0
        y_pred = np.argmax(y_pred, 1)
    else:
        roc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred)
    report = {'acc': acc, 'roc': roc, 'ap': ap}
    return report


def draw_cf_matrix_heatmap(cf_matrix, save_name):
    df_cm = pd.DataFrame(cf_matrix, range(len(cf_matrix)), range(len(cf_matrix)))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    plt.savefig(save_name)


def pred_task():
    model = load_model()
    test_loader = get_loader(opt, phase='test_set', shuffle=False, k=topk)

    loss, report, cf_matrix = predict(opt, test_loader, model, loss_weight)
    print('confusion matrix:')
    print(cf_matrix)
    draw_cf_matrix_heatmap(cf_matrix, save_name='./cf_matrix.png')
    class_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
    print(f'class acc: {[round(class_acc[i], 4) for i in range(len(class_acc))]}')
    return loss, report


if __name__ == '__main__':
    # python evaluation.py --resume result/image_retrieval/model_best.pth.tar --cnn_type resnet18 --flip
    # -bs 128 -g 3 --load_into_memory --num_load 2000

    # use the top K retrieved data
    topk = 20
    loss_weight = 100

    opt = parse_args()
    _, report = pred_task()
    print("acc:", round(report['acc'], 4))
    if opt.task == 2:
        print("ap:", round(report['ap'], 4))
        print("roc auc:", round(report['roc'], 4))



