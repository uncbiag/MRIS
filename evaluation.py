import torch
import random
import numpy as np
from torch.autograd import Variable

import data
from setting import parse_args
from model import RetrievalModel
from utils import data_augmentation


def prepare_data(opt, loader, is_train=False):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    img1 = loader['img1']
    img2 = loader['img2']
    name = loader['name']
    if opt.one_per_patient:
        bs = len(img1[list(img1.keys())[0]])
        data1 = [[] for _ in range(bs)]
        data2 = [[] for _ in range(bs)]
        data_name = [[] for _ in range(bs)]

        # remove the non-exist months
        for month in img1.keys():  # loop by patient_side
            for i in range(len(img1[month])):  # img1[month] shape: [bs, H, W]
                if img1[month][i].max().item() > 0 and img2[month][i].max().item() > 0:
                    data1[i].append(img1[month][i])
                    data2[i].append(img2[month][i])
                    data_name[i].append(name[month][i])

        # if train: sample one image per patient_side, else: use the first image of each patient_side
        for i in range(bs):
            idx = random.sample(range(len(data1[i])), 1)[0] if is_train else 0
            data1[i] = data1[i][idx]
            data2[i] = data2[i][idx]
            data_name[i] = data_name[i][idx]

        data1 = Variable(torch.stack(data1).to(torch.float32)).to(device)
        data2 = Variable(torch.stack(data2).to(torch.float32)).to(device)
    else:
        data1 = Variable(img1).to(device)
        data2 = Variable(img2).to(device)
        data_name = name

    if opt.augmentation and is_train:
        data1 = data_augmentation(data1)

    # make the data to 3 channels to fit ResNet
    data1 = data1.unsqueeze(1) if len(data1.shape) == (len(opt.input_size1) + 1) else data1
    data2 = data2.unsqueeze(1) if len(data2.shape) == (len(opt.input_size2) + 1) else data2
    data1 = data1.repeat(1, 3, 1, 1) if len(opt.input_size1) == 2 else data1
    data2 = data2.repeat(1, 3, 1, 1) if len(opt.input_size2) == 2 else data2
    return data1, data2, data_name


def encode_data(opt, model, data_loader, is_train=False):
    # switch to evaluate mode
    model.val_start()
    total_loss = 0
    # save all the embeddings and names
    data1_embs, data2_embs, data_names = [], [], []
    with torch.no_grad():
        for loader in data_loader:
            data1, data2, name = prepare_data(opt, loader, is_train)
            data1_emb, data2_emb = model.forward_emb(data1, data2)
            total_loss += model.forward_loss(data1_emb, data2_emb).item()

            data1_embs.extend(data1_emb.data.cpu().numpy().copy())
            data2_embs.extend(data2_emb.data.cpu().numpy().copy())
            data_names.extend(name)
    return np.array(data1_embs), np.array(data2_embs), np.array(data_names), total_loss / len(data_loader.dataset)


def retrieve_data(data1, data2, topk=[1, 5, 10]):
    npts = int(data1.shape[0])
    ranks = np.zeros(npts)
    for index in range(npts):
        im = data1[index].reshape(1, data1.shape[1])    # the query image
        d = np.dot(im, data2.T).flatten()    # cosine distance
        inds = np.argsort(d)[::-1]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank

    topk_recall = []
    for k in topk:
        topk_recall.append(100.0 * len(np.where(ranks < k)[0]) / len(ranks))
    median_recall = np.floor(np.median(ranks)) + 1
    mean_recall = ranks.mean() + 1
    return topk_recall, median_recall, mean_recall


def load_model(opt):
    model = RetrievalModel(opt, pretrained=[False, False])
    if opt.resume is None:
        print("no checkpoint is being loaded")
        return
    checkpoint = torch.load(opt.resume)
    start_epoch = checkpoint['epoch']
    best = checkpoint['best']
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint '{}' (epoch {}, best recall {})".format(opt.resume, start_epoch, best))
    return model


def get_embeddings():
    model = load_model(opt)
    test_loader = data.get_loader(opt, phase='test_set', shuffle=False)
    data1_embs, data2_embs, data_names, _ = encode_data(opt, model, test_loader, is_train=False)
    return data_names, data1_embs, data2_embs


if __name__ == '__main__':
    # python evaluation.py --resume result/model_best.pth.tar --cnn_type resnet18 --flip
    # -bs 128 -g 3 --region fc --input_size2 310 310 --one_per_patient --load_into_memory
    opt = parse_args()
    topk = [1, 5, 10, 50, 100]
    _, data1_embs, data2_embs = get_embeddings()
    topk_recall, median_recall, mean_recall = retrieve_data(data1_embs, data2_embs, topk=topk)
    print("data1 to data2 recall:")
    for i in range(len(topk)):
        print("top {}: {:.2f}".format(topk[i], topk_recall[i]))
    print("median recall: {:.2f}".format(median_recall))
    print("mean recall: {:.2f}".format(mean_recall))



