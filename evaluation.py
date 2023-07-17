import torch
import numpy as np

import data
from setting import parse_args
from model import RetrievalModel


def encode_data(opt, model, data_loader, is_train=False):
    total_loss = 0
    model.val_start()
    data1_embs, data2_embs, data_names = [], [], []
    with torch.no_grad():
        for loader in data_loader:
            data1, data2, name = data.prepare_data(opt, loader, is_train)
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
        img = data1[index].reshape(1, data1.shape[1])    # the query image
        dist = np.dot(img, data2.T).flatten()    # cosine distance
        inds = np.argsort(dist)[::-1]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank

    topk_recall = []
    for k in topk:
        topk_recall.append(100.0 * len(np.where(ranks < k)[0]) / len(ranks))
    return topk_recall


def load_model(opt):
    model = RetrievalModel(opt, pretrained=[False, False])
    if opt.ckp_path is None:
        print("no checkpoint is being loaded")
        return
    checkpoint = torch.load(opt.ckp_path)
    start_epoch = checkpoint['epoch']
    best = checkpoint['best']
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint '{}' (epoch {}, best recall {})".format(opt.ckp_path, start_epoch, best))
    return model


def get_embeddings():
    model = load_model(opt)
    test_loader = data.get_loader(opt, phase='test_set', shuffle=False)
    data1_embs, data2_embs, data_names, _ = encode_data(opt, model, test_loader, is_train=False)
    return data_names, data1_embs, data2_embs


if __name__ == '__main__':
    opt = parse_args()
    topk = [1, 5, 10, 20]
    _, data1_embs, data2_embs = get_embeddings()
    topk_recall = retrieve_data(data1_embs, data2_embs, topk=topk)
    print("data1 to data2 recall:")
    for i in range(len(topk)):
        print("top {}: {:.2f}".format(topk[i], topk_recall[i]))




