import time
import torch
import numpy as np
from tqdm import tqdm

import data
from setting import parse_args
from model import RetrievalModel
from utils import plot_curve, save_checkpoint
from evaluation import retrieve_data, encode_data


def main():
    # Hyper Parameters
    opt = parse_args()
    print(opt)

    # set seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Load data loaders
    train_loader = data.get_loader(opt, phase=['train_set/fold1', 'train_set/fold2', 'train_set/fold3', 'train_set/fold4'], shuffle=True, drop_last=True)
    val_loader = data.get_loader(opt, phase='train_set/fold5', shuffle=False, drop_last=True)

    # Construct the model
    model = RetrievalModel(opt, pretrained=[True, True])

    # record loss, recall
    train_losses, val_losses = [], []
    val_recall = [[] for _ in range(3)]    # for top 1, 5, 10

    # start training
    best = 0
    start = time.time()
    for epoch in range(opt.num_epochs):
        print("Epoch: {}/{}.. lr1: {}.. lr2: {}.. ".format(epoch + 1, opt.num_epochs, model.optimizer1.param_groups[0]['lr'], model.optimizer2.param_groups[0]['lr']))

        # train for one epoch & evaluation
        train_loss = train(opt, train_loader, model)
        val_loss, topk_recall = validate(opt, val_loader, model)
        print("Train Loss: {:.4f}.. Val Loss: {:.4f}.. ".format(train_loss, val_loss))
        print("Val Recall @ 1, 5, 10: {}..".format([round(topk_recall[i], 4) for i in range(len(topk_recall))]))

        # remember best topk_recall and save checkpoint
        recall_sum = sum(topk_recall)
        is_best = recall_sum > best
        best = max(recall_sum, best)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best': best,
            'opt': opt,
        }, is_best, prefix=opt.save_name + '/')

        # record the loss, acc
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for i in range(len(topk_recall)):
            val_recall[i].append(topk_recall[i])
        plot_curve(curves=[[train_losses, val_losses], val_recall],
                   labels=[['training loss', 'validation loss'], ['top1 recall', 'top5 recall', 'top10 recall']],
                   save_folder=opt.save_name,
                   save_names=["loss_curve.png", "recall_curve.png"])

    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print("finish training in: {:d}:{:02d}:{:02d}".format(int(h), int(m), int(s)))


def train(opt, train_loader, model):
    total_loss = 0
    model.train_start()
    pbar = tqdm(train_loader)
    for i, loader in enumerate(pbar):
        data1, data2, data_name = data.prepare_data(opt, loader, is_train=True)
        data1_emb, data2_emb = model.forward_emb(data1, data2)
        total_loss += model.train_emb(data1_emb, data2_emb)
    model.scheduler1.step()
    model.scheduler2.step()
    return total_loss / len(train_loader.dataset)


def validate(opt, val_loader, model):
    data1_embs, data2_embs, _, loss = encode_data(opt, model, val_loader, is_train=False)
    topk_recall = retrieve_data(data1_embs, data2_embs)
    return loss, topk_recall


if __name__ == '__main__':
    main()
