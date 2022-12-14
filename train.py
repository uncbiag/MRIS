import time
import torch
import shutil
import numpy as np

import data
from setting import parse_args
from model import RetrievalModel
from utils import plot_curve
from evaluation import retrieve_data, encode_data, prepare_data


def main():
    # Hyper Parameters
    opt = parse_args()
    print(opt)

    # set seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Load data loaders
    train_loader = data.get_loader(opt, phase=['train_set1/fold1', 'train_set1/fold2', 'train_set1/fold3', 'train_set1/fold4'], shuffle=True)
    val_loader = data.get_loader(opt, phase='train_set1/fold5', shuffle=False)

    # Construct the model
    model = RetrievalModel(opt, pretrained=[True, True])

    # record loss, recall
    train_losses, val_losses = [], []
    val_recall = [[] for _ in range(3)]    # for top 1, 5, 10

    # start training
    best = 0
    start = time.time()
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt.learning_rate1, opt.lr_update, model.optimizer1, epoch)
        adjust_learning_rate(opt.learning_rate2, opt.lr_update, model.optimizer2, epoch)

        print("Epoch: {}/{}.. lr1: {}.. lr2: {}.. ".format(epoch + 1, opt.num_epochs,
                                                           model.optimizer1.param_groups[0]['lr'],
                                                           model.optimizer2.param_groups[0]['lr']))

        # train for one epoch & evaluation
        train_loss = train(opt, train_loader, model)
        val_loss, topk_recall, mean_recall = validate(opt, val_loader, model)
        print("Train Loss: {:.4f}.. Val Loss: {:.4f}.. ".format(train_loss, val_loss))
        print("Val Recall @ 1, 5, 10: {}.. Mean Recall: {:.4f}".format([round(topk_recall[i], 4) for i in range(len(topk_recall))], mean_recall))

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
                   labels=[['training loss', 'validation loss'],
                           ['top1 recall', 'top5 recall', 'top10 recall']],
                   save_folder=opt.save_name,
                   save_names=["loss_curve.png", "recall_curve.png"])

    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print("finish training in: {:d}:{:02d}:{:02d}".format(int(h), int(m), int(s)))


def train(opt, train_loader, model):
    total_loss = 0
    # switch to train mode
    model.train_start()
    for loader in train_loader:
        data1, data2, data_name = prepare_data(opt, loader, is_train=True)
        data1_emb, data2_emb = model.forward_emb(data1, data2)
        total_loss += model.train_emb(data1_emb, data2_emb)
    return total_loss / len(train_loader.dataset)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    data1_embs, data2_embs, _, loss = encode_data(opt, model, val_loader, is_train=False)
    # data1 retrieve data2
    topk_recall, _, mean_recall = retrieve_data(data1_embs, data2_embs)
    return loss, topk_recall, mean_recall


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        print("better model")
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(init_lr, lr_update, optimizer, epoch):
    """learning rate decays by 5 every lr_update epochs"""
    if lr_update > 0:
        lr = init_lr * (0.2 ** (epoch // lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    # python train.py --save_name result --cnn_type resnet18 --augmentation --flip
    # -bs 64 -lr1 1e-04 -lr2 1e-05 -e 450 --lr_update 150 -g 2 3 --region fc --input_size2 310 310
    # --one_per_patient --load_into_memory
    main()
