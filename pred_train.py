# compare klg prediction between using 1) xray 2) gt thickness map 3) retrieved thickness map
import time
import torch
import shutil
import numpy as np

from setting import parse_args
from utils import plot_curve
from downstream_task.data import get_loader
from downstream_task.model import PredModel
from pred_eval import prepare_data, predict, get_report
from sklearn.utils.class_weight import compute_class_weight


def main():
    learning_rate = opt.learning_rate2
    num_classes = 4 if opt.task == 1 else 2

    # Load data loaders
    train_loader = get_loader(opt, phase=['train_set2/fold1', 'train_set2/fold2', 'train_set2/fold3', 'train_set2/fold4'], shuffle=True, k=topk)
    val_loader = get_loader(opt, phase='train_set2/fold5', shuffle=False, k=topk)

    class_weight = calculate_weights(train_loader, num_classes)     # to count for data imbalance

    # Construct the model
    model = PredModel(opt, num_classes, class_weight=class_weight, pretrained=True)

    # record loss, acc
    all_train_loss, all_val_loss = [], []
    all_train_report = {'acc': [], 'roc': [],  'ap': []}
    all_val_report = {'acc': [], 'roc': [],  'ap': []}
    # start training
    best_measures = ['roc', 'acc']
    best = np.zeros(len(best_measures))
    start = time.time()
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(learning_rate, opt.lr_update, model.optimizer, epoch)

        print(f"Epoch: {epoch + 1}/{opt.num_epochs}.. "
              f"lr: {model.optimizer.param_groups[0]['lr']}.. ")

        # train for one epoch & evaluation
        train_loss, train_report = train(train_loader, model)
        val_loss, val_report, cf_matrix = validate(val_loader, model)
        print(f"Train Loss: {round(train_loss, 4)}.. "
              f"Val Loss: {round(val_loss, 4)}.. ")
        print(f"Train Acc: {round(train_report['acc'], 4)}.. "
              f"Val Acc: {round(val_report['acc'], 4)}.. ")
        if opt.task == 2:
            print(f"Train ROC AUC: {round(train_report['roc'], 4)}.. "
                  f"Val ROC AUC: {round(val_report['roc'], 4)}.. ")
            print(f"Train AP: {round(train_report['ap'], 4)}.. "
                  f"Val AP: {round(val_report['ap'], 4)}")
        class_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
        print(f'Val Class Acc: {[round(class_acc[i], 4) for i in range(len(class_acc))]}')

        # remember best and save checkpoint
        for i in range(len(best_measures)):
            value = val_report[best_measures[i]]
            is_best = value > best[i]
            best[i] = max(value, best[i])
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best': best[i],
                'opt': opt,
            }, is_best, prefix=opt.save_name + '/' + best_measures[i] + '_')

        # record the loss, acc
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        for k in list(all_train_report.keys()):
            all_train_report[k].append(train_report[k])
            all_val_report[k].append(val_report[k])

        # plot the curves
        all_curves = [[all_train_loss, all_val_loss]]
        all_labels = [['Train loss', 'Val loss']]
        all_save_names = ['loss_curve.png']
        for name in all_train_report.keys():
            all_curves.append([all_train_report[name], all_val_report[name]])
            all_labels.append(['Train ' + name, 'Val ' + name])
            all_save_names.append(name + '_curve.png')
        plot_curve(curves=all_curves, labels=all_labels, save_folder=opt.save_name, save_names=all_save_names)

    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print('finish training in: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))


def calculate_weights(train_loader, num_classes):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    all_label = []
    for loader in train_loader:
        data, _, label = prepare_data(opt, loader, is_train=False)
        all_label.extend(label.cpu().numpy())

    all_label = np.array(all_label)
    for j in range(num_classes):
        print(len(np.argwhere(all_label == j)), end=' ')
    weight = compute_class_weight('balanced', classes=np.unique(all_label), y=all_label)
    weight = [round(w, 2) for w in weight]
    print("\nclass weight:", weight)
    weight = torch.tensor(weight).to(device).to(torch.float32)
    return weight


def train(train_loader, model):
    total_loss = 0
    # switch to train mode
    model.train_start()
    y_true, y_pred = [], []
    for loader in train_loader:
        data, _, label = prepare_data(opt, loader, is_train=True)

        loss, pred = model.train_pred(data, label, weight=loss_weight)
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
    return total_loss / len(train_loader.dataset), report


def validate(val_loader, model):
    loss, report, cf_matrix = predict(opt, val_loader, model, loss_weight)
    return loss, report, cf_matrix


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
    # python pred_train.py --save_name result/ --cnn_type resnet18 --augmentation --flip
    # -bs 64 -lr1 1e-04 -lr2 1e-04 -e 30 --lr_update 10 -g 3 --load_into_memory --region all
    # --task 2 --modality 2 --num_load 2000

    # use the top K retrieved data
    topk = 20
    loss_weight = 100

    # Hyper Parameters
    opt = parse_args()
    print(opt)

    # set seed and device
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # task: 1: klg prediction, 2: progression prediction
    # modality: 1: xray, 2: GT thickness map, 3: retrieved thickness map
    main()
