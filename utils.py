import os
import shutil
import torch
import numpy as np
from matplotlib import pyplot as plt


def set_device(gpu_ids):
    has_cuda = gpu_ids[0] >= 0
    if has_cuda:
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
    return device


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        print("better model")
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def get_weight(distances, use_weighted):    # calculate weights according to the distance value
    return [d/sum(distances) for d in distances] if use_weighted else np.ones(len(distances)) / len(distances)


def l2_norm(emb):
    norm = torch.pow(emb, 2).sum(dim=1, keepdim=True).sqrt()
    emb = torch.div(emb, norm)
    return emb


def cosine_sim(emb1, emb2):     # cosine similarity
    return emb1.mm(emb2.t())


def plot_curve(curves, labels, save_folder, save_names):    # plot and save the curves for accuracy and loss
    assert len(curves) == len(labels)
    assert len(curves) == len(save_names)
    for i in range(len(curves)):
        plt.figure(i+1)
        for j in range(len(curves[i])):
            if len(curves[i][j]) > 0:
                plt.plot(curves[i][j], label=labels[i][j])
        plt.legend(frameon=False)
        plt.savefig(os.path.join(save_folder, save_names[i]))
        plt.clf()


def plot_image(images, title='image', save_name=None):
    num_images = len(images)
    plt.figure()
    plt.suptitle(title)
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i], vmin=0, vmax=4)
        plt.colorbar()
        plt.axis('off')
    plt.savefig(save_name)
    plt.clf()

