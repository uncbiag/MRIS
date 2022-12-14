import os
import torch
import numpy as np
from monai.transforms import *
from matplotlib import pyplot as plt


def data_augmentation(data):
    img_sz = data.shape
    rotate = np.pi/12   # 15 degree
    rotate_range = (rotate, rotate) if len(img_sz) == 3 else (rotate, rotate, rotate)

    transforms = Compose(
        [RandAffine(prob=0.8, rotate_range=rotate_range, padding_mode='border',),
         RandGaussianNoise(prob=0.5),
         RandAdjustContrast(prob=0.5)])
    transformed_data = transforms(data)
    return transformed_data


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
