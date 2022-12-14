import glob
import os
import pickle
from functools import partial

import cv2
import numpy as np
import torch
from deeppipeline.common.normalization import normalize_channel_wise
from deeppipeline.common.transforms import apply_by_index
from solt import core as slc, transforms as slt
from torchvision import transforms as tvt

from kneel.data.utils import read_dicom, process_xray
from kneel.inference import NFoldInferenceModel, wrap_slt, unwrap_slt
from kneel.model import init_model_from_args


class LandmarkAnnotator(object):
    def __init__(self, snapshot_path, mean_std_path, device='cpu', jit_trace=True):
        self.fold_snapshots = glob.glob(os.path.join(snapshot_path, 'fold_*.pth'))
        models = []
        self.device = device
        with open(os.path.join(snapshot_path, 'session.pkl'), 'rb') as f:
            snapshot_session = pickle.load(f)
        snp_args = snapshot_session['args'][0]

        for snp_name in self.fold_snapshots:
            net = init_model_from_args(snp_args)
            snp = torch.load(snp_name, map_location=device)['model']
            net.load_state_dict(snp)
            models.append(net.eval())

        self.net = NFoldInferenceModel(models).to(self.device)
        self.net.eval()
        if jit_trace:
            dummy = torch.FloatTensor(2, 3, snp_args.crop_x, snp_args.crop_y).to(device=self.device)
            with torch.no_grad():
                self.net = torch.jit.trace(self.net, dummy)
        mean_vector, std_vector = np.load(mean_std_path)

        self.annotator_type = snp_args.annotations
        self.img_spacing = getattr(snp_args, f'{snp_args.annotations}_spacing')

        norm_trf = partial(normalize_channel_wise, mean=mean_vector, std=std_vector)
        norm_trf = partial(apply_by_index, transform=norm_trf, idx=[0, 1])

        self.trf = tvt.Compose([
            partial(wrap_slt, annotator_type=self.annotator_type),
            slc.Stream([
                slt.PadTransform((snp_args.pad_x, snp_args.pad_y), padding='z'),
                slt.CropTransform((snp_args.crop_x, snp_args.crop_y), crop_mode='c'),
            ]),
            partial(unwrap_slt, norm_trf=norm_trf),
        ])

    @staticmethod
    def pad_img(img, pad):
        if pad is not None:
            if not isinstance(pad, tuple):
                pad = (pad, pad)
            row, col = img.shape
            tmp = np.zeros((row + 2 * pad[0], col + 2 * pad[1]))
            tmp[pad[0]:pad[0] + row, pad[1]:pad[1] + col] = img
            return tmp
        else:
            return img

    @staticmethod
    def read_dicom(img_path, new_spacing, return_orig=False, pad_img=None):
        res = read_dicom(img_path)
        if res is None:
            return []
        img_orig, orig_spacing, _ = res
        img_orig = process_xray(img_orig).astype(np.uint8)
        img_orig = LandmarkAnnotator.pad_img(img_orig, pad_img)

        h_orig, w_orig = img_orig.shape

        img = LandmarkAnnotator.resize_to_spacing(img_orig, orig_spacing, new_spacing)

        if return_orig:
            return img, orig_spacing, h_orig, w_orig, img_orig
        return img, orig_spacing, h_orig, w_orig

    @staticmethod
    def resize_to_spacing(img, spacing, new_spacing):
        if new_spacing is None:
            return img
        scale = spacing / new_spacing
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    def predict_img(self, img, h_orig=None, w_orig=None, rounded=True):
        img_batch = self.trf(img)
        res = self.batch_inference(img_batch).squeeze()

        res = self.handle_lc_out(res, h_orig, w_orig)
        if rounded:
            return np.round(res).astype(int)
        return res

    @staticmethod
    def handle_lc_out(res, h_orig, w_orig):
        # right preds
        res[0, 0] = (w_orig // 2 + w_orig % 2) * res[0, 0]
        res[0, 1] = h_orig * res[0, 1]

        # left preds
        res[1, 0] = w_orig // 2 + w_orig // 2 * res[1, 0]
        res[1, 1] = h_orig * res[1, 1]

        return res

    def batch_inference(self, batch: torch.tensor):
        if batch.device != self.device:
            batch = batch.to(self.device)
            with torch.no_grad():
                res = self.net(batch)
        return res.to('cpu').numpy()
