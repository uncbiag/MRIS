import os
import numpy as np
import pydicom as dicom
from ast import literal_eval
from monai.transforms import *
from skimage.transform import resize


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def dicom_img_spacing(data):
    spacing = None
    for spacing_param in ["Imager Pixel Spacing", "ImagerPixelSpacing", "PixelSpacing", "Pixel Spacing"]:
        if hasattr(data, spacing_param):
            spacing_attr_value = getattr(data, spacing_param)
            if isinstance(spacing_attr_value, str):
                if isfloat(spacing_attr_value):
                    spacing = float(spacing_attr_value)
                else:
                    spacing = float(spacing_attr_value.split()[0])
            elif isinstance(spacing_attr_value, dicom.multival.MultiValue):
                if len(spacing_attr_value) != 2:
                    return None
                spacing = list(map(lambda x: float(x), spacing_attr_value))[0]
            elif isinstance(spacing_attr_value, float):
                spacing = spacing_attr_value
        else:
            continue
        if spacing is not None:
            break
    return spacing


def read_dicom(filename):
    data = dicom.read_file(filename)
    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)
    if data.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    img = img.reshape((data.Rows, data.Columns))
    spacing = dicom_img_spacing(data)
    return img, spacing


def name2folder(name):
    # find the image folder given image name
    patient_id, side, month = name.split('_')
    month = 'ENROLLMENT' if month == '00' else month+'_MONTH'
    folder = os.path.join(patient_id, 'MR_SAG_3D_DESS', side+'_KNEE', month)
    return folder


def read_thickness(data_folder, data_name, replace_nan=True):
    # read the thickness map given image name
    if not isinstance(data_name, list):      # only read one thickness map
        data_name = [data_name]

    all_fc_map, all_tc_map = [], []
    for name in data_name:
        folder = name2folder(name)
        fc_map = np.load(os.path.join(data_folder, folder, 'avsm/FC_2d_thickness.npy'))
        tc_map = np.load(os.path.join(data_folder, folder, 'avsm/TC_2d_thickness.npy'))
        if replace_nan:
            fc_map[np.isnan(fc_map)] = 0
            tc_map[np.isnan(tc_map)] = 0
        all_fc_map.append(fc_map)
        all_tc_map.append(tc_map)

    if len(data_name) == 1:
        all_fc_map = all_fc_map[0]
        all_tc_map = all_tc_map[0]

    return all_fc_map, all_tc_map


def normalize(img, percentage_clip=-1, max_value=-1, zero_centered=False):
    # normalize into [0, 1] if not zero_centered else [-1, 1]
    img = img - img.min()
    if percentage_clip > 0:
        norm_img = img / np.percentile(img, percentage_clip) * (percentage_clip/100)
    if max_value > 0:
        norm_img = img / (max_value - img.min())
    if zero_centered:
        norm_img = norm_img * 2 - 1
    return norm_img


def crop_image(df, img, roi_size_pix, name, month=None):
    # crop the xray image according to the keypoints detected by kneel
    patient_id = name.split('_')[0]
    month = month + '_MONTH' if month is not None else name.split('_')[-1] + '_MONTH'
    side = 'keypoint_right' if name.split('_')[1] == 'RIGHT' else 'keypoint_left'

    s = roi_size_pix // 2
    if isinstance(df, list):
        for d in df:
            row = d.loc[(d["patient_id"] == int(patient_id)) & (d["month"] == month)]
            if len(row) > 0:
                coords = getattr(row, side).values[0]
                break
    else:
        row = df.loc[(df["patient_id"] == int(patient_id)) & (df["month"] == month)]
        coords = getattr(row, side).values[0]
    coords = literal_eval(coords)
    roi = img[coords[1] - s:coords[1] + s, coords[0] - s:coords[0] + s]
    return roi


def resize_image(img, target_size, ignore_nan=False):
    if ignore_nan:
        mask = np.zeros_like(img)
        mask[np.isnan(img)] = 1
        mask = resize(mask, target_size)
        img[np.isnan(img)] = 0
        img = resize(img, target_size)
        img = np.where(mask > 0.5, np.nan, img)
    else:
        img = resize(img, target_size)
    return img


def flip_image(img):
    img = np.fliplr(img)
    return img


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


