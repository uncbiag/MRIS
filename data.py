import os
import blosc
import torch
import random
import numpy as np
import pandas as pd
import progressbar as pb
from multiprocessing import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from preprocess import read_dicom, crop_image, resize_image, normalize, flip_image, data_augmentation


# img1: xray, img2: thickness
class OAIData:
    def __init__(self, opt, phase, transform=None, replace_nan=True):
        self.opt = opt
        self.phase = phase
        self.transform = transform
        self.replace_nan = replace_nan  # only for thickness map

        self.num_images = 2
        self.side = ['RIGHT', 'LEFT']
        self.month_dict = {'0': '00', '1': '12', '2': '18', '3': '24', '4': '30', '5': '36', '6': '48', '8': '72'}

        self.sheet_path = [os.path.join(opt.data_sheet_folder, p, opt.data_sheet_name) for p in self.phase] \
            if isinstance(phase, list) else os.path.join(opt.data_sheet_folder, phase, opt.data_sheet_name)
        self.df_kp = pd.concat([pd.read_csv(os.path.join(opt.data_sheet_folder, p, opt.kp_sheet_name), sep=',') for p in self.phase]) \
            if isinstance(phase, list) else pd.read_csv(os.path.join(opt.data_sheet_folder, phase, opt.kp_sheet_name), sep=',')

        self.image_path_dic = {}    # save all the image names (keys) and path (items)
        self.image_name_list = []  # save all the image name (keys)
        self.image_list = []    # save all the images that are loaded into memory

        self.get_data()
        if self.opt.load_into_memory:
            self.init_img_pool()
        print("finish loading data")

    def get_image_path(self, row):
        # xray
        image_path1 = os.path.join(self.opt.data_path1, row['xray'])
        if not os.path.exists(image_path1):
            return None, None

        # right & left side thickness
        image_path2 = []
        patient_id = str(row['patient_id'])
        month = 'ENROLLMENT' if row['month'] == '00_MONTH' else row['month']
        for s in self.side:
            path = [os.path.join(self.opt.data_path2, patient_id, "MR_SAG_3D_DESS", s+'_KNEE', month, "avsm/FC_2d_thickness.npy"),
                    os.path.join(self.opt.data_path2, patient_id, "MR_SAG_3D_DESS", s+'_KNEE', month, "avsm/TC_2d_thickness.npy")]
            # check if the path exist
            for p in path:
                if not os.path.exists(p):
                    return None, None
            image_path2.append(path)
        return image_path1, image_path2

    def load_name_path(self, patient_id, month, image_path1, image_path2, i):
        image_name = patient_id + '_' + self.side[i] + '_' + month
        self.image_path_dic[image_name] = {'img1': image_path1, 'img2': image_path2[i]}
        self.image_name_list.append(image_name)

    def load_image_dict(self, sheet):
        df = pd.read_csv(sheet, sep=',')
        for index, row in df.iterrows():
            if len(self.image_name_list) >= self.opt.num_load > 0:  # read until reach the max num_load
                return
            image_path1, image_path2 = self.get_image_path(row)
            if image_path1 is None or image_path2 is None:  # missing image
                continue
            patient_id = str(row['patient_id'])
            month = self.month_dict[row['xray'].split('.')[0]]
            for i in range(len(self.side)):
                self.load_name_path(patient_id, month, image_path1, image_path2, i)

    def get_data(self):
        if isinstance(self.sheet_path, list):
            for sp in self.sheet_path:
                self.load_image_dict(sp)
        else:
            self.load_image_dict(self.sheet_path)

    def split_dict(self, dict_to_split, split_num):
        split_dict = []
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj = dict(dict_to_split_items[index_split[i][0]:index_split[i][-1]+1])
            split_dict.append(dj)
        return split_dict

    def read_data_into_zipnp(self, image_path_dic, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_path_dic)).start()
        count = 0
        for name, image_path in image_path_dic.items():
            image_label_np_dic = {}
            all_img = self.read_data(image_path, name)
            for i in range(len(all_img)):
                image_label_np_dic[f'img{i + 1}'] = blosc.pack_array(all_img[i])
            img_dic[name] = image_label_np_dic
            count += 1
            pbar.update(count)
        pbar.finish()

    def load_image(self, img_dic, image_name):
        images = []
        for k in img_dic[image_name].keys():
            images.append(img_dic[image_name][k])
        self.image_list.append(images)

    def init_img_pool(self):
        manager = Manager()
        img_dic = manager.dict()
        split_dict = self.split_dict(self.image_path_dic, self.opt.workers)
        procs = []
        for i in range(self.opt.workers):
            p = Process(target=self.read_data_into_zipnp, args=(split_dict[i], img_dic))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} images have been loaded".format(len(img_dic)))
        for image_name in self.image_name_list:
            self.load_image(img_dic, image_name)

    def read_img1(self, image_path,  name, month):
        img1, spacing = read_dicom(image_path)
        roi_size_px = int(self.opt.roi_size_mm * 1. / spacing)
        img1 = crop_image(self.df_kp, img1, roi_size_px, name, month)
        img1 = resize_image(img1.astype(np.float32), target_size=self.opt.input_size1)
        img1 = normalize(img1, percentage_clip=99, zero_centered=False)
        if not self.opt.no_flip and 'RIGHT' in name:
            img1 = flip_image(img1)
        return img1

    def read_img2(self, image_path):
        img2 = np.load(image_path)
        if self.replace_nan:
            img2[np.isnan(img2)] = 0
        if not self.opt.no_norm:
            img2 = normalize(img2, max_value=3, zero_centered=False)
        img2 = resize_image(img2.astype(np.float32), target_size=self.opt.input_size2)
        return img2

    def read_data(self, image_path, name, month=None):
        all_img = []
        # x-ray
        img1 = self.read_img1(image_path['img1'] + '/001', name, month)
        all_img.append(img1)

        # MR-extracted thickness map
        img2_1 = self.read_img2(image_path['img2'][0])
        img2_2 = self.read_img2(image_path['img2'][1])
        if self.opt.region == 'all':
            img2 = np.concatenate((img2_1, img2_2), axis=0)
        else:
            img2 = img2_1 if self.opt.region == 'fc' else img2_2
        all_img.append(img2)

        return all_img

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_name = self.image_name_list[idx]
        sample = {}
        if not self.opt.load_into_memory:
            img_list = self.read_data(self.image_path_dic[image_name], image_name)
        else:
            img_list = [blosc.unpack_array(item) for item in self.image_list[idx]]

        sample['name'] = image_name
        for i in range(self.num_images):
            sample[f'img{i + 1}'] = img_list[i]
            if self.transform is not None:
                sample[f'img{i + 1}'] = self.transform(sample[f'img{i + 1}'])
        return sample


class OAIDataByPatient(OAIData):
    def __init__(self, opt, phase, transform=None, replace_nan=True):
        self.input_size2 = opt.input_size2.copy()
        if opt.region == 'all':
            self.input_size2[0] = opt.input_size2[0] * 2
        super(OAIDataByPatient, self).__init__(opt, phase, transform, replace_nan)

    def load_name_path(self, patient_id, month, image_path1, image_path2, i):
        image_name = patient_id + '_' + self.side[i]
        if image_name not in self.image_path_dic.keys():
            self.image_path_dic[image_name] = {month: {'img1': image_path1, 'img2': image_path2[i]}}
            self.image_name_list.append(image_name)
        else:
            self.image_path_dic[image_name].update({month: {'img1': image_path1, 'img2': image_path2[i]}})

    def read_data_into_zipnp(self, image_path_dic, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_path_dic)).start()
        count = 0
        for name, image_path in image_path_dic.items():
            image_label_np_dic = {}
            for month, path in image_path.items():
                all_img = self.read_data(path, name, month)
                image_label_np_dic[month] = {}
                for i in range(len(all_img)):
                    image_label_np_dic[month][f'img{i+1}'] = blosc.pack_array(all_img[i])

            img_dic[name] = image_label_np_dic
            count += 1
            pbar.update(count)
        pbar.finish()

    def load_image(self, img_dic, image_name):
        dic = {}
        for month in img_dic[image_name].keys():
            images = []
            for k in img_dic[image_name][month].keys():
                images.append(img_dic[image_name][month][k])
            dic[month] = images
        self.image_list.append(dic)

    def __getitem__(self, idx):
        image_name = self.image_name_list[idx]
        sample = {}
        image_dic = {}
        if not self.opt.load_into_memory:
            for month in list(self.image_path_dic[image_name].keys()):
                img_list = self.read_data(self.image_path_dic[image_name][month], image_name, month)
                image_dic[month] = img_list
        else:
            for month in list(self.image_list[idx].keys()):
                image_dic[month] = [blosc.unpack_array(item) for item in self.image_list[idx][month]]

        for month in list(self.month_dict.values()):
            if 'name' not in sample.keys():
                sample['name'] = {month: image_name + '_' + month}
                for i in range(self.num_images):
                    if month in list(self.image_path_dic[image_name].keys()):
                        sample[f'img{i+1}'] = {month: image_dic[month][i]}
                    else:
                        if i == 1:
                            sample['img2'] = {month: np.zeros(self.input_size2)}
                        else:
                            sample[f'img{i+1}'] = {month: np.zeros(getattr(self.opt, f'input_size{i+1}'))}
            else:
                sample['name'].update({month: image_name + '_' + month})
                for i in range(self.num_images):
                    if month in list(self.image_path_dic[image_name].keys()):
                        sample[f'img{i+1}'].update({month: image_dic[month][i]})
                    else:
                        if i == 1:
                            sample['img2'].update({month: np.zeros(self.input_size2)})
                        else:
                            sample[f'img{i+1}'].update({month: np.zeros(getattr(self.opt, f'input_size{i+1}'))})
            if self.transform is not None:
                for i in range(self.num_images):
                    sample[f'img{i+1}'][month] = self.transform(sample[f'img{i+1}'][month])
        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample.copy())
        return n_tensor


def get_loader(opt, phase, shuffle, replace_nan=True, drop_last=False):
    transform = transforms.Compose([ToTensor()])
    if opt.one_per_patient:
        dataset = OAIDataByPatient(opt, phase=phase, transform=transform, replace_nan=replace_nan)
    else:
        dataset = OAIData(opt, phase=phase, transform=transform, replace_nan=replace_nan)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, drop_last=drop_last)
    print("finish loading {} data".format(len(dataset)))
    return loader


def prepare_data(opt, loader, is_train=False):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] >= 0 else torch.device('cpu')

    img1 = loader['img1']
    img2 = loader['img2']
    name = loader['name']
    if opt.one_per_patient:
        bs = len(img1[list(img1.keys())[0]])
        valid_img1, valid_img2, valid_name = [[] for _ in range(bs)], [[] for _ in range(bs)], [[] for _ in range(bs)]
        data1, data2, data_name = [], [], []

        # remove the non-exist months
        for month in img1.keys():  # loop by patient_side
            for i in range(bs):  # img1[month] shape: [bs, H, W]
                if img1[month][i].max().item() > 0 and img2[month][i].max().item() > 0:
                    valid_img1[i].append(img1[month][i])
                    valid_img2[i].append(img2[month][i])
                    valid_name[i].append(name[month][i])

        # if train: sample one image per patient_side, else: use the first image of each patient_side
        for i in range(bs):
            idx = random.sample(range(len(valid_name[i])), 1)[0] if is_train else 0
            data1.append(valid_img1[i][idx])
            data2.append(valid_img2[i][idx])
            data_name.append(valid_name[i][idx])

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


