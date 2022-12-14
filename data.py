import os
import torch
import blosc
import numpy as np
import pandas as pd
import progressbar as pb
from multiprocessing import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from preprocess import read_dicom, crop_image, resize_image, normalize, flip_image


# img1: xray, img2: thickness
class OAIData:
    def __init__(self, opt, phase, transform=None, replace_nan=True):
        self.phase = phase
        self.transform = transform
        self.replace_nan = replace_nan  # only for thickness map

        self.data_root1 = opt.data_path1
        self.data_root2 = opt.data_path2
        self.resize1 = opt.input_size1
        self.resize2 = opt.input_size2
        self.flip = opt.flip    # flip all right knee to left knee for xray image
        self.region = opt.region  # all / fc / tc
        self.one_per_patient = opt.one_per_patient
        self.pad = opt.pad
        self.roi_size_mm = opt.roi_size_mm
        self.num_load = opt.num_load
        self.num_of_workers = opt.workers
        self.load_into_memory = opt.load_into_memory

        self.side = ['RIGHT', 'LEFT']
        self.month_dict = {'0': '00', '1': '12', '2': '18', '3': '24', '4': '30', '5': '36', '6': '48', '8': '72'}

        self.sheet_path = [os.path.join(opt.data_sheet_folder, p, opt.data_sheet_name) for p in self.phase] \
            if isinstance(phase, list) else os.path.join(opt.data_sheet_folder, phase, opt.data_sheet_name)
        self.df_kp = pd.concat(
            [pd.read_csv(os.path.join(opt.data_sheet_folder, p, opt.kp_sheet_name), sep=',') for p in self.phase]) \
            if isinstance(phase, list) else pd.read_csv(os.path.join(opt.data_sheet_folder, phase, opt.kp_sheet_name), sep=',')

        self.image_path_dic = {}    # save all the image names (keys) and path (items)
        self.image_name_list = []  # save all the image name (keys)
        self.image_list = []    # save all the images that are loaded into memory

        self.get_data()
        if self.load_into_memory:
            self.init_img_pool()
        print("finish loading data")

    def get_image_path(self, row):
        image_path1 = os.path.join(self.data_root1, row['xray'])  # xray
        if not os.path.exists(image_path1):
            return None, None
        image_path2 = []  # right & left side thickness
        patient_id = str(row['patient_id'])
        month = 'ENROLLMENT' if row['month'] == '00_MONTH' else row['month']
        for s in self.side:
            path = [os.path.join(self.data_root2, patient_id, "MR_SAG_3D_DESS", s+'_KNEE', month, "avsm/FC_2d_thickness.npy"),
                    os.path.join(self.data_root2, patient_id, "MR_SAG_3D_DESS", s+'_KNEE', month, "avsm/TC_2d_thickness.npy")]
            # check if the path exist
            for p in path:
                if not os.path.exists(p):
                    return None, None
            image_path2.append(path)
        return image_path1, image_path2

    def load_image_dict(self, sheet):
        df = pd.read_csv(sheet, sep=',')
        for index, row in df.iterrows():
            if len(self.image_name_list) >= self.num_load > 0:  # read until reach the max num_load
                return
            image_path1, image_path2 = self.get_image_path(row)
            if image_path1 is None or image_path2 is None:     # missing image
                continue
            patient_id = str(row['patient_id'])
            month = self.month_dict[row['xray'].split('.')[0]]
            for i in range(len(self.side)):
                image_name = patient_id + '_' + self.side[i] if self.one_per_patient else patient_id + '_' + self.side[i] + '_' + month
                if self.one_per_patient:
                    if image_name not in self.image_path_dic.keys():
                        self.image_path_dic[image_name] = {month: {'img1': image_path1, 'img2': image_path2[i]}}
                        self.image_name_list.append(image_name)
                    else:
                        self.image_path_dic[image_name].update({month: {'img1': image_path1, 'img2': image_path2[i]}})
                else:
                    self.image_path_dic[image_name] = {'img1': image_path1, 'img2': image_path2[i]}
                    self.image_name_list.append(image_name)

    def get_data(self):
        if isinstance(self.sheet_path, list):
            for sp in self.sheet_path:
                self.load_image_dict(sp)
        else:
            self.load_image_dict(self.sheet_path)

    def init_img_pool(self):
        manager = Manager()
        img_dic = manager.dict()
        split_dict = self.split_dict(self.image_path_dic, self.num_of_workers)
        procs = []
        for i in range(self.num_of_workers):
            p = Process(target=self.read_data_into_zipnp, args=(split_dict[i], img_dic))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print("the loading phase finished, total {} images have been loaded".format(len(img_dic)))
        for image_name in self.image_name_list:
            if self.one_per_patient:
                image_dic = {}
                for month in img_dic[image_name].keys():
                    image_dic[month] = [img_dic[image_name][month]['img1'], img_dic[image_name][month]['img2']]
                self.image_list.append(image_dic)
            else:
                self.image_list.append([img_dic[image_name]['img1'], img_dic[image_name]['img2']])

    def read_data(self, image_path, name, month=None):
        img1, spacing = read_dicom(image_path['img1'] + '/001')
        roi_size_px = int(self.roi_size_mm * 1. / spacing)
        img1 = crop_image(self.df_kp, img1, roi_size_px, name, month) if self.one_per_patient \
            else crop_image(self.df_kp, img1, roi_size_px, name, month=None)
        img1 = resize_image(img1.astype(np.float32), target_size=self.resize1)
        img1 = normalize(img1, percentage_clip=99, zero_centered=False)
        if self.flip and 'RIGHT' in name:
            img1 = flip_image(img1)

        img2_1 = np.load(image_path['img2'][0])
        img2_2 = np.load(image_path['img2'][1])
        if self.replace_nan:
            img2_1[np.isnan(img2_1)] = 0
            img2_2[np.isnan(img2_2)] = 0
        if self.region == 'fc':
            img2 = img2_1
        elif self.region == 'tc':
            img2 = img2_2
        else:
            img2 = np.concatenate((img2_1, img2_2), axis=0)
        img2 = resize_image(img2.astype(np.float32), target_size=self.resize2)
        return img1, img2

    def read_data_into_zipnp(self, image_path_dic, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_path_dic)).start()
        count = 0
        for name, image_path in image_path_dic.items():
            image_label_np_dic = {}
            if self.one_per_patient:
                for month, path in image_path.items():
                    img1, img2 = self.read_data(path, name, month)
                    image_label_np_dic[month] = {'img1': blosc.pack_array(img1)}
                    image_label_np_dic[month].update({'img2': blosc.pack_array(img2)})
            else:
                img1, img2 = self.read_data(image_path, name)
                image_label_np_dic['img1'] = blosc.pack_array(img1)
                image_label_np_dic['img2'] = blosc.pack_array(img2)

            img_dic[name] = image_label_np_dic
            count += 1
            pbar.update(count)
        pbar.finish()

    def split_dict(self, dict_to_split, split_num):
        split_dict = []
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        dict_to_split_items = list(dict_to_split.items())
        for i in range(split_num):
            dj = dict(dict_to_split_items[index_split[i][0]:index_split[i][-1]+1])
            split_dict.append(dj)
        return split_dict

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_name = self.image_name_list[idx]
        sample = {}
        if self.one_per_patient:
            if not self.load_into_memory:
                image_dic = {}
                for month in list(self.image_path_dic[image_name].keys()):
                    img1, img2 = self.read_data(self.image_path_dic[image_name][month], image_name, month)
                    image_dic[month] = [img1, img2]
            else:
                image_dic = {}
                for month in list(self.image_list[idx].keys()):
                    image_dic[month] = [blosc.unpack_array(item) for item in self.image_list[idx][month]]

            for month in list(self.month_dict.values()):
                if 'img1' not in sample.keys():
                    sample['name'] = {month: image_name + '_' + month}
                    if month in list(self.image_path_dic[image_name].keys()):
                        sample['img1'] = {month: image_dic[month][0]}
                        sample['img2'] = {month: image_dic[month][1]}
                    else:
                        sample['img1'] = {month: np.zeros(self.resize1)}
                        sample['img2'] = {month: np.zeros(self.resize2)}
                else:
                    sample['name'].update({month: image_name + '_' + month})
                    if month in list(self.image_path_dic[image_name].keys()):
                        sample['img1'].update({month: image_dic[month][0]})
                        sample['img2'].update({month: image_dic[month][1]})
                    else:
                        sample['img1'].update({month: np.zeros(self.resize1)})
                        sample['img2'].update({month: np.zeros(self.resize2)})

                if self.transform is not None:
                    sample['img1'][month] = self.transform(sample['img1'][month])
                    sample['img2'][month] = self.transform(sample['img2'][month])
        else:
            if not self.load_into_memory:
                img1, img2 = self.read_data(self.image_path_dic[image_name], image_name)
                img_list = [img1, img2]
            else:
                img_list = [blosc.unpack_array(item) for item in self.image_list[idx]]

            sample = {'img1': img_list[0]}
            sample['img2'] = img_list[1]
            sample['name'] = image_name

            if self.transform is not None:
                sample['img1'] = self.transform(sample['img1'])
                sample['img2'] = self.transform(sample['img2'])

        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample.copy())
        return n_tensor


def get_loader(opt, phase, shuffle, replace_nan=True):
    transform = transforms.Compose([ToTensor()])
    dataset = OAIData(opt, phase=phase, transform=transform, replace_nan=replace_nan)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle)
    print("finish loading {} data".format(len(dataset)))
    return loader


