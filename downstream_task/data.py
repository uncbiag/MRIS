import os
import torch
import blosc
import numpy as np
import pandas as pd
import progressbar as pb
from ast import literal_eval
from multiprocessing import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from preprocess import read_dicom, read_smooth_thickness, crop_image, resize_image, normalize, flip_image
from utils import get_weight


# img1: xray, img2: thickness
class OAIData:
    def __init__(self, opt, phase, transform=None, replace_nan=True, k=20):  # extra can be klg or next timpoint
        self.phase = phase
        self.transform = transform
        self.replace_nan = replace_nan  # only for thickness map
        self.k = k

        self.data_root1 = opt.data_path1
        self.data_root2 = opt.data_path2
        self.resize1 = opt.input_size1
        self.resize2 = opt.input_size2
        self.flip = opt.flip    # flip all right knee to left knee for xray image
        self.pad = opt.pad
        self.roi_size_mm = opt.roi_size_mm
        self.num_load = opt.num_load
        self.num_of_workers = opt.workers
        self.load_into_memory = opt.load_into_memory
        self.task = opt.task
        self.modality = opt.modality

        self.df_klg, self.df_prog = None, None
        if self.task == 1:
            self.df_klg = pd.read_csv(os.path.join(opt.data_sheet_folder, 'data_sheets/KLG_score.csv'), sep=',')
        if self.task == 2:
            self.df_prog = pd.concat([pd.read_csv(os.path.join(opt.data_sheet_folder, p, 'progression_speed.csv'), sep=',') for p in self.phase]) \
                if isinstance(phase, list) else pd.read_csv(os.path.join(opt.data_sheet_folder, phase, 'progression_speed.csv'), sep=',')

        if self.modality == 3:  # for the synthesized thickness map
            phase_name = phase[0] if isinstance(phase, list) else phase
            if opt.resume is not None:
                retrieval_result = os.path.join(opt.resume.replace(opt.resume.split('/')[-1], ''),
                                                'retrieval_result/result_' + phase_name.split('_')[0] + '.csv')
                if not os.path.exists(retrieval_result):
                    print('please run retrieve_image.py first')
                    return
                self.df_retrieve = pd.read_csv(retrieval_result)
            else:
                retrieval_result_fc = os.path.join(opt.resume_fc.replace(opt.resume_fc.split('/')[-1], ''),
                                                   'retrieval_result/result_' + phase_name.split('_')[0] + '.csv')
                retrieval_result_tc = os.path.join(opt.resume_tc.replace(opt.resume_tc.split('/')[-1], ''),
                                                   'retrieval_result/result_' + phase_name.split('_')[0] + '.csv')
                if not os.path.exists(retrieval_result_fc) or not os.path.exists(retrieval_result_tc):
                    print('please first run retrieve_image.py first')
                    return
                self.df_retrieve = [pd.read_csv(retrieval_result_fc), pd.read_csv(retrieval_result_tc)]

        self.side = ['RIGHT', 'LEFT']
        self.month_dict = {'0': '00', '1': '12', '2': '18', '3': '24', '4': '30', '5': '36', '6': '48', '8': '72'}
        self.next_month_dict = {'00': '24', '12': '36'}

        self.sheet_path = [os.path.join(opt.data_sheet_folder, p, opt.data_sheet_name) for p in self.phase] \
            if isinstance(phase, list) else os.path.join(opt.data_sheet_folder, phase, opt.data_sheet_name)
        self.df_kp = pd.concat(
            [pd.read_csv(os.path.join(opt.data_sheet_folder, p, opt.kp_sheet_name), sep=',') for p in self.phase]) \
            if isinstance(phase, list) else pd.read_csv(os.path.join(opt.data_sheet_folder, phase, opt.kp_sheet_name), sep=',')

        self.image_path_dic = {}    # save all the image names (keys) and path (items)
        self.image_name_list = []  # save all the image name (keys)
        self.image_list = []    # save all the images that are loaded into memory
        self.image_klg_list = []   # save klg information
        self.image_prog_list = []   # save progression information

        self.get_data()
        if self.load_into_memory:
            self.init_img_pool()
        print("finish loading data")

    def get_image_path(self, row):
        image_path1 = os.path.join(self.data_root1, row['xray'])  # xray
        if not os.path.exists(image_path1):
            return None, None
        image_path2 = []  # right & left side thickness / DESS image
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

    def get_next_image_path(self, df, image_path1, image_path2):
        month_dict = {'0': '00_MONTH', '1': '12_MONTH'}
        next_month_dict = {'00_MONTH': '24_MONTH', 'ENROLLMENT': '24_MONTH', '12_MONTH': '36_MONTH'}

        patient_id = image_path1.split('/')[-3]
        month = image_path1.split('.')[0].split('/')[-1]
        if month not in list(month_dict.keys()):
            return None, None
        row = df.loc[(df.patient_id == int(patient_id)) & (df.month == next_month_dict[month_dict[month]])]
        if len(row) == 0:
            return None, None
        image_path1_next = os.path.join(self.data_root1, getattr(row, 'xray').values[0])

        current_month = image_path2[0][0].split('/')[-3]
        next_month = next_month_dict[current_month]
        image_path2_next = [[image_path2[i][j].replace(current_month, next_month) for j in range(len(image_path2[i]))] for i in range(len(image_path2))]
        for path in image_path2_next:
            for p in path:
                if not os.path.exists(p):
                    return None, None
        return image_path1_next, image_path2_next

    def match_klg(self, image_name):
        klg_dict = {'00': 'V00XRKL', '12': 'V01XRKL', '24': 'V03XRKL', '36': 'V05XRKL',
                    '48': 'V06XRKL', '72': 'V08XRKL', '96': 'V10XRKL'}
        side_dict = {'RIGHT': 1, 'LEFT': 2}
        name_splits = image_name.split('_')
        patient_id, side = name_splits[:2]
        row = self.df_klg.loc[(self.df_klg.ID == int(patient_id)) & (self.df_klg.SIDE == side_dict[side])]
        if len(row) == 0:   # patient_side not found
            return -1

        month = name_splits[-1]
        klg = np.nanmax(getattr(row, klg_dict[month]).values)   # in case multiple rows
        if np.isnan(klg):   # no klg
            return -1
        if int(klg) > 0:
            return int(klg) - 1
        return int(klg)

    def match_progression(self, image_name):
        row = self.df_prog.loc[self.df_prog.Name == image_name]
        if len(row) == 0:
            return -1
        prog = getattr(row, 'Speed').values[0]
        prog = 1 if prog > 0 else 0
        return prog

    def load_image_dict(self, sheet):
        df = pd.read_csv(sheet, sep=',')
        for index, row in df.iterrows():
            if len(self.image_name_list) >= self.num_load > 0:  # read until reach the max num_load
                return
            image_path1, image_path2 = self.get_image_path(row)
            if image_path1 is None or image_path2 is None:     # missing image
                continue
            if self.task == 2:
                image_path1_next, image_path2_next = self.get_next_image_path(df, image_path1, image_path2)
                if image_path1_next is None:
                    continue
            patient_id = str(row['patient_id'])
            month = self.month_dict[row['xray'].split('.')[0]]
            for i in range(len(self.side)):
                image_name = patient_id + '_' + self.side[i] + '_' + month
                if self.task == 1:
                    klg = self.match_klg(image_name)
                    if klg == -1:  # extra information is invaild
                        continue
                    self.image_path_dic[image_name] = {'img1': image_path1, 'img2': image_path2[i]}
                    self.image_klg_list.append(klg)
                if self.task == 2:
                    prog = self.match_progression(image_name)
                    if prog == -1:  # extra information is invaild
                        continue
                    self.image_path_dic[image_name] = {'img1': image_path1, 'img2': image_path2[i],
                                                       'img1_next': image_path1_next, 'img2_next': image_path2_next[i]}
                    self.image_prog_list.append(prog)
                self.image_name_list.append(image_name)

    def get_data(self):
        if 'val' in self.phase and self.num_load > 0 and not isinstance(self.phase, list):
            self.num_load = self.num_load // 5 if self.num_load > 5 else self.num_load
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
            if self.task == 2:
                self.image_list.append([img_dic[image_name]['img'], img_dic[image_name]['img_next']])
            else:
                self.image_list.append([img_dic[image_name]['img']])

    def read_img1(self, path, name):
        img1, spacing = read_dicom(path)
        roi_size_px = int(self.roi_size_mm * 1. / spacing)
        img1 = crop_image(self.df_kp, img1, roi_size_px, name, month=None)
        img1 = resize_image(img1.astype(np.float32), target_size=self.resize1)
        img1 = normalize(img1, percentage_clip=99, zero_centered=False)
        if self.flip and 'RIGHT' in name:
            img1 = flip_image(img1)
        return img1

    def read_img2(self, path):
        img2_1 = np.load(path[0])
        img2_2 = np.load(path[1])
        if self.replace_nan:
            img2_1[np.isnan(img2_1)] = 0
            img2_2[np.isnan(img2_2)] = 0
        img2 = np.concatenate((img2_1, img2_2), axis=0)
        img2 = resize_image(img2.astype(np.float32), target_size=self.resize2)
        return img2

    def read_img3_from_file(self, df, name, smooth):
        row = df.loc[df.Name == name]
        if len(row) == 0:
            return None
        topk_names = literal_eval(row['TopK_Name'].item())[:self.k]
        topk_distance = literal_eval(row['TopK_Distance'].item())[:self.k]
        weights = get_weight(topk_distance, True)
        fc_maps, tc_maps = read_smooth_thickness(self.data_root2, topk_names, smooth=smooth)
        synthesis_fc_map = np.sum([weights[i] * fc_maps[i] for i in range(len(fc_maps))], axis=0)
        synthesis_tc_map = np.sum([weights[i] * tc_maps[i] for i in range(len(tc_maps))], axis=0)
        return synthesis_fc_map, synthesis_tc_map

    def read_img3(self, name, smooth):
        if isinstance(self.df_retrieve, list):
            synthesis_fc_map, _ = self.read_img3_from_file(self.df_retrieve[0], name, smooth)
            _, synthesis_tc_map = self.read_img3_from_file(self.df_retrieve[1], name, smooth)
        else:
            synthesis_fc_map, synthesis_tc_map = self.read_img3_from_file(self.df_retrieve, name, smooth)
        if synthesis_fc_map is None or synthesis_tc_map is None:
            return None
        return np.concatenate((synthesis_fc_map, synthesis_tc_map))

    def read_data(self, image_path, name):
        if self.modality == 1:
            img = self.read_img1(image_path['img1'] + '/001', name)
        elif self.modality == 2:
            img = self.read_img2(image_path['img2'])
        else:   # retrieve thickness map
            img = self.read_img3(name, smooth=False)

        if self.task == 2:
            name_next = name[:-2] + self.next_month_dict[name.split('_')[-1]]
            if self.modality == 1:
                img_next = self.read_img1(image_path['img1_next'] + '/001', name_next)
            elif self.modality == 2:
                img_next = self.read_img2(image_path['img2_next'])
            else:
                img_next = self.read_img3(name_next, smooth=False)
            return img, img_next
        return img

    def read_data_into_zipnp(self, image_path_dic, img_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_path_dic)).start()
        count = 0
        for name, image_path in image_path_dic.items():
            image_label_np_dic = {}
            if self.task == 2:
                img, img_next = self.read_data(image_path, name)
                image_label_np_dic['img_next'] = blosc.pack_array(img_next)
            else:
                img = self.read_data(image_path, name)
            image_label_np_dic['img'] = blosc.pack_array(img)

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
        if self.df_klg is not None:
            klg = self.image_klg_list[idx]
        if self.df_prog is not None:
            prog = self.image_prog_list[idx]

        if not self.load_into_memory:
            if self.task == 2:
                img, img_next = self.read_data(self.image_path_dic[image_name], image_name)
                img_list = [img, img_next]
            else:
                img = self.read_data(self.image_path_dic[image_name], image_name)
                img_list = [img]
        else:
            img_list = [blosc.unpack_array(item) for item in self.image_list[idx]]

        sample = {'img': img_list[0]}
        sample['name'] = image_name
        if self.task == 1:
            sample['klg'] = klg
        if self.task == 2:
            sample['img_next'] = img_list[1]
            sample['prog'] = prog

        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
            if self.task == 2:
                sample['img_next'] = self.transform(sample['img_next'])

        return sample


class ToTensor(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample.copy())
        return n_tensor


def get_loader(opt, phase, shuffle, replace_nan=True, k=20):
    transform = transforms.Compose([ToTensor()])
    dataset = OAIData(opt, phase=phase, transform=transform, replace_nan=replace_nan, k=k)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle)
    print("finish loading {} data".format(len(dataset)))
    return loader


