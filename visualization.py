import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from ast import literal_eval
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from setting import parse_args
from preprocess import read_smooth_thickness, read_thickness, read_erode_mask, read_dicom, resize_image, normalize, flip_image


def save_image(image, save_name):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(save_name)
    plt.clf()


def save_seg(seg, save_name):
    plt.imshow(seg, cmap='Reds')
    plt.axis('off')
    plt.savefig(save_name)
    plt.clf()


def save_thickness(thickness, save_name):
    plt.imshow(thickness)
    plt.axis('off')
    plt.savefig(save_name)
    plt.clf()


def plot_mri(image_name):
    patient_id, side, month = image_name.split('_')
    if month == '00':
        image_folder = os.path.join(opt.data_path2, patient_id, 'MR_SAG_3D_DESS', side + '_KNEE', 'ENROLLMENT')
    else:
        image_folder = os.path.join(opt.data_path2, patient_id, 'MR_SAG_3D_DESS', side + '_KNEE', month + '_MONTH')
    image_path = os.path.join(image_folder, 'image_preprocessed.nii.gz')
    image = sitk.ReadImage(image_path)
    np_image = sitk.GetArrayFromImage(image)
    image_shape = np_image.shape

    slice_x = np_image[image_shape[0] // 2, :, :]
    slice_y = np_image[:, image_shape[1] // 2, :]
    slice_z = np_image[:, :, image_shape[2] // 2]
    save_image(slice_x, os.path.join(save_folder, image_name, 'mri_slice_x.png'))
    save_image(slice_y, os.path.join(save_folder, image_name, 'mri_slice_y.png'))
    save_image(slice_z, os.path.join(save_folder, image_name, 'mri_slice_z.png'))

    # segmentation
    probmap_fc = sitk.ReadImage(os.path.join(image_folder, 'FC_probmap.nii.gz'))
    nda_probmap_fc = sitk.GetArrayFromImage(probmap_fc)
    segmentation_fc = np.where((nda_probmap_fc > 0.5) & (nda_probmap_fc <= 1), 1.0, 0.0)
    probmap_tc = sitk.ReadImage(os.path.join(image_folder, 'TC_probmap.nii.gz'))
    nda_probmap_tc = sitk.GetArrayFromImage(probmap_tc)
    segmentation_tc = np.where((nda_probmap_tc > 0.5) & (nda_probmap_tc <= 1), 1.0, 0.0)
    segmentation = np.where(segmentation_tc == 1.0, 1.0, segmentation_fc)

    slice_x = segmentation[image_shape[0] // 2, :, :]
    slice_y = segmentation[:, image_shape[1] // 2, :]
    slice_z = segmentation[:, :, image_shape[2] // 2]
    save_seg(slice_x, os.path.join(save_folder, image_name, 'mri_slice_xseg.png'))
    save_seg(slice_y, os.path.join(save_folder, image_name, 'mri_slice_yseg.png'))
    save_seg(slice_z, os.path.join(save_folder, image_name, 'mri_slice_zseg.png'))


def plot_xray_roi(image_name, xray_path, kp, side):
    img, spacing = read_dicom(os.path.join(opt.data_path1, xray_path + '/001'))
    roi_size_px = int(opt.roi_size_mm * 1. / spacing)
    s = roi_size_px // 2
    roi = img[kp[1] - s:kp[1] + s, kp[0] - s:kp[0] + s]
    # roi = resize_image(roi.astype(np.float32), target_size=opt.input_size1)
    # roi = normalize(roi, percentage_clip=99, zero_centered=False)
    # if opt.flip and side == 'right':
    #     roi = flip_image(roi)
    save_image(roi, os.path.join(save_folder, image_name, 'xray.png'))

    # plt.plot(kp[0], kp[1], marker='o', color="red")
    # plt.gca().add_patch(Rectangle((kp[0]-s, kp[1]-s), roi_size_px, roi_size_px, edgecolor='red', facecolor='none', lw=2))
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.savefig(os.path.join(save_folder, 'xray.png'))
    # plt.clf()


def plot_thickness(image_name):
    fc, tc = read_smooth_thickness(opt.data_path2, image_name, smooth, replace_nan)
    if opt.region == 'fc':
        img = fc
        mask = thickness_mask[0]
    elif opt.region == 'tc':
        img = tc
        mask = thickness_mask[1]
    else:
        img = np.concatenate((fc, tc), axis=0)
        mask = np.concatenate((thickness_mask[0], thickness_mask[1]))

    if smooth:
        save_thickness(img, os.path.join(save_folder, image_name, 'smooth_thickness.png'))
    else:
        save_thickness(img, os.path.join(save_folder, image_name, 'thickness.png'))

    if erode:
        img = np.where(mask == 0, np.nan, img)
        save_thickness(img, os.path.join(save_folder, image_name, 'erode_thickness.png'))


def plot_synthesis_thickness(df, name, type='fc'):
    row = df.loc[df.Name == name]
    if len(row) == 0:
        return None, None
    # top k thickness err
    topk_name = literal_eval(row['TopK_Name'].item())[:topk]
    weights = literal_eval(row['TopK_Distance'].item())[:topk]
    fc_maps, tc_maps = read_thickness(opt.data_path2, topk_name, replace_nan=True)
    if type == 'fc':
        synthesis_map = np.sum([weights[i] * fc_maps[i] for i in range(len(fc_maps))], axis=0)
        return synthesis_map, fc_maps
    else:
        synthesis_map = np.sum([weights[i] * tc_maps[i] for i in range(len(tc_maps))], axis=0)
        return synthesis_map, tc_maps


def plot_all_images(image_name, row, row_kp, side):
    # plot original xray image
    xray_path = row['xray'].item()
    kp = literal_eval(row_kp['keypoint_' + side].item())
    plot_xray_roi(image_name, xray_path, kp, side)

    # plot MRI image & seg
    plot_mri(image_name)

    # plot thickness map
    plot_thickness(image_name)

    # plot synthesised map
    synthesis_fc_map, retrieved_fc_maps = plot_synthesis_thickness(df_fc, image_name, type='fc')
    synthesis_tc_map, retrieved_tc_maps = plot_synthesis_thickness(df_tc, image_name, type='tc')
    if synthesis_fc_map is not None and synthesis_tc_map is not None:
        synthesis_map = np.concatenate((synthesis_fc_map, synthesis_tc_map))
        save_thickness(synthesis_map, os.path.join(save_folder, image_name, 'synthesis_thickness.png'))
        for i in range(len(retrieved_fc_maps)):
            retrieved_map = np.concatenate((retrieved_fc_maps[i], retrieved_tc_maps[i]))
            save_thickness(retrieved_map, os.path.join(save_folder, image_name, 'retrieved_thickness' + str(i+1) + '.png'))


if __name__ == '__main__':
    # python visualization.py --save_name result/visualize --resume result/
    # --criterion 9757487_LEFT_12 9722580_LEFT_24 9933459_LEFT_24 9319367_RIGHT_12

    topk = 20
    smooth = False
    erode = False
    replace_nan = True

    opt = parse_args()
    save_folder = opt.save_name
    criterion = opt.criterion

    if opt.resume is None:
        result_file_fc = os.path.join(opt.resume_fc, 'retrieval_result/result_test.csv')
        result_file_tc = os.path.join(opt.resume_tc, 'retrieval_result/result_test.csv')
    else:
        result_file_fc = result_file_tc = os.path.join(opt.resume, 'retrieval_result/result_test.csv')
    df_fc = pd.read_csv(result_file_fc, sep=',')
    df_tc = pd.read_csv(result_file_tc, sep=',')

    thickness_mask = read_erode_mask()

    # find the matching row
    for name in criterion:
        patient_id, side, month = name.split('_')
        month = month + '_MONTH'
        side = 'left' if side == 'LEFT' else 'right'
        os.makedirs(os.path.join(save_folder, name), exist_ok=True)

        for set in ['train_set1', 'train_set2', 'test_set']:
            sheet_name = os.path.join(opt.data_sheet_folder, set, opt.data_sheet_name)
            if os.path.exists(sheet_name):  # for test_set
                df = pd.read_csv(sheet_name, sep=',')
                row = df.loc[(df.patient_id == int(patient_id)) & (df.month == month)]
                if len(row) == 0:
                    continue
                df_kp = pd.read_csv(os.path.join(opt.data_sheet_folder, set, opt.kp_sheet_name), sep=',')
                row_kp = df_kp.loc[(df_kp.patient_id == int(patient_id)) & (df_kp.month == month)]
                plot_all_images(name, row, row_kp, side)
            else:
                for i in range(5):  # for train_set
                    sheet_name = os.path.join(opt.data_sheet_folder, set, 'fold' + str(i+1), opt.data_sheet_name)
                    df = pd.read_csv(sheet_name, sep=',')
                    row = df.loc[(df.patient_id == int(patient_id)) & (df.month == month)]
                    if len(row) == 0:
                        continue
                    df_kp = pd.read_csv(os.path.join(opt.data_sheet_folder, set, 'fold' + str(i+1), opt.kp_sheet_name), sep=',')
                    row_kp = df_kp.loc[(df_kp.patient_id == int(patient_id)) & (df_kp.month == month)]
                    plot_all_images(name, row, row_kp, side)


