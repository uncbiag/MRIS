import os
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from setting import parse_args
from matplotlib import pyplot as plt
from kneel.inference.pipeline import LandmarkAnnotator


def detect_keypoint(phase):
    df = pd.read_csv(os.path.join(main_folder, phase, 'mri_xray.csv'))
    all_rows = []
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        month = row['month']
        image_path = os.path.join(image_folder, row['xray'] + '/001')
        res = global_searcher.read_dicom(image_path, new_spacing=global_searcher.img_spacing, return_orig=True)
        if len(res) > 0:
            img, orig_spacing, h_orig, w_orig, img_orig = res
        else:
            return None
        global_coords = global_searcher.predict_img(img, h_orig, w_orig)

        all_rows.append([patient_id, month, global_coords[0].tolist(), global_coords[1].tolist()])
    return all_rows


def localize_left_right_rois(img, roi_size_pix, coords):
    s = roi_size_pix // 2

    roi_right = img[coords[0, 1] - s:coords[0, 1] + s,
                coords[0, 0] - s:coords[0, 0] + s]

    roi_left = img[coords[1, 1] - s:coords[1, 1] + s,
               coords[1, 0] - s:coords[1, 0] + s]

    return roi_right, roi_left


def plot_knee(patient_id, month):
    path = df_path[(df_path["patient_id"] == patient_id) & (df_path["month"] == month)].iloc[0]['xray']
    path = os.path.join(image_folder, path + '/001')

    keypoint_right = literal_eval(row['keypoint_right'])
    keypoint_left = literal_eval(row['keypoint_left'])
    global_coords = np.array([keypoint_right, keypoint_left])

    res = global_searcher.read_dicom(path, new_spacing=global_searcher.img_spacing, return_orig=True)
    if len(res) > 0:
        img, orig_spacing, h_orig, w_orig, img_orig = res
    else:
        return

    img_orig = LandmarkAnnotator.pad_img(img_orig, opt.pad if opt.pad != 0 else None)
    global_coords += opt.pad
    roi_size_px = int(opt.roi_size_mm * 1. / orig_spacing)
    right_roi_orig, left_roi_orig = localize_left_right_rois(img_orig, roi_size_px, global_coords)

    plt.figure()
    plt.suptitle('right & left knee')
    plt.subplot(1, 2, 1)
    plt.imshow(right_roi_orig)
    plt.subplot(1, 2, 2)
    plt.imshow(left_roi_orig)
    plt.show()


if __name__ == '__main__':
    opt = parse_args()
    main_folder = opt.data_sheet_folder
    image_folder = opt.data_path1
    lc_snapshot_path = '/playpen-raid/bqchen/code/OAIRetrieval/kneel/pretrained_model/lext-devbox_2019_07_14_16_04_41'
    mean_std_path = '/playpen-raid/bqchen/code/OAIRetrieval/kneel/pretrained_model/mean_std.npy'
    phase_name = ['train_set1/fold1', 'train_set1/fold2', 'train_set1/fold3', 'train_set1/fold4', 'train_set1/fold5',
                  'train_set2/fold1', 'train_set2/fold2', 'train_set2/fold3', 'train_set2/fold4', 'train_set2/fold5',
                  'test_set']

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.gpu_id >= 0:
        device = torch.device("cuda")
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    global_searcher = LandmarkAnnotator(snapshot_path=lc_snapshot_path,
                                        mean_std_path=mean_std_path,
                                        device=device, jit_trace=True)
    for phase in phase_name:
        print("begin {} set".format(phase))
        rows = detect_keypoint(phase)
        sheet_name = os.path.join(main_folder, phase, 'xray_keypoints.csv')
        pd.DataFrame(np.array(rows), columns=['patient_id', 'month', 'keypoint_right', 'keypoint_left']).to_csv(sheet_name, index=0)

    # # visualize result
    # criterion = '9014209_RIGHT_00'
    # for phase in phase_name:
    #     df_keypoint = pd.read_csv(os.path.join(main_folder, phase, 'xray_keypoints.csv'))
    #     df_path = pd.read_csv(os.path.join(main_folder, phase, 'mri_xray.csv'))
    #     all_rows = []
    #     if criterion is None:
    #         for _, row in df_keypoint.iterrows():
    #             patient_id = row['patient_id']
    #             month = row['month']
    #             plot_knee(patient_id, month)
    #     else:
    #         patient_id = int(criterion.split('_')[0])
    #         month = criterion.split('_')[-1] + '_MONTH'
    #         plot_knee(patient_id, month)




