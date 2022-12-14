import os
import random
import numpy as np
import pandas as pd
from setting import parse_args
from pre_analysis.progression_analysis import get_progression_speed


def get_patient_month_image():
    patient_dict = {}
    for name in data_sheet_names:   # loop each data sheet
        df = pd.read_csv(os.path.join(data_sheet_folder, name+'.csv'), sep=',')
        for _, row in df.iterrows():
            if row['SeriesDescription'] in list(descr_dict.keys()):   # is mri or xray
                image_type = descr_dict[row['SeriesDescription']]
                patient_id = str(row['ParticipantID'])
                image_folder = row['Folder']
                month = row['StudyDescription'].split('^')[2].replace(' ', '_')
                if month == 'SCREENING' or month == 'ENROLLMENT':
                    month = '00_MONTH'
                if patient_id + '_' + month in bad_image_list:      # do not count the bad thickness maps
                    continue
                # check path exists
                if image_type == 'xray':
                    if not os.path.exists(os.path.join(opt.data_path1, image_folder)):
                        continue
                else:
                    thickness_month = 'ENROLLMENT' if month == '00_MONTH' else month
                    fc_right_path = os.path.join(opt.data_path2, patient_id, "MR_SAG_3D_DESS", 'RIGHT_KNEE',
                                                 thickness_month, "avsm/FC_2d_thickness.npy")
                    tc_right_path = os.path.join(opt.data_path2, patient_id, "MR_SAG_3D_DESS", 'RIGHT_KNEE',
                                                 thickness_month, "avsm/TC_2d_thickness.npy")
                    fc_left_path = os.path.join(opt.data_path2, patient_id, "MR_SAG_3D_DESS", 'LEFT_KNEE',
                                                thickness_month, "avsm/FC_2d_thickness.npy")
                    tc_left_path = os.path.join(opt.data_path2, patient_id, "MR_SAG_3D_DESS", 'LEFT_KNEE',
                                                thickness_month,  "avsm/TC_2d_thickness.npy")

                    if (not os.path.exists(fc_right_path)) or (not os.path.exists(tc_right_path)) or \
                            (not os.path.exists(fc_left_path)) or (not os.path.exists(tc_left_path)):
                        continue
                if patient_id not in patient_dict.keys():
                    patient_dict[patient_id] = {month: {image_type: image_folder}}
                else:
                    if month not in patient_dict[patient_id].keys():
                        patient_dict[patient_id].update({month: {image_type: image_folder}})
                    else:
                        patient_dict[patient_id][month].update({image_type: image_folder})
    all_patient_ids = list(patient_dict.keys())
    for patient_id in all_patient_ids:
        for month in list(patient_dict[patient_id]):
            if len(patient_dict[patient_id][month]) < 3:
                patient_dict[patient_id].pop(month)
        if len(patient_dict[patient_id]) == 0:
            patient_dict.pop(patient_id)
    return patient_dict


def get_row_by_patient(dict, all_patient):
    rows = []
    image_type = list(descr_dict.values())
    for p in all_patient:
        months = list(dict[p].keys())
        for m in months:
            if len(dict[p][m]) == len(list(descr_dict.keys())):
                rows.append([p, m, dict[p][m][image_type[0]], dict[p][m][image_type[1]], dict[p][m][image_type[2]]])
    return rows


def match_split_images(dict, splits):
    all_rows = []
    for i in range(len(splits)):
        if isinstance(splits[i][0], list):
            rows = []
            for j in range(len(splits[i])):
                rows.append(get_row_by_patient(dict, splits[i][j]))
        else:
            rows = get_row_by_patient(dict, splits[i])
        all_rows.append(rows)
    columns = ['patient_id', 'month', 'mri_left', 'mri_right', 'xray']
    return all_rows, columns


def get_bad_images():
    bad_image_list = []
    for name in bad_sheet_names:   # loop each data sheet
        df = pd.read_csv(os.path.join(data_sheet_folder, name+'.csv'), sep=',')
        for _, row in df.iterrows():
            patient_id = str(int(row['patient_id']))
            month = str(int(row['timepoint'])) + '_MONTH'
            month = '00_MONTH' if month == '0_MONTH' else month
            bad_image_list.append(patient_id + '_' + month)
    return bad_image_list


def split_folds(patients):
    all_patients = []
    num_patient_per_fold = len(patients) // folds
    for i in range(folds):
        random_patient = random.sample(patients, num_patient_per_fold)
        for rp in random_patient:
            patients.remove(rp)
        all_patients.append(random_patient)
    return all_patients


def write_patient(patients, save_name):
    with open(save_name, 'w') as f:
        for p in patients:
            f.write(p + '\n')
    f.close()


if __name__ == '__main__':
    opt = parse_args()

    data_sheet_folder = os.path.join(opt.data_sheet_folder, 'data_sheets')
    df_klg = pd.read_csv(os.path.join(opt.data_sheet_folder, "data_sheets/KLG_score.csv"), sep=',')
    data_sheet_names = ['contents.0E1', 'contents.1E1', 'contents.2D2', 'contents.3E1',
                        'contents.4G1', 'contents.5E1', 'contents.6E1', 'contents.8E1']
    bad_sheet_names = ['outlier_zero_LF', 'outlier_zero_LT', 'outlier_zero_RF', 'outlier_zero_RT']

    save_folder = opt.data_sheet_folder
    save_subfolder = ['train_set1', 'train_set2', 'test_set']

    folds = 5
    descr_dict = {'SAG_3D_DESS_LEFT': 'mri_left',
                  'SAG_3D_DESS_RIGHT': 'mri_right',
                  'Bilateral PA Fixed Flexion Knee': 'xray'}

    # get bad images (actually are the thickness maps)
    bad_image_list = get_bad_images()
    # patient_dict = {patient_id: {month: {image_type: path}}}
    patient_dict = get_patient_month_image()
    # get all patients
    patients = list(patient_dict.keys())    # total 4788 patients
    print('# patients: ', len(patients))
    # get number of patients per set
    num_patients = [2000, 1750]     # train1 & train2 set
    num_patients.append(len(patients) - sum(num_patients))  # test set

    # get patient has progression prediction (put them in the train2 and test set)
    _, prog_patients = get_progression_speed(patients, df_klg)
    print('# progression patients: ', len(prog_patients))
    non_prog_patients = [x for x in patients if x not in prog_patients]

    # split the patients
    patient_splits = []
    random_patient = random.sample(non_prog_patients, num_patients[0])
    for rp in random_patient:
        non_prog_patients.remove(rp)
    random_patient_folds = split_folds(random_patient)
    patient_splits.append(random_patient_folds)   # for metric learning
    prog_patients.extend(non_prog_patients)

    random_patient = random.sample(prog_patients, num_patients[1])
    for rp in random_patient:
        prog_patients.remove(rp)
    random_patient_folds = split_folds(random_patient)
    patient_splits.append(random_patient_folds)   # for downstream task
    patient_splits.append(prog_patients)    # for testing
    print('# patients: ', end='')
    for i in range(len(patient_splits)):
        if isinstance(patient_splits[i][0], list):
                print(len(patient_splits[i][0]), '*', len(patient_splits[i]), end=',')
        else:
            print(len(patient_splits[i]))

    # get matched images
    row_splits, columns = match_split_images(patient_dict, patient_splits)
    print('# rows:', end='')
    for i in range(len(row_splits)):
        print()
        if isinstance(row_splits[i][0][0], list):
            for j in range(len(row_splits[i])):
                folder_name = os.path.join(save_folder, save_subfolder[i], 'fold' + str(j+1))
                os.makedirs(folder_name, exist_ok=True)
                save_name = os.path.join(folder_name, 'mri_xray.csv')
                pd.DataFrame(np.array(row_splits[i][j]), columns=columns).to_csv(save_name, index=0)
                print(len(row_splits[i][j]), end='\t')
        else:
            folder_name = os.path.join(save_folder, save_subfolder[i])
            os.makedirs(folder_name, exist_ok=True)
            save_name = os.path.join(folder_name, 'mri_xray.csv')
            pd.DataFrame(np.array(row_splits[i]), columns=columns).to_csv(save_name, index=0)
            print(len(row_splits[i]))




