import os
import numpy as np
import pandas as pd
from setting import parse_args


def get_klg(rows):
    # 00: 00m, 01:12m, 03: 24m, 05: 36m, 06: 48m, 08: 72m, 10:96m
    all_month = ['V00XRKL', 'V01XRKL', 'V03XRKL', 'V05XRKL', 'V06XRKL', 'V08XRKL', 'V10XRKL']
    all_klg = [getattr(rows, month).values for month in all_month]
    klg = np.nanmax(np.array(all_klg), axis=1).flatten()
    return klg


def get_speed(klg, month):
    all_month = ['00', '12', '24', '36', '48', '72', '96']
    period_dict = {'00': ['36', '72'], '12': ['48', '96']}
    start_klg = klg[all_month.index(month)]
    middle_klg = klg[all_month.index(period_dict[month][0])]
    end_klg = klg[all_month.index(period_dict[month][1])]
    if np.isnan(start_klg) or np.isnan(middle_klg) or np.isnan(end_klg):
        return -1
    # change all klg=0 to 1
    start_klg = 1 if start_klg == 0 else start_klg
    middle_klg = 1 if middle_klg == 0 else middle_klg
    end_klg = 1 if end_klg == 0 else end_klg

    if start_klg == middle_klg == end_klg:  # no progression
        return 0
    if middle_klg > start_klg:  # fast progression
        return 2
    return 1    # progression


def get_progression_speed(all_patients, df_klg):
    side_dict = {1: 'RIGHT', 2: 'LEFT'}
    qualify_patients = []
    patient_speed = [[] for _ in range(3)]
    for patient in all_patients:
        for side in list(side_dict.keys()):  # right: 1, left: 2
            row = df_klg.loc[(df_klg.ID == int(patient)) & (df_klg.SIDE == side)]
            if len(row) == 0:   # no such patient side
                continue
            all_klg = get_klg(row)
            if all_klg[0] == 4:
                continue
            for month in ['00', '12']:
                speed = get_speed(all_klg, month)
                if speed < 0:   # either start, middle or final klg is nan
                    continue
                if patient not in qualify_patients:
                    qualify_patients.append(patient)
                patient_speed[speed].append(str(int(patient)) + '_' + side_dict[side] + '_' + month)
    return patient_speed, qualify_patients


def get_all_patients(phase):
    all_patients = []
    df = pd.read_csv(os.path.join(opt.data_sheet_folder, phase, opt.data_sheet_name))
    for _, row in df.iterrows():
        if row['patient_id'] not in all_patients:
            all_patients.append(row['patient_id'])
    return all_patients


def match_speed(all_rows, patient_id, month):
    left_name = str(patient_id) + '_LEFT' + '_' + month
    right_name = str(patient_id) + '_RIGHT' + '_' + month
    for i in range(len(patient_speed)):
        if left_name in patient_speed[i]:
            all_rows.append([left_name, i])
        if right_name in patient_speed[i]:
            all_rows.append([right_name, i])
    return all_rows


def save_progression_speed(phase, save_name):
    all_rows = []
    data_sheet_path = os.path.join(opt.data_sheet_folder, phase, opt.data_sheet_name)
    df = pd.read_csv(data_sheet_path, sep=',')
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        month = row['month']
        all_rows = match_speed(all_rows, patient_id, month.split('_')[0])
    print(len(all_rows))
    pd.DataFrame(np.array(all_rows), columns=['Name', 'Speed']).to_csv(os.path.join(opt.data_sheet_folder, phase, save_name), index=0)


if __name__ == '__main__':
    opt = parse_args()
    klg_sheet = os.path.join(opt.data_sheet_folder, "data_sheets/KLG_score.csv")
    df_klg = pd.read_csv(klg_sheet, sep=',')
    phase = ['train_set2/fold1', 'train_set2/fold2', 'train_set2/fold3', 'train_set2/fold4', 'train_set2/fold5', 'test_set']

    for p in phase:
        patients = get_all_patients(p)
        print('\n', len(patients))
        patient_speed, patients = get_progression_speed(patients, df_klg)
        print(len(patients))
        print([len(patient_speed[i]) for i in range(len(patient_speed))])
        save_progression_speed(p, save_name='progression_speed.csv')


