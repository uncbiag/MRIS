# given a query image in the TEST set, retrieve the top K images from the TRAIN set
# use weighted average to synthesis a thickness map

import os
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import median_abs_deviation

import data
from utils import get_weight, plot_image
from setting import parse_args
from preprocess import read_thickness, resize_image
from evaluation import load_model, encode_data


def get_embeddings():
    # get embeddings from the trained network
    model = load_model(opt)
    test_loader = data.get_loader(opt, phase, shuffle=False)
    data1_embs, _, query_names, _ = encode_data(opt, model, test_loader, is_train=False)  # xray
    train_loader = data.get_loader(opt, phase=['train_set/fold1', 'train_set/fold2', 'train_set/fold3',
                                               'train_set/fold4', 'train_set/fold5'], shuffle=False)
    _, data2_embs,lookup_names, _ = encode_data(opt, model, train_loader, is_train=False)    # thickness
    data_names = [query_names, lookup_names]
    return data_names, data1_embs, data2_embs


def match_features(names, query, lookup, save_name):
    # retrieve the topk result based on cosine distance
    all_rows = []
    for i in range(len(query)):
        query_name = names[0][i] if len(names) == 2 else names[i]
        lookup_names = names[1] if len(names) == 2 else names

        img = query[i].reshape(1, -1)    # reshape query image
        dist = np.dot(img, lookup.T).flatten()  # cosine distance
        inds = np.argsort(dist)[::-1]
        topk_inds = inds[:topk]
        topk_names = [lookup_names[ind] for ind in topk_inds]
        topk_distance = [dist[ind] for ind in topk_inds]

        all_rows.append([query_name, topk_names, topk_distance])
    print('save result to ', save_name)
    pd.DataFrame(np.array(all_rows), columns=columns).to_csv(save_name, index=0)


def get_err_map(query, syn):
    # synthesis image err maps
    err_maps = []  # dim: [len(types), 2, H, W]
    for t in map_type:  # loop all the types
        results = []
        for i in range(len(syn)):  # loop fc & tc
            syn_map = syn[i]
            if 'err' in t:  # (abs)err, (abs)rel_err
                syn_map = syn_map - query[i]
            if 'abs' in t:  # abs_err, abs_rel_err
                syn_map = np.abs(syn_map)
            if 'rel' in t:
                syn_map = syn_map / query[i]
            results.append(syn_map)
        err_maps.append(results)
    return err_maps


def load_klg(df):
    patient_klg_dict = {}
    month_dict = {'00': 'V00XRKL', '12': 'V01XRKL', '24': 'V03XRKL', '36': 'V05XRKL',
                  '48': 'V06XRKL', '72': 'V08XRKL', '96': 'V10XRKL'}
    side_dict = {1: 'RIGHT', 2: 'LEFT'}
    all_month = list(month_dict.keys())
    for _, row in df.iterrows():
        patient_id = str(int(row['ID']))
        side = side_dict[row['SIDE']]
        for month in all_month:
            klg = row[month_dict[month]]
            name = patient_id + '_' + side + '_' + month
            if not np.isnan(klg) and name not in list(patient_klg_dict.keys()):
                patient_klg_dict[name] = int(klg)
    return patient_klg_dict


def get_synthesis_stat(data_folder):
    top_err = [[] for _ in range(len(err_type))]
    top_err_klg = [[[] for _ in range(4)] for _ in range(len(err_type))]   # by klg

    df_klg = pd.read_csv(os.path.join(opt.data_sheet_folder, 'KLG_score.csv'), sep=',')
    patient_klg_dict = load_klg(df_klg)
    print('finish loading KLG')

    df = pd.read_csv(result_file, sep=',')
    for index, row in df.iterrows():
        query_fc, query_tc = read_thickness(data_folder, row['Name'], replace_nan=False)
        query_fc = resize_image(query_fc, opt.input_size2, ignore_nan=True)
        query_tc = resize_image(query_tc, opt.input_size2, ignore_nan=True)

        # synthesis image maps from topk images
        topk_name = literal_eval(row['TopK_Name'])[:topk]
        topk_distance = literal_eval(row['TopK_Distance'])[:topk]
        fc_maps, tc_maps = read_thickness(data_folder, topk_name, replace_nan=False)
        weights = get_weight(topk_distance, use_weighted)
        syn_fc = np.sum(weights[i] * fc_maps[i] for i in range(len(weights)))
        syn_tc = np.sum(weights[i] * tc_maps[i] for i in range(len(weights)))
        syn_fc = resize_image(syn_fc, opt.input_size2, ignore_nan=True)
        syn_tc = resize_image(syn_tc, opt.input_size2, ignore_nan=True)

        # get the err_map
        top_err_maps = get_err_map([query_fc, query_tc], [syn_fc, syn_tc])
        top_err_maps = np.reshape(top_err_maps, (len(top_err_maps), len(top_err_maps[0]), -1))  # [type, fc/tc, size, size]
        for i in range(len(err_type)):
            if err_type[i] == 'iqr':
                q1_err = np.nanpercentile(top_err_maps, q=25, axis=-1)
                q3_err = np.nanpercentile(top_err_maps, q=75, axis=-1)
                top_statitics = q3_err - q1_err
            elif err_type[i] == 'mad':
                top_statitics = median_abs_deviation(top_err_maps, axis=-1, nan_policy='omit')
            else:
                top_statitics = getattr(np, 'nan'+err_type[i])(top_err_maps, axis=-1)
            top_err[i].append(top_statitics)
            if row['Name'] in patient_klg_dict:
                klg = patient_klg_dict[row['Name']]
                idx = klg - 1 if klg > 0 else 0
                top_err_klg[i][idx].append(top_statitics)

    # top & bottom err: [3 classes, # data in each class, # map types, fc/tc]
    mean_top_err = [np.nanmean(top_err[i], axis=0) for i in range(len(top_err))]
    mean_top_err_klg = [[np.nanmean(top_err_klg[i][j], axis=0) for j in range(len(top_err_klg[i]))] for i in range(len(top_err_klg))]

    return mean_top_err, mean_top_err_klg


def write_results(err, err_klg, name):
    statistics_result_folder = os.path.join(result_folder, 'statistics')
    os.makedirs(statistics_result_folder, exist_ok=True)
    f = open(os.path.join(statistics_result_folder, name + '_error.txt'), 'w')
    for i in range(len(err_type)):
        for j in range(len(map_type)):
            f.write(err_type[i] + ' ' + map_type[j] + ' ' + name + ':\n')
            f.write(' overall: ' + str(err[i][j]) + '\n')
            print(err_type[i], map_type[j], name)
            print(f" overall: {err[i][j]}..")  # [median/mean, abs_err/abs_rel_err, fc&tc]
            for k in range(len(err_klg[i])):    # [median/mean, klg, abs_err/abs_rel_err, fc&tc]
                f.write(' klg=' + str(k+1) + ': ' + str(err_klg[i][k][j]) + '\n')
                print(f" klg={k+1}: {err_klg[i][k][j]}..")


def save_result(data_folder, batch_size, max_save=10):
    save_folder = os.path.join(result_folder, 'synthesized_images')
    os.makedirs(save_folder, exist_ok=True)
    df = pd.read_csv(result_file, sep=',')
    count = 0
    for index, row in df.iterrows():
        if count > max_save:
            return
        if index % batch_size != 0:
            continue
        query_fc, query_tc = read_thickness(data_folder, row['Name'], replace_nan=True)
        query_fc = resize_image(query_fc, opt.input_size2, ignore_nan=True)
        query_tc = resize_image(query_tc, opt.input_size2, ignore_nan=True)

        # synthesis image maps from topk images
        topk_name = literal_eval(row['TopK_Name'])[:topk]
        topk_distance = literal_eval(row['TopK_Distance'])[:topk]
        fc_maps, tc_maps = read_thickness(data_folder, topk_name, replace_nan=True)
        weights = get_weight(topk_distance, use_weighted)
        syn_fc = np.sum(weights[i] * fc_maps[i] for i in range(len(weights)))
        syn_tc = np.sum(weights[i] * tc_maps[i] for i in range(len(weights)))
        syn_fc = resize_image(syn_fc, opt.input_size2, ignore_nan=False)
        syn_tc = resize_image(syn_tc, opt.input_size2, ignore_nan=False)

        save_name = os.path.join(save_folder, row['Name'] + '.png')
        if opt.region == 'fc':
            plot_image([query_fc, syn_fc], save_name=save_name)
        elif opt.region == 'tc':
            plot_image([query_tc, syn_tc], save_name=save_name)
        else:
            plot_image([query_fc, query_tc, syn_fc, syn_tc], save_name=save_name)
        count += 1


if __name__ == '__main__':
    topk = 20      # retrieve the top
    use_weighted = True     # use weighted average of the retrieved data

    opt = parse_args()

    phase = ['test_set']
    result_folder = os.path.join(opt.ckp_path.replace(opt.ckp_path.split('/')[-1], ''), 'retrieval_result')
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, 'result_' + phase[0].split('_')[0] + '.csv')
    columns = ['Name', 'TopK_Name', 'TopK_Distance']

    # save the retrieved results to csv file
    data_names, data1_embs, data2_embs = get_embeddings()
    match_features(data_names, data1_embs, data2_embs, save_name=result_file)

    # get the statistics
    err_type = ['median', 'mean', 'std', 'iqr', 'mad']
    map_type = ['abs_err']
    top_err, top_err_klg = get_synthesis_stat(opt.data_path2)
    write_results(top_err, top_err_klg, 'top'+str(topk))

    # visualize image
    save_result(opt.data_path2, opt.batch_size, max_save=10)
