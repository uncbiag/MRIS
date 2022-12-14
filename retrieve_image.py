# given a query image in the TEST set, retrieve the top K images from the TRAIN set
# use weighted average to synthesis a thickness map

import os
import numpy as np
import pandas as pd
from ast import literal_eval

import data
from setting import parse_args
from evaluation import load_model, encode_data
from utils import get_weight
from preprocess import read_thickness, read_smooth_thickness, read_erode_mask


def get_embeddings():
    # get embeddings from the trained network
    model = load_model(opt)

    test_loader = data.get_loader(opt, phase, shuffle=False)
    data1_embs, _, query_names, _ = encode_data(opt, model, test_loader, is_train=False)  # xray
    train_loader = data.get_loader(opt, phase=['train_set1/fold1', 'train_set1/fold2', 'train_set1/fold3',
                                               'train_set1/fold4', 'train_set1/fold5'], shuffle=False)
    _, data2_embs,lookup_names, _ = encode_data(opt, model, train_loader, is_train=False)    # thickness
    data_names = [query_names, lookup_names]
    return data_names, data1_embs, data2_embs


def match_features(names, query, lookup, save_name, num_bad=0):
    # retrieve the topk and bottomk result based on cosine distance
    all_rows = []
    for i in range(len(query)):
        query_name = names[0][i] if len(names) == 2 else names[i]
        lookup_names = names[1] if len(names) == 2 else names

        im = query[i].reshape(1, -1)    # reshape query image
        dist = np.dot(im, lookup.T).flatten()  # cosine distance
        sorted_ind = np.argsort(dist)[::-1]

        topk_inds = sorted_ind[:k]
        topk_names = [lookup_names[ind] for ind in topk_inds]
        topk_distance = [dist[ind] for ind in topk_inds]

        bottomk_inds = sorted_ind[-(num_bad+k):-num_bad]
        bottomk_names = [lookup_names[ind] for ind in bottomk_inds]
        bottomk_distance = [dist[ind] for ind in bottomk_inds]
        all_rows.append([query_name, topk_names, topk_distance, bottomk_names, bottomk_distance])
    print('save result to ', save_name)
    pd.DataFrame(np.array(all_rows), columns=columns).to_csv(save_name, index=0)


def synthesis(query, maps, weight, types):
    # synthesis image maps
    all_results = []    # dim: [len(types), 2, H, W]
    for t in types:  # loop all the types
        results = []
        for i in range(len(maps)):  # loop fc & tc
            if t == 'std':
                synthesis_map = np.std(maps[i], axis=0)
            else:
                synthesis_map = np.sum([weight[j] * maps[i][j] for j in range(len(maps[i]))], axis=0)
                if 'err' in t:  # (abs)err, (abs)rel_err
                    synthesis_map = synthesis_map - query[i]
                if 'abs' in t:  # abs_err, abs_rel_err
                    synthesis_map = np.abs(synthesis_map)
                if 'rel' in t:
                    synthesis_map = synthesis_map / query[i]
            results.append(synthesis_map)
        all_results.append(results)
    return all_results


def synthesis_map(row, data_folder, query):
    # synthesis image maps from topk or bottomk images
    topk_name = literal_eval(row['TopK_Name'])[:topk]
    topk_distance = literal_eval(row['TopK_Distance'])[:topk]

    # get the top & bottom maps & weights
    topk_fc_maps, topk_tc_maps = read_thickness(data_folder, topk_name, replace_nan=False)
    topk_weights = get_weight(topk_distance, use_weighted)
    top_maps = synthesis(query, [topk_fc_maps, topk_tc_maps], topk_weights, map_type)

    if retrieve_bottom:
        bottomk_name = literal_eval(row['BottomK_Name'])[-topk:]
        bottomk_distance = literal_eval(row['BottomK_Distance'])[-topk:]
        bottomk_fc_maps, bottomk_tc_maps = read_thickness(data_folder, bottomk_name, replace_nan=False)
        bottomk_weights = get_weight(bottomk_distance, use_weighted)
        bottom_maps = synthesis(query, [bottomk_fc_maps, bottomk_tc_maps], bottomk_weights, map_type)
        return top_maps, bottom_maps
    return top_maps, None


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


def get_map_err_stat(data_folder, max_show=-1):
    count = 0
    top_err = [[] for _ in range(len(err_type))]
    bottom_err = [[] for _ in range(len(err_type))]
    top_err_klg = [[[] for _ in range(4)] for _ in range(len(err_type))]   # by klg
    bottom_err_klg = [[[] for _ in range(4)] for _ in range(len(err_type))]    # by klg

    df_klg = pd.read_csv('/playpen-raid/bqchen/code/OAIRetrieval/data/data_sheets/KLG_score.csv', sep=',')
    patient_klg_dict = load_klg(df_klg)
    print('finish loading KLG')

    # only for thickness erosion
    thickness_mask = read_erode_mask()

    df = pd.read_csv(result_file, sep=',')
    for index, row in df.iterrows():
        count += 1
        if count > max_show > 0:
            break
        query_fc_thickness, query_tc_thickness = read_smooth_thickness(data_folder, row['Name'], smooth=smooth, replace_nan=False)
        top_maps, bottom_maps = synthesis_map(row, data_folder, [query_fc_thickness, query_tc_thickness])
        if erode:
            for i in range(len(top_maps)):  # type
                for j in range(len(top_maps[i])):   # fc/tc
                    top_maps[i][j] = np.where(thickness_mask[j] == 0, np.nan, top_maps[i][j])
                    if retrieve_bottom:
                        bottom_maps[i][j] = np.where(thickness_mask[j] == 0, np.nan, bottom_maps[i][j])

        # [type, fc/tc, 310*310]
        top_maps = np.reshape(top_maps, (len(top_maps), len(top_maps[0]), -1))
        if retrieve_bottom:
            bottom_maps = np.reshape(bottom_maps, (len(bottom_maps), len(bottom_maps[0]), -1))

        for i in range(len(err_type)):
            top_statitics = getattr(np, 'nan'+err_type[i])(top_maps, axis=-1)
            top_err[i].append(top_statitics)
            if retrieve_bottom:
                bottom_statitics = getattr(np, 'nan'+err_type[i])(bottom_maps, axis=-1)
                bottom_err[i].append(bottom_statitics)
            if row['Name'] in patient_klg_dict:
                klg = patient_klg_dict[row['Name']]
                idx = klg - 1 if klg > 0 else 0
                top_err_klg[i][idx].append(top_statitics)
                if retrieve_bottom:
                    bottom_err_klg[i][idx].append(bottom_statitics)

    # top & bottom err: [3 classes, # data in each class, # map types, fc/tc]
    mean_top_err = [np.nanmean(top_err[i], axis=0) for i in range(len(top_err))]
    mean_top_err_klg = [[np.nanmean(top_err_klg[i][j], axis=0) for j in range(len(top_err_klg[i]))] for i in range(len(top_err_klg))]
    if retrieve_bottom:
        mean_bottom_err = [np.nanmean(bottom_err[i], axis=0) for i in range(len(bottom_err))]
        mean_bottom_err_klg = [[np.nanmean(bottom_err_klg[i][j], axis=0) for j in range(len(bottom_err_klg[i]))] for i in range(len(bottom_err_klg))]
    else:
        mean_bottom_err = None
        mean_bottom_err_klg = None

    return mean_top_err, mean_bottom_err, mean_top_err_klg, mean_bottom_err_klg


def write_results(err, err_klg, name):
    smooth_erode = "nosmooth_noerode.txt"
    if smooth:
        smooth_erode = smooth_erode.replace('nosmooth', 'smooth')
    if erode:
        smooth_erode = smooth_erode.replace('noerode', 'erode')
    statistics_result_folder = os.path.join(result_folder, 'statistics')
    os.makedirs(statistics_result_folder, exist_ok=True)
    f = open(os.path.join(statistics_result_folder, name + '_error_' + smooth_erode), 'w')
    for i in range(len(err_type)):
        for j in range(len(map_type)):
            f.write(err_type[i] + ' ' + map_type[j] + ' ' + name + ':\n')
            f.write(' overall: ' + str(err[i][j]) + '\n')
            print(err_type[i], map_type[j], name)
            print(f" overall: {err[i][j]}..")  # [median/mean, abs_err/abs_rel_err, fc&tc]
            for k in range(len(err_klg[i])):    # [median/mean, klg, abs_err/abs_rel_err, fc&tc]
                f.write(' klg=' + str(k+1) + ': ' + str(err_klg[i][k][j]) + '\n')
                print(f" klg={k+1}: {err_klg[i][k][j]}..")


if __name__ == '__main__':
    # !!! REMEMBER TO REMOVE ONE_PER_PATIENT!!! #
    # python retrieve_image.py --resume result/model_best.pth.tar --cnn_type resnet18 --flip
    # -bs 64 -g 3 --region fc --input_size2 310 310 --load_into_memory

    k = 20      # retrieve the top & bottom k
    use_weighted = True     # use weighted average of the retrieved data
    smooth = False
    erode = False
    retrieve_bottom = False  # beside the top k, also retrieve the bottom k

    opt = parse_args()

    phase = ['test_set']
    # phase = ['train_set2/fold1', 'train_set2/fold2', 'train_set2/fold3', 'train_set2/fold4', 'train_set2/fold5']
    result_folder = os.path.join(opt.resume.replace(opt.resume.split('/')[-1], ''), 'retrieval_result')
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, 'result_'+phase[0].split('_')[0]+'.csv')
    columns = ['Name', 'TopK_Name', 'TopK_Distance', 'BottomK_Name', 'BottomK_Distance']

    # save the retrieved results to csv file
    data_names, data1_embs, data2_embs = get_embeddings()
    num_bad = len(data2_embs) // 50  # assume 2% of the data are bad when retrieve bottom k
    print('assume ', num_bad, 'bad data out of ', len(data2_embs))
    match_features(data_names, data1_embs, data2_embs, save_name=result_file, num_bad=num_bad)

    # get the statistics
    topk = 20
    err_type = ['median', 'mean']
    map_type = ['abs_err']
    top_err, bottom_err, top_err_klg, bottom_err_klg = get_map_err_stat(opt.data_path2, max_show=-1)

    write_results(top_err, top_err_klg, 'top'+str(topk))
    if retrieve_bottom:
        write_results(bottom_err, bottom_err_klg, 'bottom'+str(topk))

