import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MRIS")
    # sheets and data path
    parser.add_argument('--data_sheet_folder', default='./data', help='path to the data sheet folder')
    parser.add_argument('--data_sheet_name', default='mri_xray.csv', help='data sheet name')
    parser.add_argument('--kp_sheet_name', default='xray_keypoints.csv', help='keypoints sheet name')
    parser.add_argument('--data_path1', default='/playpen-raid/data/OAI', help='path to the xray data folder')
    parser.add_argument('--data_path2', default='/playpen-raid/bqchen/data/oai/longleaf', help='path to the thickness data folder')

    # save and load checkpoint path
    parser.add_argument('--save_name', default='result', help='Path to save the ckeckpoint save folder')
    parser.add_argument('--ckp_path', default=None, type=str, metavar='PATH', help='path to checkpoint')

    # image settings
    parser.add_argument('--input_size1', default=[256, 256], nargs='+', type=int, help='input size of data1')
    parser.add_argument('--input_size2', default=[256, 256], nargs='+', type=int, help='input size of data2')
    parser.add_argument('--roi_size_mm', type=int, default=140, help='xray region of interest size by mm')
    parser.add_argument('--region', default='fc', help='use fc/tc/all for retrieval task')
    parser.add_argument('--augmentation', action='store_true', help='Add data augmentation during training.')
    parser.add_argument('--no_flip',  action='store_true', help='do not flip all knee to the same side')
    parser.add_argument('--no_norm', action='store_true', help='do not normalize all thickness maps')

    # train settings
    parser.add_argument('-e', '--num_epochs', default=450, type=int, help='Number of training epochs.')
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Size of a training mini-batch.')
    parser.add_argument('-lr1', '--learning_rate1', default=1e-04, type=float, help='Initial learning rate for data1.')
    parser.add_argument('-lr2', '--learning_rate2', default=1e-05, type=float, help='Initial learning rate for data2.')
    parser.add_argument('--lr_update', default=150,  type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
    parser.add_argument('-g', '--gpu_ids', default=[0], nargs='+', type=int, help='the gpu ids to use, if < 0 use cpu')
    parser.add_argument('--grad_clip', default=1., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--cnn_type', default='resnet18', help='The CNN used for image encoder (e.g. resnet18, resnet152)')
    parser.add_argument('--margin', default=0.1, type=float, help='triplet loss margin.')
    parser.add_argument('--load_into_memory', action='store_true', help='Load all data into memory')
    parser.add_argument('--num_load', default=-1, type=int, help='Maximum number of data loaded (for debug)')
    parser.add_argument('--seed', default=246810, type=int, help='set seed for reproduce')
    parser.add_argument('--one_per_patient', action='store_true', help='sample one data per patient for each batch')

    args = parser.parse_args()
    return args

