import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="image retrieval")
    # sheets
    parser.add_argument('--data_sheet_folder', default='/playpen-raid/bqchen/code/RetrievalForSynthesis/data',
                        help='path to the data sheet folder')
    parser.add_argument('--data_sheet_name', default='mri_xray.csv',
                        help='data sheet name')
    parser.add_argument('--kp_sheet_name', default='xray_keypoints.csv',
                        help='keypoints sheet name')
    # read & save path
    parser.add_argument('--data_path1', default='/playpen-raid/data/OAI',
                        help='path to the xray data')
    parser.add_argument('--data_path2', default='/playpen-raid/bqchen/data/oai/longleaf',
                        help='path to the thickness/mri data')
    parser.add_argument('--save_name', default='result/retrieval',
                        help='Path to save the model')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # xray roi
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)

    # load image
    parser.add_argument('--input_size1', default=[384, 384], nargs='+', type=int,
                        help='input size of data1')
    parser.add_argument('--input_size2', default=[620, 310], nargs='+', type=int,
                        help='input size of data2')
    parser.add_argument('--region', default='all',
                        help='use fc/tc/all for retrieval task')
    parser.add_argument('--augmentation', action='store_true',
                        help='Add data augmentation during training.')
    parser.add_argument('--flip', action='store_true',
                        help='flip all knee to the same side')

    # train settings
    parser.add_argument('-e', '--num_epochs', default=450, type=int,
                        help='Number of training epochs.')
    parser.add_argument('-bs', '--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('-lr1', '--learning_rate1', default=1e-04, type=float,
                        help='Initial learning rate for data1.')
    parser.add_argument('-lr2', '--learning_rate2', default=1e-05, type=float,
                        help='Initial learning rate for data2.')
    parser.add_argument('--lr_update', default=150, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('-g', '--gpu_ids', default=0, nargs='+', type=int,
                        help='the gpu ids to use, if < 0 use cpu')
    parser.add_argument('--grad_clip', default=1., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--cnn_type', default='resnet18',
                        help='The CNN used for image encoder (e.g. resnet18, resnet152)')
    parser.add_argument('--margin', default=0.1, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--load_into_memory', action='store_true',
                        help='Load all data into memory')
    parser.add_argument('--num_load', default=-1, type=int,
                        help='Maximum number of data loaded')
    parser.add_argument('--seed', default=246810, type=int,
                        help='set seed for reproduce')
    parser.add_argument('--one_per_patient', action='store_true',
                        help='sample one data per patient for each batch')
    parser.add_argument('--inverse', action='store_true',
                        help='add inverse retrieval')
    # downstream task
    parser.add_argument('--resume_fc', default=None, type=str, metavar='PATH',
                        help='path to FC checkpoint (for downstream task)')
    parser.add_argument('--resume_tc', default=None, type=str, metavar='PATH',
                        help='path to TC checkpoint (for downstream task)')
    parser.add_argument('--resume_pred', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint for downstream task(default: none)')
    parser.add_argument('--task', default=1, type=int,
                        help='1: KLG pred, 2: progression pred with 2 input')
    parser.add_argument('--modality', default=2, type=int,
                        help='1: xray, 2: GT thickness map, 3: retrieved thickness map')

    # visualization
    parser.add_argument('--criterion', default=['9757487_LEFT_12', '9722580_LEFT_24', '9933459_LEFT_24',
                                                '9319367_RIGHT_12'], nargs='+', type=str, help='visualize names')

    args = parser.parse_args()
    return args

