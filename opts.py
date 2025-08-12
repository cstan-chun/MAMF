# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='ravdess_preprocessing/annotations_augmented.txt', type=str, help='')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--n_classes', default=8, type=int, help='Number of classes')
    parser.add_argument('--model', default='mutualattention', type=str, help='')
    # ===================================================================================================================
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--dataset', default='RAVDESS', type=str, help='Used dataset. Currently supporting Ravdess')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    parser.add_argument('--learning_rate', default=0.02, type=float, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--feature_dim', default=128, type=int, help='')
    parser.add_argument('--audio_features', default='mfcc', type=str, help='')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_folds', default=5, type=int, help='')
    parser.add_argument('--dropout', default=0.3, type=float, help='')
    # ==================================================================================================================
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='')
    parser.add_argument('--begin_epoch', default=1, type=int, help='')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='EfficientFace_Trained_on_AffectNet7.pth.tar', type=str, help='')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=True)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=True)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--n_threads', default=3, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='')
    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='noise')
    args = parser.parse_args()

    return args
