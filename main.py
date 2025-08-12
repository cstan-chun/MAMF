import os
import json
import torch
import numpy as np
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.optim import lr_scheduler  # 用于动态调整学习率的调度器。

import matplotlib.pyplot as plt
import seaborn as sns
from opts import parse_opts
from model import generate_model
from datasets.get_dataloder import get_training_loader, get_validation_loader, get_test_loader
from utils import Logger, adjust_learning_rate, save_checkpoint, draw

from train import train_epoch
from validation import val_epoch
from test import test_for_score

if __name__ == '__main__':
    opt = parse_opts()
    n_folds = opt.n_folds  # 5折交叉验证
    test_accuracies = []

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if opt.dataset == 'IEMOCAP':
        opt.annotation_path = 'ravdess_preprocessing/annotations_iemocap_augment.txt'
        opt.n_folds = 1
        opt.n_classes = 6

    opt.result_path = 'results/' + opt.result_path + '_' + opt.dataset + '_dim' + str(opt.feature_dim) + '_lr' + str(opt.learning_rate) + '_drop' + str(opt.dropout) 

    pretrained = opt.pretrain_path != 'None'

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    print(opt)
    with open(os.path.join(opt.result_path, 'opts' + '.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    start_time = datetime.now()

    for fold in range(n_folds):
        if n_folds != 1:
            print("this is fold{0}".format(fold))

        torch.manual_seed(opt.manual_seed)

        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)

        if not opt.no_train:
            training_dataloader = get_training_loader(opt, fold)

            train_logger = Logger(
                os.path.join(opt.result_path, 'train_epoch' + '_fold' + str(fold) + '.log'),
                ['epoch', 'lr', 'loss', 'prec1'])

            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch' + '_fold' + str(fold) + '.log'),
                ['epoch', 'batch', 'lr', 'loss', 'prec1'])

            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,  # 动量项，用于加速梯度下降并减少振荡
                dampening=opt.dampening,  # 控制动量的抑制因子
                weight_decay=opt.weight_decay,  # 权重衰减（L2正则化），用于防止过拟合。
                nesterov=False)  # 指定是否使用Nesterov加速梯度
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)

        if not opt.no_val:
            validation_dataloader = get_validation_loader(opt, fold)

            val_logger = Logger(
                os.path.join(opt.result_path, 'val_epoch' + '_fold' + str(fold) + '.log'),
                ['epoch', 'loss', 'prec1'])

        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)

                train_epoch(i, training_dataloader, model, criterion, optimizer, opt, train_logger, train_batch_logger)

                # 保存当前模型的状态信息到一个检查点（checkpoint）文件中
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
                save_checkpoint(state, False, opt, fold)

            if not opt.no_val:
                validation_loss, prec1 = val_epoch(i, validation_dataloader, model, criterion, opt, val_logger)
                best_prec1 = max(prec1, best_prec1)
                is_best = prec1 > best_prec1

                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
                save_checkpoint(state, is_best, opt, fold)

        # 画loss和prec曲线图
        #draw(opt.result_path, fold, 'epoch')
        #draw(opt.result_path, fold, 'batch')

        if opt.test:
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + '_fold' + str(fold) + '.log'),
                ['batch', 'loss', 'prec1'])

            test_dataloader = get_test_loader(opt, fold)

            best_state = torch.load(
                '%s/%s_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
            model.load_state_dict(best_state['state_dict'])

            test_loss, test_prec1, all_preds, all_targets = test_for_score(test_dataloader, model, criterion, opt,
                                                                           test_logger)
            if opt.dataset == 'RAVDESS':
                label_names = ['neu', 'cal', 'hap', 'sad', 'ang', 'fea', 'dis', 'sur']
            elif opt.dataset == 'IEMOCAP':
                label_names = ['neu', 'hap', 'sad', 'ang', 'exc', 'fru']
            report = classification_report(all_targets, all_preds, zero_division=0, target_names=label_names,digits=4)
            cm = confusion_matrix(all_targets, all_preds)
            # 可视化
            plt.figure(figsize=(6, 5))
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                        xticklabels=label_names, yticklabels=label_names)
            ax.xaxis.set_ticks_position('top')
            plt.tight_layout()
            plt.savefig("confusion_matrix_on_RAVDESS.png", dpi=600)
            plt.close()
            
            with open(os.path.join(opt.result_path, 'test_result_fold' + str(fold) + '.txt'), 'a') as f:
                f.write(f"Prec1:{test_prec1:.2f}\t"
                        f"Loss:{test_loss:.2f}\n"
                        f"{report}")

            # test_accuracies.append(test_prec1)


    end_time = datetime.now()
    print('训练开始时间为: ', start_time)
    print('训练结束时间为: ', end_time)
    print('训练总时间为: ', start_time - end_time)

    # 记录测试精度的平均值和标准差
    with open(os.path.join(opt.result_path, 'test_avg_result.txt'), 'a') as f:
        f.write(
            '\n' + 'Prec1: ' + str(np.mean(np.array(test_accuracies))) + '+' + str(np.std(np.array(test_accuracies))))

