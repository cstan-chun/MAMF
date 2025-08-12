import csv
import os

import torch
import shutil
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    # print('target', target, 'output', output)
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        # print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()), list(pred[0].cpu().numpy()))
        # print('F1: ', f1)
        return res, f1 * 100
    # print(res)
    return res


def save_checkpoint(state, is_best, opt, fold):
    torch.save(state, '%s/%s_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth',
                        '%s/%s_best_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth')


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        # param_group['lr'] = opt.learning_rate


def draw(result_path, fold, type):
    # 读取 CSV 文件
    df = pd.read_csv(os.path.join(result_path, 'train_' + type + '_fold' + str(fold) + '.log'), sep="\t")
    # print(df)

    loss = df["loss"]
    prec1 = df["prec1"]
    x = range(len(loss))

    # 设置绘图风格
    plt.style.use('seaborn')
    # 创建图像
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 画 Loss 曲线（左轴）
    ax1.plot(x, loss, "r-", label="Loss")
    ax1.set_xlabel(type)
    ax1.set_ylabel("Loss", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    # 共享 x 轴，创建第二个 y 轴
    ax2 = ax1.twinx()
    ax2.plot(x, prec1, "b-", label="Prec")
    ax2.set_ylabel("Prec@1 (%)", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # 添加标题
    plt.title(f"{type}:Loss & Prec@1 Curve")
    plt.legend()

    # 显示图像
    # plt.show()
    output_png = type + '_loss_prec_curve.png'
    plt.savefig(os.path.join(result_path, output_png), dpi=300)
