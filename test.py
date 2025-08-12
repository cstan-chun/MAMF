import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy


def test_for_score(data_loader, model, criterion, opt, logger, modality='both', dist=None):
    model.eval()  # 评估模式
    print('test start')

    all_preds = []
    all_targets = []
    losses = AverageMeter()
    prec1 = AverageMeter()

    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):

        # 在验证阶段根据指定的模态（音频或视频）和失真类型对输入数据进行预处理。
        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':  # 用随机噪声替代视频输入。
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise':  # opt.mask == -4:
                # 在视频输入中添加噪声。噪声是通过计算视频输入的均值和标准差生成的。
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (
                        torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size()))
            elif dist == 'zeros':  # 用全零张量替代视频输入。
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise':  # opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (
                        torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size()))
            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())

        # 调整视频输入的形状：
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2],
                                              inputs_visual.shape[3], inputs_visual.shape[4])
        targets = targets.to(opt.device)

        with torch.no_grad():  # 不需要计算梯度
            inputs_visual = Variable(inputs_visual)  # Variable 是 PyTorch 中的一个封装类，用于支持自动求导和梯度计算。
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)

            # 前向传播
            multi_logits = model(inputs_audio, inputs_visual)
            # 计算损失
            multi_loss = criterion(multi_logits, targets)

        # 更新记录器
        accuracy, _ = calculate_accuracy(multi_logits.data, targets.data, topk=(1, 5))
        prec1.update(accuracy, inputs_audio.size(0))
        losses.update(multi_loss.data, inputs_audio.size(0))

        if multi_logits.dim() == 1:  # 如果 batch_size=1，multi_logits 可能是一维
            preds = torch.argmax(multi_logits, dim=0)  # 用 dim=0 取最大类别
        else:
            preds = torch.argmax(multi_logits, dim=1)  # batch_size > 1 正常使用 dim=1

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

        print(
            f"Batch:[{i + 1}/{len(data_loader)}]\t"
            f"loss:{losses.val:.4f}\t"
            f"prec1:{prec1.val:.2f}\t"
            f"loss-avg:{losses.avg:.4f}\t"
            f"prec1-avg:{prec1.avg:.2f}")

        logger.log({
            'batch': i + 1,
            'loss': f"{losses.avg.item():.4f}",
            'prec1': f"{prec1.avg.item():.2f}"})

    return losses.avg.item(), prec1.avg.item(), all_preds, all_targets
