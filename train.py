import numpy as np
import torch
import time
from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()
    losses = AverageMeter()  # 用于记录和计算每个批次的损失值。
    prec1 = AverageMeter()  # 用于记录和计算准确率（Top-1 准确率）。
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):

        # 如果需要应用数据掩码（如噪声或软硬掩码）
        if opt.mask is not None:
            with torch.no_grad():

                if opt.mask == 'noise':
                    # 将随机噪声添加到音频输入中，并将原始音频输入拼接在一起
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)
                    # 将随机噪声添加到视觉输入中，并将原始视觉输入拼接在一起
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
                    # 将目标标签重复三次,以匹配扩展后的输入数据集。
                    targets = torch.cat((targets, targets, targets), dim=0)
                    # 打乱数据的顺序，以增加训练的鲁棒性和模型的泛化能力。
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

                # 实现了一种混合掩码技术，其中包括将软掩码和硬掩码应用于输入数据。
                elif opt.mask == 'softhard':
                    # 生成系数，用于应用软硬掩码
                    coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
                    vision_coefficients = 1 - coefficients

                    # 将系数扩展到与音频输入数据的维度相匹配
                    # 将音频掩码系数沿第二和第三维度复制，以匹配音频输入数据的形状。
                    coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
                    # 将视觉系数扩展到与视觉输入数据的维度相匹配
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                                                 visual_inputs.size(1),
                                                                                                 visual_inputs.size(2),
                                                                                                 visual_inputs.size(3),
                                                                                                 visual_inputs.size(4))

                    audio_inputs = torch.cat(
                        (audio_inputs, audio_inputs * coefficients, torch.zeros(audio_inputs.size()), audio_inputs),
                        dim=0)
                    visual_inputs = torch.cat((visual_inputs, visual_inputs * vision_coefficients, visual_inputs,
                                               torch.zeros(visual_inputs.size())), dim=0)

                    # 将目标标签数据拼接四次，以匹配音频和视觉数据的扩展。
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    # 打乱数据的顺序。
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

        visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)  # permute 是 PyTorch 中用于重新排列张量维度的方法。
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])

        audio_inputs = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
        targets = targets.to(opt.device)

        # 前向传播
        multi_logits = model(audio_inputs, visual_inputs)

        # 计算损失
        multi_loss = criterion(multi_logits, targets)
        # 更新记录器
        losses.update(multi_loss.data, audio_inputs.size(0))

        # 计算准确率
        accuracy, _ = calculate_accuracy(multi_logits.data, targets.data, topk=(1, 5))
        prec1.update(accuracy, audio_inputs.size(0))

        # 反向传播和优化
        optimizer.zero_grad()
        multi_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch:[{epoch}/{opt.n_epochs}]\t"
                  f"Batch:[{i}/{len(data_loader)}]\t"
                  f"lr:{optimizer.param_groups[0]['lr']:.6f}\t"
                  f"loss:{losses.val:.4f}\t"
                  f"prec1:{prec1.val:.2f}\t"
                  f"loss-avg:{losses.avg:.4f}\t"
                  f"prec1-avg:{prec1.avg:.2f}")

        # 记录每个批次的日志
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'loss': f"{losses.avg.item():.4f}",
            'prec1': f"{prec1.avg.item():.2f}"
        })

    # 记录 epoch 结束时的日志
    epoch_logger.log({
        'epoch': epoch,
        'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
        'loss': f"{losses.avg.item():.4f}",
        'prec1': f"{prec1.avg.item():.2f}"
    })
