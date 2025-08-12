import torch
from torch.utils.data import DataLoader

from datasets import transforms
from datasets.iemocap import IEMOCAP
from datasets.mosi_mosei import MOSI_MOSEI
from datasets.ravdess import RAVDESS


def get_training_loader(opt, fold):
    assert opt.dataset in ['RAVDESS', 'IEMOCAP', 'MOSI', 'MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    video_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotate(),
        transforms.ToTensor(opt.video_norm_value)])

    audio_transform = transforms.Compose([
        transforms.AddNoise(),
        transforms.PitchShift()])

    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'train',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=audio_transform)
    elif opt.dataset == 'IEMOCAP':
        training_data = IEMOCAP(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'train',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=audio_transform)

    training_dataloader = DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=False)

    return training_dataloader


def get_validation_loader(opt, fold):
    assert opt.dataset in ['RAVDESS', 'IEMOCAP', 'MOSI', 'MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    video_transform = transforms.Compose([
        transforms.ToTensor(opt.video_norm_value)])

    if opt.dataset == 'RAVDESS':

        valid_dataset = RAVDESS(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'val',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=None)

    elif opt.dataset == 'IEMOCAP':

        valid_dataset = IEMOCAP(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'val',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=None)

    validation_dataloader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)

    return validation_dataloader


def get_test_loader(opt, fold):
    assert opt.dataset in ['RAVDESS', 'IEMOCAP', 'MOSI', 'MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    video_transform = transforms.Compose([
        transforms.ToTensor(opt.video_norm_value)])

    if opt.dataset == 'RAVDESS':

        test_dataset = RAVDESS(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'test',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=None)

    elif opt.dataset == 'IEMOCAP':
        test_dataset = IEMOCAP(
            opt.annotation_path,
            fold,
            opt.n_folds,
            'test',
            opt.audio_features,
            spatial_transform=video_transform, data_type='audiovisual', audio_transform=None)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False)
    return test_dataloader
