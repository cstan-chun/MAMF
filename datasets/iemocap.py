import torch
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.functions import get_default_video_loader, load_audio, get_mfccs, get_fbank, get_multifeatures


def make_dataset(fold, n_folds, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()

    # 标签处理为编号
    # valid_emotions = {'neu', 'hap', 'sad', 'ang', 'exc', 'fru'}
    label_ids = {'neu': 1, 'hap': 2, 'sad': 3, 'ang': 4, 'exc': 5, 'fru': 6}

    dataset = []
    labels = []
    for line in annots:
        # filename（视频文件名）、audiofilename（音频文件名）、text、label（标签）
        filename, audiofilename, label = line.strip().split(';')
        label_id = label_ids[label] - 1
        sample = {'video_path': filename,
                  'audio_path': audiofilename,
                  'label': label_id}

        dataset.append(sample)
        labels.append(label_id)

    train_set, test_val_set, _, test_val_labels = train_test_split(dataset, labels, test_size=0.2, stratify=labels,
                                                                   random_state=42)

    val_set, test_set, _, _ = train_test_split(test_val_set, test_val_labels, test_size=0.5, stratify=test_val_labels,
                                               random_state=42)

    return train_set, val_set, test_set

    """
    folds = [[[4], [5], [1, 2, 3]], [[1], [2], [3, 4, 5]], [[2], [3], [1, 4, 5]], [[3], [4], [1, 2, 5]],
             [[5], [1], [2, 3, 4]]]
    fold_ids = folds[fold]
    test_ids, val_ids, train_ids = fold_ids

    train_set, val_set, test_set = [], [], []

    for line in annots:
        # filename（视频文件名）、audiofilename（音频文件名）、text、label（标签）
        filename, audiofilename, label = line.strip().split(';')
        sample = {'video_path': filename,
                  'audio_path': audiofilename,
                  'label': label_ids[label] - 1}

        session_ids = int(filename.split('\\')[-1].split('_')[0][4])

        if session_ids in train_ids:
            train_set.append(sample)
        elif session_ids in val_ids:
            val_set.append(sample)
        else:
            test_set.append(sample)

    return train_set, val_set, test_set
    """


class IEMOCAP(data.Dataset):
    def __init__(self,
                 annotation_path,
                 fold,
                 n_folds,
                 subset,
                 features_type='mfcc',
                 spatial_transform=None, data_type='audiovisual', audio_transform=None,
                 get_loader=get_default_video_loader, ):

        train_set, val_set, test_set = make_dataset(fold, n_folds, annotation_path)

        if subset == 'train':
            self.data = train_set
        elif subset == 'val':
            self.data = val_set
        elif subset == 'test':
            self.data = test_set
        else:
            raise ValueError("split_type must be 'train', 'val' or 'test'.")
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type
        self.features_type = features_type  # or mfcc or fbank

    def __getitem__(self, index):
        target = self.data[index]['label']

        if self.data_type == 'video' or self.data_type == 'audiovisual':
            # 获取视频路径和加载视频数据
            path = self.data[index]['video_path']
            clip = self.loader(path)

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()  # 随机化变换的参数，确保每个视频帧应用相同的随机变换。
                clip = [self.spatial_transform(img) for img in clip]
            # torch.stack(clip, 0) 将多个帧（图像）堆叠成一个四维张量。第一个维度表示帧的数量。
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # 重新排列张量的维度

            if self.data_type == 'video':
                return clip, target

        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=16000)

            if self.audio_transform is not None:  # 这里没有
                self.audio_transform.randomize_parameters()
                y = self.audio_transform(y).astype(np.float32)

            if self.features_type == 'mfcc':
                audio_features = get_mfccs(y, sr)
            elif self.features_type == 'fbank':
                audio_features = get_fbank(y, sr)
            elif self.features_type == 'multi':
                audio_features = get_multifeatures(y, sr)
            else:
                audio_features = y

            if self.data_type == 'audio':
                return audio_features, target

        if self.data_type == 'audiovisual':
            return audio_features, clip, target

    def __len__(self):
        return len(self.data)
