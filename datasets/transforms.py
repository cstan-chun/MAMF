'''
Parts of this code are based on https://github.com/okankop/Efficient-3DCNNs
'''


import numbers
import torch
from PIL import Image
try:
    import accimage # 用于加速图像加载
except ImportError:
    accimage = None

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        #print(img.size(), img.float().div(self.norm_value)) 
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass




class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomRotate(object):

    def __init__(self):
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)

        return ret_img

    def randomize_parameters(self):
        self.rotate_angle = random.randint(-30, 30)


import torchvision.transforms.functional as F


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5))

    def _check_input(self, value, name, center=1, bound=(0, float('inf'))):
        # 检查并标准化输入参数
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} should be non-negative.")
            value = [center - value, center + value]
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [max(bound[0], value[0]), min(bound[1], value[1])]
        else:
            raise ValueError(f"{name} should be a float or tuple of length 2.")
        return value

    def __call__(self, img):
        # 随机生成调整因子
        transforms = []
        if self.brightness:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(lambda img: F.adjust_brightness(img, brightness_factor))
        if self.contrast:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(lambda img: F.adjust_contrast(img, contrast_factor))
        if self.saturation:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(lambda img: F.adjust_saturation(img, saturation_factor))
        if self.hue:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(lambda img: F.adjust_hue(img, hue_factor))

        # 随机打乱顺序
        random.shuffle(transforms)

        # 依次应用调整
        for transform in transforms:
            img = transform(img)
        return img

    def randomize_parameters(self):
        pass

import numpy as np
import librosa

import numpy as np
import librosa
import random

class AddNoise:
    def __init__(self, min_noise=0.01, max_noise=0.05):
        """
        初始化噪声增强器
        :param min_noise: 最小噪声强度
        :param max_noise: 最大噪声强度
        """
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.noise_level = None  # 用于存储当前随机噪声强度
        self.randomize_parameters()  # 初始化时先随机化参数

    def randomize_parameters(self):
        """随机化噪声强度"""
        self.noise_level = np.random.uniform(self.min_noise, self.max_noise)

    def __call__(self, waveform):
        """
        向音频添加随机噪声
        :param waveform: 输入音频波形 (numpy array)
        :return: 添加噪声后的音频
        """
        noise = self.noise_level * np.random.randn(len(waveform))  # 生成随机噪声
        return waveform + noise


class PitchShift:
    def __init__(self, min_steps=-3, max_steps=3):
        """
        初始化音高增强器
        :param min_steps: 最小音高变化 (向下)
        :param max_steps: 最大音高变化 (向上)
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.n_steps = None  # 用于存储当前音高变化步数
        self.randomize_parameters()  # 初始化时先随机化参数

    def randomize_parameters(self):
        """随机化音高变化步数"""
        self.n_steps = random.randint(self.min_steps, self.max_steps)

    def __call__(self, waveform, sr=16000):
        """
        进行随机音高变换
        :param waveform: 输入音频波形 (numpy array)
        :param sr: 采样率
        :return: 变换后的音频
        """
        return librosa.effects.pitch_shift(waveform, sr, n_steps=self.n_steps)



