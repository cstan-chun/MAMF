import librosa
import numpy as np
import random
import cv2  # OpenCV用于图像增强

class AudioAugmentor:
    def __init__(self):
        self.randomize_parameters()  # 初始化随机参数

    def randomize_parameters(self):
        """生成随机参数"""
        self.noise_level = np.random.uniform(0.01, 0.05)  # 随机噪声强度
        self.volume_factor = np.random.uniform(0.7, 1.3)  # 音量调整因子
        self.pitch_shift_steps = np.random.randint(-3, 3 + 1)  # 音高变化半音
        self.time_stretch_rate = np.random.uniform(0.8, 1.2)  # 时间伸缩比例

    def add_noise(self, y):
        """添加随机高斯噪声"""
        noise = self.noise_level * np.random.randn(len(y))
        return y + noise

    def adjust_volume(self, y):
        """调整音量"""
        return y * self.volume_factor

    def pitch_shift(self, y, sr):
        """改变音高"""
        return librosa.effects.pitch_shift(y, sr, n_steps=self.pitch_shift_steps)

    def time_stretch(self, y):
        """时间伸缩（改变语速）"""
        return librosa.effects.time_stretch(y, self.time_stretch_rate)

    def apply_augmentation(self, y, sr):
        """对音频应用数据增强"""
        self.randomize_parameters()  # 每次调用时重新生成随机参数
        y = self.add_noise(y)
        y = self.adjust_volume(y)
        y = self.pitch_shift(y, sr)
        y = self.time_stretch(y)
        return y


class FacesAugmentor:
    def __init__(self, resize_shape=(224, 224), crop_range=20, rotation_angle=15):
        self.resize_shape = resize_shape
        self.crop_range = crop_range
        self.rotation_angle = rotation_angle

    def augment_image(self, image):
        # 随机翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)

        # 随机旋转
        angle = random.randint(-self.rotation_angle, self.rotation_angle)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h))

        # 随机裁剪
        top = random.randint(0, self.crop_range)
        bottom = random.randint(0, self.crop_range)
        left = random.randint(0, self.crop_range)
        right = random.randint(0, self.crop_range)
        image = image[top:h - bottom, left:w - right]

        # 尺寸检查与缩放
        if image.shape[0] < self.resize_shape[0] or image.shape[1] < self.resize_shape[1]:
            image = cv2.resize(image, self.resize_shape)

        return image

    def augment_sequence(self, frames):
        augmented_frames = [self.augment_image(frame) for frame in frames]
        return np.array(augmented_frames)
