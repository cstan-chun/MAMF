# -*- coding: utf-8 -*-

import librosa
import os
import soundfile as sf
import numpy as np


class AudioAugment:
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


def main():
    root = r'/datasets/RAVDESS'
    output_path = r'/datasets/RAVDESS_croppad'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    target_time = 5.0  # sec
    sr = 16000
    augmenter = AudioAugment()

    for actor in os.listdir(root):
        for audiofile in os.listdir(os.path.join(root, actor)):

            if not audiofile.endswith('.wav') or 'croppad' in audiofile:
                continue

            audios = librosa.core.load(os.path.join(root, actor, audiofile), sr=sr)

            y = audios[0]
            sr = audios[1]
            target_length = int(sr * target_time)
            if len(y) < target_length:
                y = np.array(list(y) + [0 for i in range(target_length - len(y))])
            else:
                remain = len(y) - target_length
                y = y[remain // 2:-(remain - remain // 2)]

            audio_output_path = os.path.join(output_path, actor)
            if not os.path.exists(audio_output_path):
                os.makedirs(audio_output_path)

            sf.write(os.path.join(audio_output_path, audiofile[:-4] + '_croppad.wav'), y, sr)

            # 在这做数据增强并保存为新的数据
            augmented_y = augmenter.apply_augmentation(y, sr)
            if len(augmented_y) < target_length:
                augmented_y = np.array(list(augmented_y) + [0 for i in range(target_length - len(augmented_y))])
            else:
                remain = len(augmented_y) - target_length
                augmented_y = augmented_y[remain // 2:-(remain - remain // 2)]

            augmented_output_path = os.path.join(output_path, actor)
            if not os.path.exists(augmented_output_path):
                os.makedirs(augmented_output_path)

            sf.write(os.path.join(augmented_output_path, audiofile[:-4] + '_augmented.wav'), augmented_y, sr)


if __name__ == '__main__':
    main()
