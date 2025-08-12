from PIL import Image
import functools
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler


def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    video_data = []  # video 是一个多维数组，通常形状为 (frames, height, width, channels)
    for i in range(np.shape(video)[0]):
        # 通过遍历视频的每一帧，将每帧（即一个二维图像）转换为 PIL.Image 对象（使用 Image.fromarray）
        video_data.append(Image.fromarray(video[i, :, :, :]))
    return video_data


# 这个函数返回的是一个基于 video_loader 函数的部分应用函数（实际上没有固定任何参数）
def get_default_video_loader():
    return functools.partial(video_loader)


# 加载一个音频文件，并返回音频信号数据和采样率。
def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr)  # sr：表示采样率
    y = audios[0]
    return y, sr


# 从音频信号中提取梅尔频率倒谱系数（MFCCs）。
def get_mfccs(signal, sr):
    N_FFT = 2048
    N_MELS = 80
    N_MFCC = 13

    mel_spec = librosa.feature.melspectrogram(y=signal,
                                              sr=sr,
                                              n_fft=N_FFT,
                                              hop_length=512,
                                              win_length=None,
                                              n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=N_MFCC)

    delta_mfcc = librosa.feature.delta(data=mfcc)
    delta2_mfcc = librosa.feature.delta(data=mfcc, order=2)
    mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc


# 得到fbank特征
def get_fbank(y, sr):
    # 计算梅尔谱图（相当于 FBank 特征）
    n_mels = 40
    fmax = 16000  #

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

    # 转换为对数
    fbank_features = librosa.power_to_db(mel_spectrogram)
    return fbank_features


def get_multifeatures(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms_energy = librosa.feature.rms(y=y)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300)
    f0 = np.nan_to_num(f0)  # 将无声音的帧设置为0
    fbank = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=16000)
    log_fbank = librosa.power_to_db(fbank)
    combined_features = np.vstack([
        spectral_centroid,
        spectral_bandwidth,
        spectral_contrast,
        zero_crossing_rate,
        rms_energy,
        f0[np.newaxis, :],  # 基音频率 (F0)，增加一个维度与其他特征对齐
        log_fbank
    ])
    # 标准化处理（对频率维度 D 上的每个特征标准化）
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features.T).T

    return normalized_features
