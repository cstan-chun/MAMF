import os
import glob
from random import random

import numpy as np
from PIL import Image
import numpy as pickle
import torch
import cv2
from scipy.io import wavfile
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
import soundfile as sf
from facenet_pytorch import MTCNN
from ravdess_preprocessing.data_augmented import AudioAugmentor, FacesAugmentor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 人脸检测
mtcnn = MTCNN(image_size=224, margin=30, keep_all=False, post_process=False, device=device)

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]


def parse_evaluation_transcript(eval_lines, transcript_lines):
    metadata = {}

    # Parse Evaluation
    for line in eval_lines:
        if line.startswith('['):
            tokens = line.strip().split('\t')
            time_tokens = tokens[0][1:-1].split(' ')
            start_time, end_time = float(time_tokens[0]), float(time_tokens[2])
            uttr_id, label = tokens[1], tokens[2]
            metadata[uttr_id] = {'start_time': start_time, 'end_time': end_time, 'label': label}

    # Parse Transcript
    trans = []
    for line in transcript_lines:
        if line.startswith('S'):
            tokens = line.split(':')
            uttr_id = tokens[0].split(' ')[0]
            if '_' not in uttr_id:
                continue
            text = tokens[-1].strip()
            try:
                metadata[uttr_id]['text'] = text
            except KeyError:
                print(f'KeyError: {uttr_id},line:{line}')
    return metadata


def pad_cut_audio(audio, sr):
    target_time = 5.0
    target_length = int(sr * target_time)
    if len(audio) < target_length:
        audio = np.array(list(audio) + [0 for i in range(target_length - len(audio))])
    else:
        remain = len(audio) - target_length
        audio = audio[remain // 2:-(remain - remain // 2)]
    return audio


def retrieve_audio(signal, sr, start_time, end_time):
    # sr = 16000
    start_idx = int(sr * start_time)
    end_idx = int(sr * end_time)
    audio_segment = signal[start_idx:end_idx]

    # 分别提取左右声道
    left_channel = pad_cut_audio(audio_segment[:, 0], sr)  # 左声道
    right_channel = pad_cut_audio(audio_segment[:, 1], sr)  # 右声道
    # 对音频进行填充或者截取

    return left_channel, right_channel, sr


def crop(imgs, target_size=224):
    # imgs.shape = (18, 480, 360, 3)
    _, h, w, _ = imgs.shape
    offset_h = h // 4
    # offset_w = (w - target_size) // 2
    imgs = imgs[:, offset_h:-offset_h, :, :]
    return imgs


def retrieve_video(video, fps, start_time, end_time, select_frames, uttrd_id):
    start_idx = int(fps * start_time)
    end_idx = int(fps * end_time)
    if end_idx >= len(video):
        end_idx = len(video)
    if (end_idx - start_idx) > select_frames:

        # images = frames[start_idx:end_idx,:,:,:]
        frames_ids = select_distributed(select_frames, (end_idx - start_idx))  # 15:要提取的帧数  108:总帧数
        frames_to_select = [x + start_idx for x in frames_ids]
        clips = video.get_batch(frames_to_select)  # [15,480,720,3]
        clips = clips.asnumpy()
        # print(f"视频总帧数：{len(video)},uttrd_id：{uttrd_id},起止帧：{start_idx}~{end_idx}:{end_idx - start_idx},clip形状：{clips.shape}")
    else:
        print(f"该样本少于{select_frames}帧")
        clips = video.get_batch(range(start_idx, end_idx))
        clips = clips.asnumpy()
        num_frames = clips.shape[0]
        if not num_frames == select_frames:
            missing_frames = select_frames - num_frames
            zero_padding = np.zeros((missing_frames, clips.shape[1], clips.shape[2], clips.shape[3]), dtype=np.uint8)
            clips = np.concatenate((clips, zero_padding), axis=0)

    img_segment_L, img_segment_R = (clips[:, :, :clips.shape[2] // 2, :], clips[:, :, clips.shape[2] // 2:, :])
    # 中心裁剪为224X224，去掉环境和旁人
    img_segment_L = crop(img_segment_L)
    img_segment_R = crop(img_segment_R)
    return img_segment_L, img_segment_R


def create_annotations(face_path, audio_path, label):
    annotation_file = f'annotations_iemocap_augment.txt'
    with open(annotation_file, 'a') as f:
        f.write(f"{face_path};{audio_path};{label}\n")


def save_create(wav_output_path, faces_output_path, uttr_id, audio, sr, faces, label, subset):
    # 将音频保存为.wav的文件
    # print(audio.dtype)
    sf.write(os.path.join(wav_output_path, uttr_id + '_' + subset + '.wav'), audio, sr)
    # 将图像帧序列保存为.npy格式文件
    # np.save(os.path.join(imgs_output_path, uttr_id + '_imgscroppad.npy'), np.array(metadata['visual']))
    # 将人脸保存为.npy格式文件
    np.save(os.path.join(faces_output_path, uttr_id + '_face' + subset + '.npy'), np.array(faces))
    # 创建annotations_immocap.txt
    create_annotations(os.path.join(faces_output_path, uttr_id + '_face' + subset + '.npy'),
                       os.path.join(wav_output_path, uttr_id + '_' + subset + '.wav'),
                       label)


def main():
    IEMOCAP_dir = r"\datasets\IEMOCAP"
    target_dir = "dialog\\EmoEvaluation"  # 每个视频对应的标签
    # audio_dir = "/sentences/wav/" # 分段音频,根据每句话划分 ，28个文件，分别对应下面的
    video_dir = "dialog\\avi\\DivX"  # 完整视频 ,28个视频
    audio_dir = "dialog\\wav"  # 每个视频的音频文件 ，28个音频
    text_dir = "dialog\\transcriptions"  # 每个视频的对话文本 ，28个文本

    output_path = r"\IEMOCAP\IEMOCAP_cropped"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    during_sum = 0.
    sample_count = 0
    select_frames = 15
    valid_emotions = {'neu', 'hap', 'sad', 'ang', 'exc', 'fru'}
    emotions_count = {'neu': 0, 'hap': 0, 'sad': 0, 'ang': 0, 'exc': 0, 'fru': 0}

    facesaugmentor = FacesAugmentor()
    audioaugmentor = AudioAugmentor()

    Sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    # Sessions = ['Session1']
    for Session in Sessions:
        avi_path = os.path.join(IEMOCAP_dir, Session, video_dir)  # video
        script_path = os.path.join(IEMOCAP_dir, Session, text_dir)  # text
        wav_path = os.path.join(IEMOCAP_dir, Session, audio_dir)  # audio
        label_path = os.path.join(IEMOCAP_dir, Session, target_dir)  # label

        # eval_fname--标签文件  script_fname--文本转录文件，读取每一个标签文件
        for eval_fname in tqdm(glob.glob(f'{label_path}/*.txt')):
            avi_fname = os.path.join(avi_path, eval_fname.split("\\")[-1].replace(".txt", ".avi"))
            wav_fname = os.path.join(wav_path, eval_fname.split("\\")[-1].replace(".txt", ".wav"))
            script_fname = os.path.join(script_path, eval_fname.split("\\")[-1])

            # 读取数据
            eval_lines = open(eval_fname).readlines()  # label
            transcript_lines = open(script_fname).readlines()  # text
            sr, signal = wavfile.read(wav_fname)  # acoustic

            # images, fps = read_video(avi_fname)

            video = VideoReader(avi_fname, ctx=cpu(0))  # visual
            fps = video.get_avg_fps()

            # 将说话的开始时间结束时间，标签，说话人，文本保存到metas

            metas = parse_evaluation_transcript(eval_lines, transcript_lines)

            faces_output_path = os.path.join(output_path, Session, eval_fname.split("\\")[-1].split(".")[0], 'face')
            if not os.path.exists(faces_output_path):
                os.makedirs(faces_output_path)
            # imgs_output_path = os.path.join(output_path, Session, eval_fname.split("\\")[-1].split(".")[0], 'imgs')
            # if not os.path.exists(imgs_output_path):
            # os.makedirs(imgs_output_path)
            wav_output_path = os.path.join(output_path, Session, eval_fname.split("\\")[-1].split(".")[0], 'wav')
            if not os.path.exists(wav_output_path):
                os.makedirs(wav_output_path)

            for uttr_id, metadata in metas.items():

                # 保留基本情绪样本neutral, happy, sad, angry, excited, frustrated
                if metadata['label'] not in valid_emotions:
                    continue

                # if os.path.exists(os.path.join(wav_output_path, uttr_id + '_croppad.wav')):
                #    continue

                # 截取说话的音频和保存
                left_channel, right_channel, sr = retrieve_audio(signal, sr, metadata['start_time'],
                                                                 metadata['end_time'])
                metadata['sr'] = sr
                # 截取图像帧和分离说话人和检测人脸
                img_segment_L, img_segment_R = retrieve_video(video, fps, metadata['start_time'], metadata['end_time'],
                                                              select_frames, uttr_id)
                metadata['fps'] = fps

                # Ses01F表示female在左边，male在右边
                if uttr_id.split('_')[0][-1] == 'F':
                    if uttr_id.split('_')[2][0] == 'F':
                        metadata['visual'] = img_segment_L
                        metadata['audio'] = left_channel
                    else:
                        metadata['visual'] = img_segment_R
                        metadata['audio'] = right_channel
                else:
                    if uttr_id.split('_')[2][0] == 'F':
                        metadata['visual'] = img_segment_R
                        metadata['audio'] = right_channel
                    else:
                        metadata['visual'] = img_segment_L
                        metadata['audio'] = left_channel

                # 识别每一帧人脸
                imgs = metadata['visual']
                faces = []
                for frame in imgs:
                    img_pil = Image.fromarray(frame)  # 转换为 PIL 格式
                    face_tensors = mtcnn(img_pil)  # 提取人脸

                    if face_tensors is not None:
                        face_np = face_tensors.permute(1, 2, 0).byte().numpy()
                        faces.append(face_np)
                metadata['faces'] = faces

                if not len(metadata['faces']) == select_frames:
                    # print(f"{uttr_id}少于15帧，不符合，跳过")
                    continue
                # print(f"uttrd_id:{uttr_id},faces:{np.shape(metadata['faces'])}")

                # during_sum += (metadata['end_time'] - metadata['start_time'])
                # sample_count += 1

                # 统计类别数
                if emotions_count[metadata['label']] <= 500:
                    emotions_count[metadata['label']] += 1
                else:
                    continue

                save_create(wav_output_path, faces_output_path, uttr_id, metadata['audio'], metadata['sr'],
                            metadata['faces'], metadata['label'], 'croppad')

                # 数据增强
                faces_augmented = facesaugmentor.augment_sequence(metadata['faces'])
                audio_augmented = audioaugmentor.apply_augmentation(metadata['audio'], metadata['sr'])
                save_create(wav_output_path, faces_output_path, uttr_id, audio_augmented, metadata['sr'],
                            faces_augmented, metadata['label'], 'augmented')

    # 总时长=30858.367699999984,总样本数=6862,平均时长=4.496993252696004
    # print(f"总时长={during_sum},总样本数={sample_count},平均时长={during_sum / sample_count}")
    # 情绪类别数量：{'neu': 1604, 'hap': 550, 'sad': 986, 'ang': 1076, 'exc': 929, 'fru': 1717}
    # print(f"情绪类别数量：{emotions_count}")


if __name__ == '__main__':
    main()
