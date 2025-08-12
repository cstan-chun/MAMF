# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

import numpy as np
import random
import cv2  # OpenCV用于图像增强


# 1. 数据增强：图像翻转、旋转、随机裁剪
def augment_image(image):
    # 随机翻转
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # 1 为水平翻转

    # 随机旋转
    angle = random.randint(-15, 15)  # 随机旋转角度
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h))

    # 随机裁剪
    top = random.randint(0, 20)  # 随机上裁剪
    bottom = random.randint(0, 20)  # 随机下裁剪
    left = random.randint(0, 20)  # 随机左裁剪
    right = random.randint(0, 20)  # 随机右裁剪
    image = image[top:h - bottom, left:w - right]

    # 如果裁剪后尺寸小于224x224，则通过填充补齐
    if image.shape[0] < 224 or image.shape[1] < 224:
        image = cv2.resize(image, (224, 224))
    noise_level = np.random.uniform(0.01, 0.05)  # 随机噪声强度
    noise = noise_level * np.random.randn((image.shape))

    return image


# 2. 对所有帧进行增强
def augment_sequence(numpy_faces):
    augmented_faces = []
    for face in numpy_faces:
        augmented_face = augment_image(face)
        augmented_faces.append(augmented_face)
    return np.array(augmented_faces)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    mtcnn = MTCNN(image_size=(720, 1280), device=device)

    # mtcnn.to(device)
    save_frames = 15
    input_fps = 30

    save_length = 3.6  # seconds
    save_avi = False

    failed_videos = []
    root = r'/datasets/RAVDESS'
    output_path = r'/datasets/RAVDESS_croppad'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    n_processed = 0
    for sess in tqdm(sorted(os.listdir(root))):
        for filename in os.listdir(os.path.join(root, sess)):

            if filename.endswith('.mp4') and filename.split('-')[0] == '01':

                cap = cv2.VideoCapture(os.path.join(root, sess, filename))
                # calculate length in frames
                framen = 0
                while True:
                    i, q = cap.read()
                    if not i:
                        break
                    framen += 1
                cap = cv2.VideoCapture(os.path.join(root, sess, filename))

                if save_length * input_fps > framen:
                    skip_begin = int((framen - (save_length * input_fps)) // 2)
                    for i in range(skip_begin):
                        _, im = cap.read()

                framen = int(save_length * input_fps)
                frames_to_select = select_distributed(save_frames, framen)
                save_fps = save_frames // (framen // input_fps)
                if save_avi:
                    out = cv2.VideoWriter(os.path.join(root, sess, filename[:-4] + '_facecroppad.avi'),
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

                numpy_video = []
                success = 0
                frame_ctr = 0

                while True:
                    ret, im = cap.read()
                    if not ret:
                        break
                    if frame_ctr not in frames_to_select:
                        frame_ctr += 1
                        continue
                    else:
                        frames_to_select.remove(frame_ctr)
                        frame_ctr += 1

                    try:
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    except:
                        failed_videos.append((sess, i))
                        break

                    temp = im[:, :, -1]
                    im_rgb = im.copy()
                    im_rgb[:, :, -1] = im_rgb[:, :, 0]
                    im_rgb[:, :, 0] = temp

                    im_rgb = torch.tensor(im_rgb)
                    im_rgb = im_rgb.to(device)

                    bbox = mtcnn.detect(im_rgb)
                    if bbox[0] is not None:
                        bbox = bbox[0][0]
                        bbox = [round(x) for x in bbox]
                        x1, y1, x2, y2 = bbox
                    im = im[y1:y2, x1:x2, :]
                    im = cv2.resize(im, (224, 224))
                    if save_avi:
                        out.write(im)
                    numpy_video.append(im)
                if len(frames_to_select) > 0:
                    for i in range(len(frames_to_select)):
                        if save_avi:
                            out.write(np.zeros((224, 224, 3), dtype=np.uint8))
                        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                if save_avi:
                    out.release()

                # 数据增强
                augmented_faces = augment_sequence(numpy_video)

                faces_output_path = os.path.join(output_path, sess)
                augmented_output_path = os.path.join(output_path, sess)
                if not os.path.exists(faces_output_path):
                    os.makedirs(faces_output_path)
                if not os.path.exists(augmented_output_path):
                    os.makedirs(augmented_output_path)

                np.save(os.path.join(faces_output_path, filename[:-4] + '_facecroppad.npy'), np.array(numpy_video))
                np.save(os.path.join(augmented_output_path, filename[:-4] + '_faceaugmented.npy'), augmented_faces)

                if len(numpy_video) != 15:
                    print('Error', sess, filename)

        n_processed += 1
        # with open('processed.txt', 'a') as f:
        # f.write(sess + '\n')
        print(failed_videos)


if __name__ == '__main__':
    main()
