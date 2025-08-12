import os

root = r'/datasets/RAVDESS_croppad'
augmented = True
# annotation_file = f'annotations_fold{fold + 1}.txt'
if augmented:
    annotation_file = f'annotations_augmented.txt'
else:
    annotation_file = f'annotations.txt'
with open(annotation_file, 'w') as f:
    for _, actor in enumerate(os.listdir(root)):
        actor_path = os.path.join(root, actor)

        if not os.path.isdir(actor_path):
            continue

        for video in os.listdir(actor_path):
            if not video.endswith('.npy') or '_face' not in video:
                continue

            if not augmented:
                if 'augmented' in video:
                    continue
                else:
                    audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'
            else:
                if 'augmented' in video:
                    audio = '03' + video.split('_face')[0][2:] + '_augmented.wav'
                else:
                    audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'

            # 01表示音视频，02表示纯视频，03表示纯音频，01跟02我们只要其中一个
            if not video.split('-')[0] == '01':
                continue

            label = str(int(video.split('-')[2]))

            video_path = os.path.join(actor_path, video)
            audio_path = os.path.join(actor_path, audio)

            f.write(f"{video_path};{audio_path};{label}\n")
