from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn as nn
from transformers import logging

logging.set_verbosity_error()


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True),
                         nn.MaxPool1d(kernel_size, stride, padding))


class wave2vecModel(nn.Module):
    def __init__(self, pretrained_name="pretrained_models/wav2vec2-base-960h"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_name)
        self.wave2vec = Wav2Vec2Model.from_pretrained(pretrained_name)
        for param in self.wave2vec.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.processor(x.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        audio_features = self.wave2vec(x.squeeze()).last_hidden_state
        return audio_features


class AudioCNNPool(nn.Module):

    def __init__(self, feature_dim, audio_features):
        super(AudioCNNPool, self).__init__()

        if audio_features == 'mfcc':
            input_channels = 39
        elif audio_features == 'fbank':
            input_channels = 40
        elif audio_features == 'multi':
            input_channels = 52
        elif audio_features == 'wave2vec':
            self.pretrained_model = wave2vecModel()
            input_channels = 768

        self.audio_fetures = audio_features

        self.conv1d_0 = conv1d_block_audio(input_channels, feature_dim // 2, 5, 2, 1)
        self.conv1d_1 = conv1d_block_audio(feature_dim // 2, feature_dim, 5, 1, 0)

    def forward(self, x):  # [B,D,T]
        if self.audio_fetures == 'wave2vec':
            x = self.pretrained_model(x)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x  # [1,128,30]

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x
