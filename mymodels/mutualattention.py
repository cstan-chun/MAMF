import torch
import torch.nn as nn

from mymodels.visualencoder import EfficientFaceTemporal, conv1d_block
from mymodels.acousticsencoder import AudioCNNPool, conv1d_block_audio
from mymodels.transformer_timm import SelfAttention
from mymodels.crossattention import CrossAttention


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, proj_drop=0.1):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drop = nn.Dropout(proj_drop)
        self.lynorm = nn.LayerNorm(hidden_size)

    def forward(self, a):
        a = self.lynorm(a)
        z = torch.sigmoid(self.drop(self.fc(a)))
        final_rep = z * a
        return final_rep


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, proj_drop=0.1):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-2)
        self.lynorm1 = nn.LayerNorm(hidden_size)
        self.lynorm2 = nn.LayerNorm(hidden_size)

    def forward(self, a, b):
        a_new = self.lynorm1(a)
        b_new = self.lynorm2(b)
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        utters = torch.cat([a_new, b_new], dim=-2)
        utters_fc = self.drop(torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2)], dim=-2))
        utters_softmax = self.softmax(utters_fc)
        utters_multi_model = utters_softmax * utters
        final_rep = torch.sum(utters_multi_model, dim=-2, keepdim=False)
        return final_rep


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)


class MutualAttention(nn.Module):
    def __init__(self, num_classes=8, feature_dim=128, seq_length=15, pretr_ef='None', num_heads=1,
                 audio_features='mfcc', dropout=0.1):
        super(MutualAttention, self).__init__()
        self.audio_model = AudioCNNPool(feature_dim=feature_dim, audio_features=audio_features)

        self.visual_model = EfficientFaceTemporal(stages_repeats=[4, 8, 4],
                                                  stages_out_channels=[29, 116, 232, 464, 1024],
                                                  im_per_sample=seq_length, feature_dim=feature_dim)

        init_feature_extractor(self.visual_model, pretr_ef)  # 初始化视觉特征提取器，加载预训练权重

        self.aa = SelfAttention(in_dim_k=feature_dim, in_dim_q=feature_dim, out_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)
        self.vv = SelfAttention(in_dim_k=feature_dim, in_dim_q=feature_dim, out_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)
        #self.aa = CrossAttention(feature_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)
        #self.vv = CrossAttention(feature_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)
        self.av = CrossAttention(feature_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)
        self.va = CrossAttention(feature_dim=feature_dim, num_heads=num_heads, proj_drop=dropout)

        self.uni_gate_a = Unimodal_GatedFusion(feature_dim, proj_drop=dropout)
        self.uni_gate_v = Unimodal_GatedFusion(feature_dim, proj_drop=dropout)
        self.mul_gate = Multimodal_GatedFusion(feature_dim, proj_drop=dropout)

        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward(self, x_audio, x_visual):
        # print(f"x_audio:{x_audio.shape},x_visual:{x_visual.shape}")
        return self.forward_feature(x_audio, x_visual)

    def forward_feature(self, x_audio, x_visual):
        x_audio = self.audio_model(x_audio)
        x_visual = self.visual_model(x_visual)
        # print(f"x_audio:{x_audio.shape},x_visual:{x_visual.shape}")

        # 原始输入
        proj_a = x_audio.permute(0, 2, 1)  # [2,30,128]
        proj_v = x_visual.permute(0, 2, 1)  # [2,30,128]

        # 自注意力
        att_aa, _ = self.aa(proj_a, proj_a)
        att_vv, _ = self.vv(proj_v, proj_v)
        # 模态内自适应融合
        proj_aa = self.uni_gate_a(proj_a + att_aa)
        proj_vv = self.uni_gate_v(proj_v + att_vv)

        # 交叉注意力
        att_av = self.av(proj_aa, proj_vv)
        att_va = self.va(proj_vv, proj_aa)

        # 模态间自适应融合
        mutual_att = self.mul_gate(proj_aa + att_av, proj_vv + att_va)

        # 加权互补信息
        mutual_audio = proj_a + mutual_att
        mutual_video = proj_v + mutual_att
        # [2,30,128]
        # ================================================================

        mutual_audio = mutual_audio.permute(0, 2, 1)
        mutual_video = mutual_video.permute(0, 2, 1)

        # 时间平均池化
        audio_pooled = mutual_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = mutual_video.mean([-1])

        # 拼接
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        # 分类
        x1 = self.classifier_1(x)
        return x1
