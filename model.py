from torch import nn
from mymodels.mutualattention import MutualAttention


def generate_model(opt):
    assert opt.model in ['mutualattention']

    if opt.model == 'mutualattention':
        model = MutualAttention(opt.n_classes, feature_dim=opt.feature_dim,
                        seq_length=opt.sample_duration,
                        pretr_ef=opt.pretrain_path,
                        num_heads=opt.num_heads,
                        audio_features=opt.audio_features, dropout=opt.dropout)

        if opt.device != 'cpu':
            model = nn.DataParallel(model, device_ids=None)
            pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                       p.requires_grad)
            print("Total number of trainable parameters: ", pytorch_total_params)

            model = model.to(opt.device)

        return model, model.parameters()
