from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, proj_drop=0.1, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = feature_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.kv = nn.Linear(feature_dim, feature_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkmatrix = None

    def forward(self, q, k):
        B, Nk, Ck = k.shape
        B, Nq, Cq = q.shape
        q = self.q(q).reshape(B, Nq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.kv(k).reshape(B, Nk, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q.squeeze(0)

        t_attn = (q @ k.transpose(-2, -1)) * self.scale  # [b,h,30,30]
        f_attn = (q.transpose(-2, -1) @ k) * self.scale  # [b,h,128,128]

        t_attn = t_attn.softmax(dim=-1)
        f_attn = f_attn.softmax(dim=-1)

        # self.qkmatrix = attn
        t_attn = self.attn_drop(t_attn)
        f_attn = self.attn_drop(f_attn)

        t = (t_attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        f = (f_attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, Nq, -1)
        x = self.proj(t+f)
        x = self.proj_drop(x)

        return x
