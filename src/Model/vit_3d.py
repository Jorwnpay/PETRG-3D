import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .position_encoding import PositionEmbeddingLearned3d

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) # 512, 512
        patch_height, patch_width = pair(image_patch_size) # 32, 32

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size # patch_dim: 3x32x32x4=10288

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # 把三维图像切成小patch。原图shape：[9, 3, 256, 256, 64]，patch_height：32，patch_width：32，frame_patch_size：4
        # 先切块+展平：[b=9, c=3, (h=8 p1=32), (w=8 p2=32), (f=16 pf=4)] -> [9, 1024, 12288]
        # 即共切成8x8x16共1024个patch，每个patch展平后变成12288维向量
        # 然后经过LayerNorm和线性层，变换为dim=768维向量 -> [9, 1024, 768] 
        # 即把每一个patch映射为768维向量
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 输入参数：256, 16, 16, 128
        self.pos_embedding = PositionEmbeddingLearned3d(dim // 3,(image_height // patch_height), (image_width // patch_width), (frames // frame_patch_size))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        B, C, H, W, D = video.shape # [9, 3, 256, 256, 64]
        x = self.to_patch_embedding(video) # [9, 1024, 768]
        b, n, _ = x.shape
        # [9, 256//32, 256//32, 64//4, x] -> [9, 8, 8, 16, x]
        pos = self.pos_embedding(B, H // self.patch_height, W // self.patch_width, D // self.frame_patch_size, x) # [9, 1024, 768]
        x += pos 
        x = self.dropout(x)

        x = self.transformer(x) # [9, 1024, 768]
        return x,pos
