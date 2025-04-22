import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19, dropout_rate: float = 0.1):
        super().__init__()
        self.mlps = nn.ModuleList([
            MLP(dim, embed_dim) for dim in dims
        ])
        
        # Enhanced fusion module with channel attention
        self.linear_fuse = nn.Sequential(
            ConvModule(embed_dim * 4, embed_dim),
            ChannelAttention(embed_dim)
        )
        
        # Enhanced prediction head
        self.linear_pred = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(embed_dim // 2, num_classes, 1)
        )

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        
        # Process features in parallel using torch.jit
        @torch.jit.script
        def process_features(feat, mlp, size):
            x = mlp(feat).permute(0, 2, 1).reshape(B, -1, *feat.shape[-2:])
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            
        outs = [process_features(features[i], self.mlps[i], (H, W)) 
                for i in range(len(features))]
        
        # Enhanced fusion and prediction
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(seg)
        
        return seg

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)