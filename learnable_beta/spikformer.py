"""
spikformer.py — Spikformer-4-384 for CIFAR-10/100
===================================================
Architecture mirrors Max-Former (max_former.py) but uses:
  - Embed_Orig (no MaxPool) for stage 1
  - Embed_Orig_later (strided-conv downsampling, no MaxPool) for stages 2/3
  - Block_SSA (vanilla Spiking Self-Attention) throughout — no DWC
  - Membrane Shortcut is preserved (it is inside Embed_Orig_later/Block_SSA)

Stage layout (depth-4 with embed_dim=384 base):
  Stage 1 : Embed_Orig(3  -> 96)   + 1× Block_SSA(96,  heads=8)   [32×32]
  Stage 2 : Embed_Orig_later(96  -> 192) + 1× Block_SSA(192, heads=8) [16×16]
  Stage 3 : Embed_Orig_later(192 -> 384) + 2× Block_SSA(384, heads=8) [ 8× 8]

Total blocks = 1+1+2 = 4, matching the "Spikformer-4-384" designation in Table 1.

Head: global spatial average → head_lif → Linear(384, num_classes)

All LIF nodes use whatever MultiStepLIFNode is bound at import time.  When
train_beta.py monkey-patches spikingjelly before importing this file, every
LIF inside Embed_Orig / Embed_Orig_later / Block_SSA / S_MLP / SSA picks up
the patched tau automatically.
"""

import sys
import os

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mixer_hub import Block_SSA, S_MLP
from embedding_hub import Embed_Orig, Embed_Orig_later

__all__ = ['spikformer']

class Spikformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dims: int = 384,
        mlp_ratios: int = 4,
        drop_rate: float = 0.0,
        depths: int = 4,
        T: int = 4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.T = T

        d1 = embed_dims // 4
        d2 = embed_dims // 2
        d3 = embed_dims // 1

        self.patch_embed1 = Embed_Orig(in_channels=in_channels, embed_dims=d1)
        self.stage1 = nn.ModuleList([
            Block_SSA(dim=d1, num_heads=8, mlp_ratio=mlp_ratios)
            for _ in range(1)
        ])

        self.patch_embed2 = Embed_Orig_later(in_channels=d1, embed_dims=d2)
        self.stage2 = nn.ModuleList([
            Block_SSA(dim=d2, num_heads=8, mlp_ratio=mlp_ratios)
            for _ in range(1)
        ])

        self.patch_embed3 = Embed_Orig_later(in_channels=d2, embed_dims=d3)
        self.stage3 = nn.ModuleList([
            Block_SSA(dim=d3, num_heads=8, mlp_ratio=mlp_ratios)
            for _ in range(2)
        ])

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.head = nn.Linear(d3, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x.mean(0))
        return x

@register_model
def spikformer(pretrained=False, pretrained_cfg=None, **kwargs):
    model = Spikformer(**kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    m = Spikformer(in_channels=3, num_classes=100, embed_dims=384, T=4).cuda()
    inp = torch.randn(2, 3, 32, 32).cuda()
    out = m(inp)
    print('output shape:', out.shape)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'params: {n:,}')
