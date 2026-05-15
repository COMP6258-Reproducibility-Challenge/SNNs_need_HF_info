"""
alt_operator_former.py  —  Experiment 2.2: Alternative High-Frequency Operators
=================================================================================
Registers three additional timm models that extend the Exp 1.1 ablation by
replacing the Max-Pool token mixer with other operators that plausibly
restore high-frequency information:

    pool_former_min         Min-Pool token mixer
    pool_former_lap         Fixed Laplacian high-pass filter (non-trainable)
    pool_former_learned_hp  Learned depthwise conv, Laplacian-initialised

Architecture — 3-stage multi-scale (identical to the updated pooling_former.py):
----------------------------------------------------------------------------------
  Stage 1: Embed_Orig(in_ch → D/4=96,   32×32) → 1 × Block_<op>(96)
  Stage 2: Embed_Max (D/4 → D/2=192,    16×16) → 1 × Block_<op>(192)
  Stage 3: Embed_Max (D/2 → D=384,       8×8)  → 2 × Block_<op>(384)
  Head:    flatten+mean(spatial) → LIF → mean(T) → Linear(384, num_classes)

The ONLY variable across pool_former_{avg,max,min,lap,learned_hp} is the
block-level token mixer.  Embedding stages are identical so Exp 2.2 results
are directly comparable with Exp 1.1.

Scientific motivation (addresses reviewer weakness W1):
  If ALL high-pass operators beat AvgFormer  → the HF Fourier explanation holds.
  If only Max-Pool wins                      → Max-Pool has advantages beyond HP.

This file must be placed inside MaxFormer/cifar10-100/ alongside pooling_former.py.
The SLURM scripts handle the copy automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg

try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
except ImportError:
    raise ImportError("spikingjelly not found. pip install spikingjelly==0.0.0.0.12")

# These modules live in MaxFormer/cifar10-100/ alongside this file
from mixer_hub import S_MLP
from embedding_hub import Embed_Orig, Embed_Max


# ── Token mixers ──────────────────────────────────────────────────────────────

class Min_Mixer(nn.Module):
    """
    Min-Pool token mixer.  Implemented as -MaxPool(-x) to reuse the
    optimised CUDA MaxPool kernel.  Applied over a 3×3 neighbourhood,
    stride=1, preserving spatial dimensions.

    Min-pool over (0/1) binary spikes = logical AND over the neighbourhood.
    Over continuous feature maps it picks local minima — preserves extreme
    values (low-freq is NOT averaged away), acting as an edge/contrast op.
    """
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.flatten(0, 1).contiguous()
        x_flat = -self.pool(-x_flat)   # min = -max(-x)
        return x_flat.reshape(T, B, -1, H, W).contiguous()


class Laplacian_Mixer(nn.Module):
    """
    Fixed (non-trainable) depthwise Laplacian high-pass filter.

    Each channel is convolved with the 8-connected discrete Laplacian:
        [[-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]]  / 8.0

    Output is added back to the input (unsharp masking):
        x_out = x + BN(laplacian(x))

    The kernel is a buffer (not a parameter) — never updated during training,
    making this a controlled ablation of a FIXED high-pass operator.
    BN stabilises the Laplacian response scale.
    """
    def __init__(self, dim):
        super().__init__()
        kernel = torch.tensor(
            [[-1., -1., -1.],
             [-1.,  8., -1.],
             [-1., -1., -1.]], dtype=torch.float32
        ) / 8.0
        w = kernel.view(1, 1, 3, 3).expand(dim, 1, 3, 3).clone()
        self.register_buffer('weight', w)
        self.bn  = nn.BatchNorm2d(dim)
        self.dim = dim

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.flatten(0, 1).contiguous()
        lap = F.conv2d(x_flat, self.weight, padding=1, groups=self.dim)
        lap = self.bn(lap)
        return (x_flat + lap).reshape(T, B, -1, H, W).contiguous()


class LearnedHP_Mixer(nn.Module):
    """
    Learned depthwise 3×3 conv initialised as a 4-connected Laplacian:
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]  / 4.0

    The kernel is a trainable nn.Parameter — the network adapts the filter
    during training.  Key question: does learning beat the fixed Laplacian?
    If yes, the network found a better HP filter for spike statistics.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                              groups=dim, bias=False)
        self.bn   = nn.BatchNorm2d(dim)
        self.dim  = dim
        self._init_laplacian()

    def _init_laplacian(self):
        lap = torch.tensor(
            [[ 0., -1.,  0.],
             [-1.,  4., -1.],
             [ 0., -1.,  0.]], dtype=torch.float32
        ) / 4.0
        with torch.no_grad():
            for i in range(self.conv.weight.shape[0]):
                self.conv.weight[i, 0] = lap

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.flatten(0, 1).contiguous()
        hp = self.bn(self.conv(x_flat))
        return (x_flat + hp).reshape(T, B, -1, H, W).contiguous()


# ── Block wrappers (mixer + S_MLP, dim-parameterised for multi-stage use) ─────

class Block_Min(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Min_Mixer(dim)
        self.mlp   = S_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        return self.mlp(self.mixer(x))


class Block_Lap(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = Laplacian_Mixer(dim)
        self.mlp   = S_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        return self.mlp(self.mixer(x))


class Block_LearnedHP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.mixer = LearnedHP_Mixer(dim)
        self.mlp   = S_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        return self.mlp(self.mixer(x))


_BLOCK_MAP = {
    'min':        Block_Min,
    'lap':        Block_Lap,
    'learned_hp': Block_LearnedHP,
}


# ── Model ─────────────────────────────────────────────────────────────────────

class AltPoolFormer(nn.Module):
    """
    3-stage multi-scale spiking transformer whose token mixer is one of
    the alternative high-pass operators: Min-Pool, Laplacian, or Learned-HP.

    Architecture is identical to the updated PoolFormer in pooling_former.py so
    Exp 2.2 results are directly comparable with Exp 1.1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dims: int = 384,
        mlp_ratios: int = 4,
        drop_rate: float = 0.0,
        T: int = 4,
        operator: str = 'lap',
        **kwargs,           # absorb depths, num_heads, img_size, etc.
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        BlockCls = _BLOCK_MAP.get(operator, Block_Lap)
        D = embed_dims      # 384

        # ── Stage 1: 32×32 → 32×32,  channels 3 → D/4 = 96 ─────────────────
        self.patch_embed1 = Embed_Orig(
            in_channels=in_channels, embed_dims=D // 4
        )
        self.stage1 = nn.ModuleList([
            BlockCls(dim=D // 4, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # ── Stage 2: 32×32 → 16×16,  channels D/4 → D/2 = 192 ──────────────
        self.patch_embed2 = Embed_Max(
            in_channels=D // 4, embed_dims=D // 2
        )
        self.stage2 = nn.ModuleList([
            BlockCls(dim=D // 2, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # ── Stage 3: 16×16 → 8×8,    channels D/2 → D = 384 ─────────────────
        self.patch_embed3 = Embed_Max(
            in_channels=D // 2, embed_dims=D // 1
        )
        self.stage3 = nn.ModuleList([
            BlockCls(dim=D // 1, mlp_ratio=float(mlp_ratios))
            for _ in range(2)
        ])

        # ── Head (matches max_former.py / updated pooling_former.py) ─────────
        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.head = nn.Linear(D, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        # NOTE: LearnedHP_Mixer's conv weight is initialised in its own
        # _init_laplacian() — we must NOT override it here.

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)        # (T, B, D/4, 32, 32)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)        # (T, B, D/2, 16, 16)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)        # (T, B, D,   8,  8)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3)     # (T, B, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)   # (T, B, C, H, W)
        x = self.forward_features(x)                       # (T, B, D)
        x = self.head_lif(x)                               # (T, B, D)
        x = self.head(x.mean(0))                           # (B, num_classes)
        return x


# ── timm registrations ────────────────────────────────────────────────────────

@register_model
def pool_former_min(pretrained: bool = False, pretrained_cfg=None,
                    **kwargs) -> AltPoolFormer:
    """MinPool token mixer.  Expected: between AvgFormer and MaxFormer-lite."""
    model = AltPoolFormer(operator='min', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_lap(pretrained: bool = False, pretrained_cfg=None,
                    **kwargs) -> AltPoolFormer:
    """Fixed Laplacian HP token mixer (non-trainable kernel)."""
    model = AltPoolFormer(operator='lap', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_learned_hp(pretrained: bool = False, pretrained_cfg=None,
                           **kwargs) -> AltPoolFormer:
    """Learned depthwise conv, Laplacian-initialised HP token mixer."""
    model = AltPoolFormer(operator='learned_hp', **kwargs)
    model.default_cfg = _cfg()
    return model


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Sanity check: alt_operator_former.py (3-stage multi-scale architecture)")
    print("=" * 60)
    configs = [
        ('min',        'Min-Pool'),
        ('lap',        'Fixed Laplacian'),
        ('learned_hp', 'Learned HP'),
    ]
    for op, desc in configs:
        m = AltPoolFormer(
            operator=op, in_channels=3, num_classes=100,
            embed_dims=384, mlp_ratios=4, T=4
        )
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {desc:25s}: input {list(x.shape)} → output {list(y.shape)},  params={n:,}")
    print("PASSED")
