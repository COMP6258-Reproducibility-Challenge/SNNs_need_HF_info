"""
beta_sweep_former.py  —  Experiment 3.2: LIF membrane decay (τ) sweep
======================================================================
Registers four MaxFormer-lite models that are IDENTICAL in every way except
the LIF membrane time constant τ (tau), which controls the low-pass strength
of each spiking neuron.

LIF update rule:  v[t] = (1 - 1/τ) * v[t-1] + I[t]
  Small τ (→ τ=1.25, β≈0.20): short memory, fast decay, HIGH-PASS behaviour
  Large τ (→ τ=10.0, β≈0.90): long memory, slow decay,  LOW-PASS behaviour

Models registered:
  pool_former_beta_02   τ=1.25  β≈0.20  (strongest high-pass)
  pool_former_beta_05   τ=2.00  β=0.50  (paper default)
  pool_former_beta_075  τ=4.00  β=0.75
  pool_former_beta_09   τ=10.0  β=0.90  (strongest low-pass)

Scientific motivation (addresses reviewer weakness W1):
  If accuracy monotonically DECREASES as τ increases:
    → empirical proof that LIF low-pass filtering causes the performance gap
  The β=0.50 model (paper default) should be the best or near-best.

The tau override is applied to EVERY MultiStepLIFNode in the network after
construction (including those inside S_MLP from mixer_hub), ensuring a clean
sweep across the entire model — not just the classification head.

This file is placed inside MaxFormer/cifar10-100/ by the SLURM script so that
imports from mixer_hub and embedding_hub resolve correctly.
"""

import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg

try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
except ImportError:
    raise ImportError("spikingjelly not found. pip install spikingjelly==0.0.0.0.12")

from mixer_hub import Block_Max
from embedding_hub import Embed_Orig, Embed_Max


# ── Model ─────────────────────────────────────────────────────────────────────

class BetaPoolFormer(nn.Module):
    """
    MaxFormer-lite (Max-Pool token mixing) with a configurable LIF time
    constant τ applied uniformly to every spiking neuron in the network.

    Architecture is identical to PoolFormer(pool_type='max') in
    pooling_former.py — only τ differs.
    """

    def __init__(
        self,
        in_channels: int  = 3,
        num_classes: int  = 100,
        embed_dims: int   = 384,
        mlp_ratios: int   = 4,
        drop_rate: float  = 0.0,
        T: int            = 4,
        tau: float        = 2.0,   # LIF time constant
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        D = embed_dims      # 384

        # ── Stage 1: 32×32, channels 3 → D/4=96 ─────────────────────────────
        self.patch_embed1 = Embed_Orig(in_channels=in_channels, embed_dims=D // 4)
        self.stage1 = nn.ModuleList([
            Block_Max(dim=D // 4, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # ── Stage 2: 32→16, channels D/4 → D/2=192 ──────────────────────────
        self.patch_embed2 = Embed_Max(in_channels=D // 4, embed_dims=D // 2)
        self.stage2 = nn.ModuleList([
            Block_Max(dim=D // 2, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # ── Stage 3: 16→8, channels D/2 → D=384 ─────────────────────────────
        self.patch_embed3 = Embed_Max(in_channels=D // 2, embed_dims=D // 1)
        self.stage3 = nn.ModuleList([
            Block_Max(dim=D // 1, mlp_ratio=float(mlp_ratios))
            for _ in range(2)
        ])

        # ── Head ──────────────────────────────────────────────────────────────
        self.head_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.head = nn.Linear(D, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # ── Override ALL LIF taus uniformly ───────────────────────────────────
        # This includes LIF nodes inside S_MLP (from mixer_hub) and
        # Embed_Orig/Embed_Max (from embedding_hub), not just head_lif.
        n_lif = 0
        for m in self.modules():
            if isinstance(m, MultiStepLIFNode):
                m.tau = tau
                n_lif += 1
        print(f"[BetaPoolFormer] Set τ={tau:.2f} (β≈{1-1/tau:.3f}) on {n_lif} LIF nodes")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)
        return x.flatten(3).mean(3)   # (T, B, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x.mean(0))
        return x


# ── timm registrations ────────────────────────────────────────────────────────

@register_model
def pool_former_beta_02(pretrained: bool = False, pretrained_cfg=None,
                        **kwargs) -> BetaPoolFormer:
    """τ=1.25, β≈0.20 — shortest time constant, strongest high-pass behaviour."""
    model = BetaPoolFormer(tau=1.25, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_beta_05(pretrained: bool = False, pretrained_cfg=None,
                        **kwargs) -> BetaPoolFormer:
    """τ=2.00, β=0.50 — paper default. Reference point for the sweep."""
    model = BetaPoolFormer(tau=2.0, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_beta_075(pretrained: bool = False, pretrained_cfg=None,
                         **kwargs) -> BetaPoolFormer:
    """τ=4.00, β=0.75 — stronger low-pass filtering."""
    model = BetaPoolFormer(tau=4.0, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_beta_09(pretrained: bool = False, pretrained_cfg=None,
                        **kwargs) -> BetaPoolFormer:
    """τ=10.0, β=0.90 — strongest low-pass filtering."""
    model = BetaPoolFormer(tau=10.0, **kwargs)
    model.default_cfg = _cfg()
    return model


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Sanity check: beta_sweep_former.py")
    print("=" * 60)
    configs = [
        ('pool_former_beta_02',  1.25),
        ('pool_former_beta_05',  2.0),
        ('pool_former_beta_075', 4.0),
        ('pool_former_beta_09',  10.0),
    ]
    for name, tau in configs:
        m = BetaPoolFormer(tau=tau, in_channels=3, num_classes=100,
                           embed_dims=384, mlp_ratios=4, T=4)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        beta = 1 - 1/tau
        print(f"  {name:28s}  τ={tau:5.2f}  β={beta:.3f}")
        print(f"    input {list(x.shape)} → output {list(y.shape)},  params={n:,}")
    print("PASSED")
