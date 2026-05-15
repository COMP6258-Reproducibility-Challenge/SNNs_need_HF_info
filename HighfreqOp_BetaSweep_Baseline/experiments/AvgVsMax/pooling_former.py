# AVG vs MAX
import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg

try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
except ImportError:
    raise ImportError(
        "pip install spikingjelly"
    )

from mixer_hub import Block_Max, Block_Avg
from embedding_hub import Embed_Orig, Embed_Max


class PoolFormer(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dims: int = 384,
        mlp_ratios: int = 4,
        drop_rate: float = 0.0,
        T: int = 4,
        pool_type: str = 'max',
        **kwargs,           
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        BlockCls = Block_Max if pool_type == 'max' else Block_Avg
        D = embed_dims          

        # Stage 1: 32×32  →  32×32
        # channels 3 → D/4 = 96 
        self.patch_embed1 = Embed_Orig(
            in_channels=in_channels, embed_dims=D // 4
        )
        self.stage1 = nn.ModuleList([
            BlockCls(dim=D // 4, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # Stage 2: 32×32  →  16×16
        # channels D/4 → D/2 = 192 
        self.patch_embed2 = Embed_Max(
            in_channels=D // 4, embed_dims=D // 2
        )
        self.stage2 = nn.ModuleList([
            BlockCls(dim=D // 2, mlp_ratio=float(mlp_ratios))
            for _ in range(1)
        ])

        # Stage 3: 16×16  →  8×8
        # channels D/2 → D = 384 
        self.patch_embed3 = Embed_Max(
            in_channels=D // 2, embed_dims=D // 1
        )
        self.stage3 = nn.ModuleList([
            BlockCls(dim=D // 1, mlp_ratio=float(mlp_ratios))
            for _ in range(2)
        ])

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.head = nn.Linear(D, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # Weights
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # Forward
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        # Stage 2
        x = self.patch_embed2(x)  
        for blk in self.stage2:
            x = blk(x)

        # Stage 3
        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) 
        x = self.forward_features(x)                       
        x = self.head_lif(x)                             
        x = self.head(x.mean(0))                          
        return x


# Model registrations

@register_model
def pool_former_avg(pretrained: bool = False, pretrained_cfg=None, **kwargs) -> PoolFormer:
    """
    AvgFormer — Avg-Pool token mixing, low-pass baseline.
    Expected top-1 on CIFAR-100: ~76.73% (paper Figure 1).
    """
    model = PoolFormer(pool_type='avg', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pool_former_max(pretrained: bool = False, pretrained_cfg=None, **kwargs) -> PoolFormer:
    """
    MaxFormer-lite — Max-Pool token mixing, high-pass variant.
    Expected top-1 on CIFAR-100: ~79.12% (paper Figure 1, +2.39% over avg).
    """
    model = PoolFormer(pool_type='max', **kwargs)
    model.default_cfg = _cfg()
    return model


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Sanity check: pooling_former.py (3-stage multi-scale architecture)")
    print("=" * 60)
    for name, pool in (('pool_former_avg', 'avg'), ('pool_former_max', 'max')):
        m = PoolFormer(pool_type=pool, in_channels=3, num_classes=100,
                       embed_dims=384, mlp_ratios=4, T=4)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name}")
        print(f"    input {list(x.shape)} → output {list(y.shape)}")
        print(f"    parameters: {n:,}")
    print("PASSED")
