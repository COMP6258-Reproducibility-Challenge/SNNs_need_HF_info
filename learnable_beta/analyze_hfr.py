"""
analyze_hfr.py — Experiment C: Disentangling temporal vs spatial frequency
===========================================================================
Loads trained checkpoints from Experiment A and measures two quantities at
every LIF layer for three beta values (low=0.1, default=0.5, high=0.95):

  T-HFR — Temporal High-Frequency Ratio (energy in upper half of time-domain DFT)
  S-HFR — Spatial  High-Frequency Ratio (energy at radial freq r > 0.5 in 2D DFT)

These are defined precisely in notes.txt Section 5.3.

Usage:
    python analyze_hfr.py \\
        --expA-dir results/expA \\
        --output-dir results/expC \\
        --data-path /scratch/$USER/data \\
        --seed 0

Produces:
  results/expC/measurements.csv
  results/expC/spectra/{model}_beta_{beta}_layer_{idx}_temporal.npy
  results/expC/spectra/{model}_beta_{beta}_layer_{idx}_spatial.npy

Limitation noted in results:
  T=4 gives only 4 temporal frequency bins: DC(0), mid(1), mid(2), high(3).
  The T-HFR contrast across beta may be coarse; we report this limitation
  in the output CSV and do not extrapolate beyond it.
"""

import sys
import os
import csv
import logging
from pathlib import Path

import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as TF

_SCRIPT_DIR = Path(__file__).resolve().parent
_MAXFORMER_DIR = _SCRIPT_DIR.parent / 'MaxFormer' / 'cifar10-100'
sys.path.insert(0, str(_MAXFORMER_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from timm.models import create_model
from spikingjelly.clock_driven import functional as sj_functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

import max_former
import spikformer

_logger = logging.getLogger('analyze_hfr')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--expA-dir', type=str, default='results/expA')
    p.add_argument('--output-dir', type=str, default='results/expC')
    p.add_argument('--data-path', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--betas', type=float, nargs='+',
                   default=[0.1, 0.5, 0.95],
                   help='Beta values to analyse (must have checkpoints in expA)')
    p.add_argument('--models', type=str, nargs='+',
                   default=['spikformer', 'max_former'])
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--layer', type=int, default=4)
    p.add_argument('--mlp-ratio', type=int, default=4)
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--num-classes', type=int, default=100)
    return p.parse_args()

def compute_T_HFR(membrane_seq: torch.Tensor) -> float:
    """
    Temporal HFR for a membrane potential sequence.

    Args:
        membrane_seq: tensor of shape [T, B, C, H, W]

    Returns:
        Scalar T-HFR averaged across (B, C, H, W).

    Definition (notes.txt §5.3):
      For each (b,c,h,w): 1D DFT along T axis → energy spectrum.
      High-freq band = bins ceil(T/2) to T-1 (upper half, excluding DC).
      T-HFR = high_energy / (total_energy - DC_energy).
      Average across all spatial/channel/batch positions.
    """

    T, B, C, H, W = membrane_seq.shape
    v = membrane_seq.detach().cpu().float().numpy()

    V = np.fft.fft(v, axis=0)
    energy = np.abs(V) ** 2

    dc_energy = energy[0]
    total_energy = energy.sum(axis=0)
    denom = total_energy - dc_energy

    hi_start = int(np.ceil(T / 2))
    hi_energy = energy[hi_start:].sum(axis=0)

    valid = denom > 1e-12
    ratio = np.where(valid, hi_energy / denom, np.nan)

    t_hfr = float(np.nanmean(ratio))
    return t_hfr

def compute_S_HFR(membrane_seq: torch.Tensor) -> tuple:
    """
    Spatial HFR for a membrane potential sequence.

    Args:
        membrane_seq: tensor of shape [T, B, C, H, W]

    Returns:
        (s_hfr_scalar, 2d_mean_spectrum) where 2d_mean_spectrum has shape [H, W].

    Definition (notes.txt §5.3):
      For each (t,b,c): 2D DFT of H×W map → fftshift → energy.
      Radial freq r = sqrt((kx/H_max)^2 + (ky/W_max)^2), normalised to [0,1].
      High-freq band: r > 0.5.
      S-HFR = high_energy / (total_energy - DC_energy).
      Average across (t, b, c).
    """

    T, B, C, H, W = membrane_seq.shape
    v = membrane_seq.detach().cpu().float().numpy()
    v_flat = v.reshape(-1, H, W)

    V2 = np.fft.fft2(v_flat, axes=(-2, -1))
    V2 = np.fft.fftshift(V2, axes=(-2, -1))
    energy2 = np.abs(V2) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(H))
    ky = np.fft.fftshift(np.fft.fftfreq(W))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    r = np.sqrt(KX**2 + KY**2)

    r_norm = r / 0.5

    hi_mask = r_norm > 1.0

    dc_idx = (H // 2, W // 2)

    dc_energy = energy2[:, dc_idx[0], dc_idx[1]]
    total_energy = energy2.sum(axis=(-2, -1))
    hi_energy = (energy2 * hi_mask[None]).sum(axis=(-2, -1))
    denom = total_energy - dc_energy

    valid = denom > 1e-12
    ratio = np.where(valid, hi_energy / denom, np.nan)

    s_hfr = float(np.nanmean(ratio))

    mean_spectrum = energy2.mean(axis=0)

    return s_hfr, mean_spectrum

def compute_mean_temporal_spectrum(membrane_seq: torch.Tensor) -> np.ndarray:
    """Return mean temporal energy spectrum, shape [T], for saving."""
    T, B, C, H, W = membrane_seq.shape
    v = membrane_seq.detach().cpu().float().numpy()
    V = np.fft.fft(v, axis=0)
    energy = np.abs(V) ** 2
    return energy.mean(axis=(1, 2, 3, 4))

class MembraneHook:
    """Captures per-timestep membrane potentials from a MultiStepLIFNode."""

    def __init__(self):
        self.potentials: list = []
        self._handle = None

    def register(self, module):
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.potentials.append(output.detach().cpu())

    def clear(self):
        self.potentials = []

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

def install_hooks(model: nn.Module) -> dict:
    """
    Install forward hooks on every MultiStepLIFNode.
    Returns dict: layer_name -> MembraneHook.
    """
    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiStepLIFNode):
            h = MembraneHook()
            h.register(module)
            hooks[name] = h
    return hooks

def remove_hooks(hooks: dict):
    for h in hooks.values():
        h.remove()

def get_fixed_batch(data_path: str, batch_size: int = 64, device='cpu'):
    """
    Return the FIRST `batch_size` images from CIFAR-100 val set, no augmentation.
    Fixed across all checkpoints for direct comparability.
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    t = TF.Compose([TF.ToTensor(), TF.Normalize(mean, std)])

    val_ds = torchvision.datasets.CIFAR100(
        root=data_path, train=False, transform=t, download=True
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)

def load_checkpoint_for_beta(expA_dir: Path, model_name: str, beta: float,
                              seed: int) -> Optional[Path]:
    beta_str = f'beta_{beta:.4f}'.replace('.', 'p')
    ckpt = expA_dir / model_name / beta_str / f'seed_{seed}' / 'checkpoint_best.pth'
    if ckpt.exists():
        return ckpt
    for s in [0, 1, 2]:
        alt = expA_dir / model_name / beta_str / f'seed_{s}' / 'checkpoint_best.pth'
        if alt.exists():
            _logger.warning(f'Seed {seed} missing for {model_name} beta={beta}; '
                            f'using seed {s}')
            return alt
    return None

def main():
    args = parse_args()
    out = Path(args.output_dir)
    spectra_dir = out / 'spectra'
    out.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(out / 'analyze.log'),
                                  logging.StreamHandler()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images, _ = get_fixed_batch(args.data_path, batch_size=args.batch_size,
                                 device=device)
    _logger.info(f'Fixed batch: {images.shape}  device={device}')

    csv_path = out / 'measurements.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'model', 'beta', 'tau', 'layer_index', 'layer_name',
        'T_HFR', 'S_HFR', 'n_samples', 'T_timesteps',
        'spatial_H', 'spatial_W',
    ])

    expA_dir = Path(args.expA_dir)

    for model_name in args.models:
        for beta in args.betas:
            tau = 1.0 / (1.0 - beta) if beta < 1.0 else float('inf')

            ckpt_path = load_checkpoint_for_beta(expA_dir, model_name, beta,
                                                  args.seed)
            if ckpt_path is None:
                _logger.error(f'No checkpoint for {model_name} beta={beta}. SKIP.')
                continue

            _logger.info(f'Loading {model_name} beta={beta} from {ckpt_path}')

            model = create_model(
                model_name,
                in_channels=3,
                num_classes=args.num_classes,
                embed_dims=args.dim,
                mlp_ratios=args.mlp_ratio,
                depths=args.layer,
                T=args.T,
            ).to(device)

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            hooks = install_hooks(model)
            layer_names = list(hooks.keys())
            _logger.info(f'  Hooked {len(hooks)} LIF layers')

            for h in hooks.values():
                h.clear()

            with torch.no_grad():
                _ = model(images)
                sj_functional.reset_net(model)

            for layer_idx, layer_name in enumerate(layer_names):
                hook = hooks[layer_name]
                if not hook.potentials:
                    _logger.warning(f'  Layer {layer_name}: no data captured')
                    continue

                out_tensor = hook.potentials[0]

                if out_tensor.ndim == 3:
                    T_sz, B, C = out_tensor.shape
                    H, W = 1, 1
                    out_5d = out_tensor.unsqueeze(-1).unsqueeze(-1)
                elif out_tensor.ndim == 4:
                    T_sz, B, C, N = out_tensor.shape
                    H_W = int(N**0.5)
                    if H_W * H_W == N:
                        H, W = H_W, H_W
                        out_5d = out_tensor.reshape(T_sz, B, C, H, W)
                    else:
                        H, W = N, 1
                        out_5d = out_tensor.unsqueeze(-1)
                elif out_tensor.ndim == 5:
                    out_5d = out_tensor
                    T_sz, B, C, H, W = out_5d.shape
                else:
                    _logger.warning(f'  Unexpected shape {out_tensor.shape}, skipping')
                    continue

                if out_5d.abs().sum() == 0:
                    _logger.warning(f'  Layer {layer_name}: all-zero output, skipping')
                    continue

                t_hfr = compute_T_HFR(out_5d)
                s_hfr_val, mean_spatial_spec = compute_S_HFR(out_5d)
                mean_temp_spec = compute_mean_temporal_spectrum(out_5d)

                stem = f'{model_name}_beta_{beta:.4f}_layer_{layer_idx:03d}'
                np.save(spectra_dir / f'{stem}_temporal.npy', mean_temp_spec)
                np.save(spectra_dir / f'{stem}_spatial.npy', mean_spatial_spec)

                _logger.info(
                    f'  [{layer_idx:3d}] {layer_name:50s}  '
                    f'T-HFR={t_hfr:.4f}  S-HFR={s_hfr_val:.4f}  '
                    f'H={H} W={W}'
                )
                csv_writer.writerow([
                    model_name, f'{beta:.4f}', f'{tau:.4f}',
                    layer_idx, layer_name,
                    f'{t_hfr:.6f}', f'{s_hfr_val:.6f}',
                    B, T_sz, H, W,
                ])
                csv_file.flush()

            remove_hooks(hooks)

        _logger.info(f'Finished {model_name}')

    csv_file.close()
    _logger.info(f'Saved: {csv_path}')

    _check_monotonicity(csv_path, args)

def _check_monotonicity(csv_path: Path, args):
    """
    Check that T-HFR is (roughly) decreasing in beta for spikformer early layers.
    Notes §5.7: 'The paper's theorem requires this.'
    Prints a warning to log if monotonicity is violated.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        sf = df[df['model'] == 'spikformer'].sort_values(['layer_index', 'beta'])
        issues = []
        for layer_idx in sf['layer_index'].unique():
            sub = sf[sf['layer_index'] == layer_idx][['beta', 'T_HFR']]
            sub = sub.sort_values('beta')
            vals = sub['T_HFR'].values
            if len(vals) >= 2:
                diffs = np.diff(vals)
                n_violations = int((diffs > 0.01).sum())
                if n_violations > len(diffs) / 2:
                    issues.append(
                        f'Layer {layer_idx}: T-HFR NOT monotone decreasing in beta '
                        f'(violations={n_violations}/{len(diffs)})'
                    )
        if issues:
            _logger.warning('SANITY CHECK FAILURES (review measurement):')
            for iss in issues:
                _logger.warning(f'  {iss}')
        else:
            _logger.info('Sanity check PASSED: T-HFR broadly decreasing in beta for spikformer')
    except ImportError:
        _logger.info('pandas not available; skipping monotonicity sanity check')

if __name__ == '__main__':
    main()
