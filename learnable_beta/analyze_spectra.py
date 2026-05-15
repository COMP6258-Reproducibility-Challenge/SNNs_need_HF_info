"""
analyze_spectra.py — Priority 2 / Priority 4: Temporal and Spatial HFR
=======================================================================
Loads a trained checkpoint (best.pt) and runs a single forward pass on a
fixed 64-image held-out batch from CIFAR-100 val (no augmentation).

Measures per LIF-layer:
  T-HFR  -- temporal high-frequency ratio (1D DFT along time axis, T=4 bins)
  S-HFR  -- spatial high-frequency ratio  (2D DFT, r_norm > 0.5)

Notes v2 §4 definitions implemented exactly.

Usage:
  python analyze_spectra.py \\
      --checkpoint results/p1_maxformer_learnable_b0/best.pt \\
      --model max_former \\
      --source-run p1 \\
      --data-path /path/to/data \\
      --output-dir results/p2_spectra

  python analyze_spectra.py \\
      --checkpoint results/p3_spikformer_learnable_b0/best.pt \\
      --model spikformer \\
      --source-run p3 \\
      --data-path /path/to/data \\
      --output-dir results/p4_spectra
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

_SCRIPT_DIR = Path(__file__).resolve().parent
_MAXFORMER_DIR = _SCRIPT_DIR.parent / 'MaxFormer' / 'cifar10-100'
sys.path.insert(0, str(_MAXFORMER_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import spikingjelly.clock_driven.neuron as _sj_n
from plif_wrapper import ParametricMultiStepLIFNode, collect_betas
_sj_n.MultiStepLIFNode = ParametricMultiStepLIFNode

from timm.models import create_model
import max_former
import spikformer as _spf

from spikingjelly.clock_driven import functional as sj_functional

HELD_OUT_N = 64

def temporal_hfr(v: np.ndarray) -> tuple:
    """
    v shape: [T, B, C, H, W]
    Returns (scalar T-HFR, 1D energy array of shape [T]).
    Notes v2 §4.4: high-freq band = bins ceil(T/2) to T-1.
    """
    T = v.shape[0]
    import math
    high_start = math.ceil(T / 2)

    fft = np.fft.fft(v, axis=0)
    energy = np.abs(fft) ** 2

    energy_1d = energy.mean(axis=(1, 2, 3, 4))

    dc_excl = energy_1d[1:].sum()
    hi_band = energy_1d[high_start:].sum()

    if dc_excl < 1e-30:
        t_hfr = float('nan')
    else:
        t_hfr = float(hi_band / dc_excl)

    return t_hfr, energy_1d

def spatial_hfr(v: np.ndarray) -> tuple:
    """
    v shape: [T, B, C, H, W]
    Returns (scalar S-HFR, 1D radial energy profile of shape [32]).
    Notes v2 §4.4: high-freq = r_norm > 0.5 after fftshift.
    """
    T, B, C, H, W = v.shape
    n_radial_bins = 32

    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    r_norm = np.sqrt((KX ** 2 + KY ** 2)) / 0.5

    total_energy = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        for b in range(B):
            for c in range(C):
                frame = v[t, b, c]
                fft2 = np.fft.fftshift(np.fft.fft2(frame))
                total_energy += np.abs(fft2) ** 2

    dc_mask = np.zeros((H, W), dtype=bool)
    dc_mask[H // 2, W // 2] = True

    dc_energy = total_energy[dc_mask].sum()
    total_excl_dc = total_energy[~dc_mask].sum()
    high_energy = total_energy[(r_norm > 1.0) & (~dc_mask)].sum()

    if total_excl_dc < 1e-30:
        s_hfr = float('nan')
    else:
        s_hfr = float(high_energy / total_excl_dc)

    max_r = r_norm.max()
    bin_edges = np.linspace(0, max_r, n_radial_bins + 1)
    radial_prof = np.zeros(n_radial_bins, dtype=np.float64)
    for i in range(n_radial_bins):
        mask_bin = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1]) & (~dc_mask)
        radial_prof[i] = total_energy[mask_bin].sum()
    if radial_prof.sum() > 0:
        radial_prof /= radial_prof.sum()

    return s_hfr, radial_prof

class MembraneHook:
    """Captures membrane potential tensor from a LIF layer's forward call."""

    def __init__(self):
        self.storage: list = []
        self._handles = []

    def register(self, module: nn.Module):
        def hook(mod, inp, out):
            with torch.no_grad():
                v = mod.v
                self.storage.append(v.detach().cpu())
        h = module.register_forward_hook(hook)
        self._handles.append(h)

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

def parse_args():
    p = argparse.ArgumentParser(description='P2/P4 spectral HFR analysis')
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--model', required=True, type=str)
    p.add_argument('--source-run', required=True, type=str,
                   help='label for this source, e.g. p1 or p3')
    p.add_argument('--data-path', required=True, type=str)
    p.add_argument('--output-dir', default='results/p2_spectra', type=str)
    p.add_argument('--num-classes', default=100, type=int)
    p.add_argument('--time-step', default=4, type=int, dest='T')
    p.add_argument('--dim', default=384, type=int)
    p.add_argument('--layer', default=4, type=int)
    p.add_argument('--mlp-ratio', default=4, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f'[spectra] loading model {args.model} from {args.checkpoint}', flush=True)
    model = create_model(
        args.model, in_channels=3, num_classes=args.num_classes,
        embed_dims=args.dim, mlp_ratios=args.mlp_ratio,
        depths=args.layer, T=args.T,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print('[spectra] model loaded', flush=True)

    layer_beta_map = {}
    lif_layers = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, ParametricMultiStepLIFNode)]
    for idx, (name, m) in enumerate(lif_layers):
        with torch.no_grad():
            t = m.tau.item()
        layer_beta_map[name] = {'tau': t, 'beta': 1.0 - 1.0 / t}

    print(f'[spectra] {len(lif_layers)} LIF layers found', flush=True)

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    val_dataset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    subset = torch.utils.data.Subset(val_dataset, list(range(HELD_OUT_N)))
    loader = torch.utils.data.DataLoader(subset, batch_size=HELD_OUT_N, shuffle=False)
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f'[spectra] held-out batch: {images.shape}  (first {HELD_OUT_N} val images)',
          flush=True)

    mem_storage = {}

    handles = []
    for name, m in lif_layers:
        _name = name

        def make_hook(layer_name):
            def hook(mod, inp, out):
                with torch.no_grad():
                    if hasattr(mod, 'v_seq') and mod.v_seq is not None:
                        mem_storage[layer_name] = mod.v_seq.detach().cpu().numpy()
                    else:
                        v_final = mod.v.detach().cpu()
                        T_val = out.shape[0]
                        mem_storage[layer_name] = v_final.unsqueeze(0).repeat(
                            T_val, 1, 1, 1, 1).numpy()
            return hook

        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    print('[spectra] running forward pass...', flush=True)
    with torch.no_grad():
        _ = model(images)
    sj_functional.reset_net(model)

    for h in handles:
        h.remove()

    print(f'[spectra] captured {len(mem_storage)} layers', flush=True)

    rows = []
    for idx, (name, m) in enumerate(lif_layers):
        if name not in mem_storage:
            print(f'[spectra] WARNING: no data for layer {name}', flush=True)
            continue

        v = mem_storage[name]
        if v.ndim == 3:
            v = v[:, :, :, None, None]
        elif v.ndim == 4:
            v = v.transpose(0, 1, 3, 2)[:, :, :, None, :]
        T_val, B_val, C_val, H_val, W_val = v.shape

        t_hfr, energy_1d = temporal_hfr(v)
        s_hfr, radial_1d = spatial_hfr(v)

        linfo = layer_beta_map.get(name, {'beta': float('nan'), 'tau': float('nan')})

        rows.append({
            'source_run': args.source_run,
            'layer_index': idx,
            'layer_name': name,
            'learned_beta': linfo['beta'],
            'learned_tau': linfo['tau'],
            'T_HFR': t_hfr,
            'S_HFR': s_hfr,
            'T_timesteps': T_val,
            'spatial_H': H_val,
            'spatial_W': W_val,
        })

        run = args.source_run
        np.save(raw_dir / f'{run}_layer_{idx:02d}_temporal.npy', energy_1d)
        np.save(raw_dir / f'{run}_layer_{idx:02d}_spatial.npy', radial_1d)

        print(f'[spectra] layer {idx:2d} {name:40s}  '
              f'beta={linfo["beta"]:.4f}  T-HFR={t_hfr:.4f}  S-HFR={s_hfr:.4f}',
              flush=True)

    csv_path = out_dir / 'measurements.csv'
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'source_run', 'layer_index', 'layer_name',
            'learned_beta', 'learned_tau', 'T_HFR', 'S_HFR',
            'T_timesteps', 'spatial_H', 'spatial_W'])
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f'[spectra] saved {csv_path}', flush=True)

    print('\n[sanity] T=4 gives a coarse temporal spectrum — relative layer comparisons valid',
          flush=True)
    nan_layers = [r['layer_name'] for r in rows
                  if np.isnan(r['T_HFR']) or np.isnan(r['S_HFR'])]
    if nan_layers:
        print(f'[sanity] WARNING: NaN HFR (all-zero feature maps?) in: {nan_layers}',
              flush=True)
    else:
        print('[sanity] no NaN HFR values', flush=True)

    if rows:
        betas = [r['learned_beta'] for r in rows if not np.isnan(r['learned_beta'])]
        t_hfrs = [r['T_HFR'] for r in rows if not np.isnan(r['T_HFR'])]
        if len(betas) > 1:
            corr = np.corrcoef(betas, t_hfrs)[0, 1]
            sign = 'EXPECTED (smaller beta -> higher T-HFR)' if corr < 0 else 'CHECK'
            print(f'[sanity] beta vs T-HFR correlation={corr:.3f}  [{sign}]', flush=True)

    print('[spectra] done', flush=True)

if __name__ == '__main__':
    main()
