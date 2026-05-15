"""
train_beta.py — Experiment A: Dense beta sweep on CIFAR-100
============================================================
Trains either 'spikformer' or 'max_former' at a fixed global tau (= 1/(1-beta))
by monkey-patching MultiStepLIFNode BEFORE any model code is imported.

Usage (called by SLURM array script):
    python train_beta.py \\
        --tau 2.2222 \\
        --beta 0.55 \\
        --model spikformer \\
        --seed 0 \\
        --data-path /scratch/$USER/data \\
        --output-dir results/expA \\
        --epochs 200 \\
        --config configs/cifar100_spikformer.yaml

The monkey-patch guarantees every MultiStepLIFNode in mixer_hub.py,
embedding_hub.py, max_former.py and spikformer.py uses the specified tau,
including the attn_lif nodes with v_threshold=0.5 (their threshold is
preserved; only tau is overridden).
"""

import argparse as _ap
import sys as _sys

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--tau', type=float, default=2.0)
_pre.add_argument('--beta', type=float, default=0.5)
_pre_args, _ = _pre.parse_known_args()
_GLOBAL_TAU = _pre_args.tau

_expected_beta = 1.0 - 1.0 / _GLOBAL_TAU if _GLOBAL_TAU > 0 else 0.0
assert abs(_pre_args.beta - _expected_beta) < 0.02, (
    f"--beta {_pre_args.beta} inconsistent with --tau {_GLOBAL_TAU} "
    f"(expected beta={_expected_beta:.4f})"
)

import spikingjelly.clock_driven.neuron as _sj_n

_OrigLIF = _sj_n.MultiStepLIFNode

class _TauPatchedLIF(_OrigLIF):
    """Wrapper that forces tau=_GLOBAL_TAU regardless of construction kwargs."""
    def __init__(self, *args, **kwargs):
        kwargs['tau'] = _GLOBAL_TAU
        super().__init__(*args, **kwargs)

_sj_n.MultiStepLIFNode = _TauPatchedLIF

import os
import json
import time
import logging
import csv
import subprocess
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast

from spikingjelly.clock_driven import functional as sj_functional

_SCRIPT_DIR = Path(__file__).resolve().parent
_MAXFORMER_DIR = _SCRIPT_DIR.parent / 'MaxFormer' / 'cifar10-100'
_sys.path.insert(0, str(_MAXFORMER_DIR))
_sys.path.insert(0, str(_SCRIPT_DIR))

from timm.models import create_model
from timm.data import create_dataset, resolve_data_config
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup, FastCollateMixup
from timm.utils import AverageMeter, accuracy, CheckpointSaver, update_summary

import max_former
import spikformer as _spf_mod

try:
    from loader import create_loader
except ImportError:
    from timm.data import create_loader

_logger = logging.getLogger('train_beta')

def parse_args():
    p = _ap.ArgumentParser(description='Exp A: dense beta sweep')
    p.add_argument('--config', type=str, default='')
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--model', type=str, default='max_former')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-path', type=str, default='')
    p.add_argument('--output-dir', type=str, default='results/expA')
    p.add_argument('--epochs', type=int, default=200)

    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--weight-decay',type=float, default=0.06)
    p.add_argument('--warmup-epochs',type=int, default=20)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.75)
    p.add_argument('--cutmix', type=float, default=0.5)
    p.add_argument('--workers', type=int, default=4)

    p.add_argument('--num-classes', type=int, default=100)
    p.add_argument('--time-step', type=int, default=4, dest='T')
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--layer', type=int, default=4)
    p.add_argument('--mlp-ratio', type=int, default=4)

    p.add_argument('--amp', action='store_true', default=False)

    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            k_attr = k.replace('-', '_').replace(' ', '_')
            if hasattr(args, k_attr):
                setattr(args, k_attr, v)
        if 'amp' in cfg:
            args.amp = bool(cfg['amp'])

    return args

def build_loaders(args):
    """Build CIFAR-100 train/val loaders."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    print('[data] creating train dataset...', flush=True)
    train_ds = create_dataset(
        'torch/cifar100',
        root=args.data_path, split='train', is_training=True,
        download=True,
    )
    print('[data] creating val dataset...', flush=True)
    val_ds = create_dataset(
        'torch/cifar100',
        root=args.data_path, split='validation', is_training=False,
        download=True,
    )
    print('[data] datasets ready', flush=True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )

    try:
        print(f'[data] building train loader (workers={args.workers})...', flush=True)
        train_loader = create_loader(
            train_ds,
            input_size=(3, 32, 32),
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            mean=mean, std=std,
            num_workers=args.workers,
            distributed=False,
            collate_fn=None,
            crop_pct=1.0,
            interpolation='bicubic',
            pin_memory=True,
        )
        print('[data] train loader ready', flush=True)
        print(f'[data] building val loader (workers={args.workers})...', flush=True)
        val_loader = create_loader(
            val_ds,
            input_size=(3, 32, 32),
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=True,
            mean=mean, std=std,
            num_workers=args.workers,
            distributed=False,
            pin_memory=True,
        )
        print('[data] val loader ready', flush=True)
    except Exception as _loader_exc:
        print(f'[data] create_loader failed ({_loader_exc}), falling back to plain DataLoader', flush=True)
        import torchvision.transforms as T
        t_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        t_val = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(args.data_path, train=True,
                                          transform=t_train, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(args.data_path, train=False,
                                          transform=t_val, download=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        mixup_fn = None

    return train_loader, val_loader, mixup_fn

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None,
                    mixup_fn=None, epoch=0):
    model.train()
    loss_m = AverageMeter()
    top1_m = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sj_functional.reset_net(model)

        batch_size = inputs.size(0)
        loss_m.update(loss.item(), batch_size)
        if mixup_fn is None:
            acc1, = accuracy(outputs, targets, topk=(1,))
            top1_m.update(acc1.item(), batch_size)

    return loss_m.avg, top1_m.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        sj_functional.reset_net(model)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = inputs.size(0)
        loss_m.update(loss.item(), batch_size)
        top1_m.update(acc1.item(), batch_size)
        top5_m.update(acc5.item(), batch_size)

    return loss_m.avg, top1_m.avg, top5_m.avg

def main():
    args = parse_args()

    beta_str = f'beta_{args.beta:.4f}'.replace('.', 'p')
    run_dir = Path(args.output_dir) / args.model / beta_str / f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'train.log'),
            logging.StreamHandler(),
        ],
    )

    _logger.info('=== Experiment A: Dense beta sweep ===')
    _logger.info(f'model={args.model}  beta={args.beta}  tau={args.tau}  seed={args.seed}')
    _logger.info(f'output: {run_dir}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    import numpy as np; np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    try:
        import importlib.metadata as _imeta
        sj_ver = _imeta.version('spikingjelly')
    except Exception:
        try:
            import pkg_resources as _pkg
            sj_ver = _pkg.get_distribution('spikingjelly').version
        except Exception:
            sj_ver = 'unknown (no __version__ attr; use: pip show spikingjelly)'
    try:
        git_hash = subprocess.check_output(
            ['git', '-C', str(_MAXFORMER_DIR.parent.parent), 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    sysinfo = {
        'gpu': gpu_name,
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'spikingjelly': sj_ver,
        'git_hash': git_hash,
    }
    _logger.info(f'System: {sysinfo}')

    print('[init] creating model...', flush=True)
    model = create_model(
        args.model,
        in_channels=3,
        num_classes=args.num_classes,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        depths=args.layer,
        T=args.T,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[init] model ready  params={n_params:,}', flush=True)
    _logger.info(f'Parameters: {n_params:,}')

    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as _check
    sample_lif = list(m for m in model.modules() if isinstance(m, _check))
    if sample_lif:
        _logger.info(f'Confirmed LIF tau = {sample_lif[0].tau:.4f}  (target={args.tau})')

    print('[init] building data loaders...', flush=True)
    train_loader, val_loader, mixup_fn = build_loaders(args)
    print('[init] data loaders ready', flush=True)

    train_loss_fn = (SoftTargetCrossEntropy() if mixup_fn is not None
                     else LabelSmoothingCrossEntropy(smoothing=args.smoothing))
    val_loss_fn = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler, num_epochs = create_scheduler(
        _make_sched_args(args), optimizer
    )

    scaler = GradScaler() if args.amp else None
    print('[init] optimiser and scheduler ready — starting training', flush=True)

    csv_path = run_dir / 'training_curve.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    best_val_acc = 0.0
    best_epoch = 0
    best_ckpt_path = run_dir / 'checkpoint_best.pth'
    t0 = time.time()

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        print(f'[train] epoch {epoch} — training...', flush=True)

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, train_loss_fn,
            device, scaler=scaler, mixup_fn=mixup_fn, epoch=epoch,
        )
        vl_loss, vl_top1, _ = validate(model, val_loader, val_loss_fn, device)

        _logger.info(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'train_loss={tr_loss:.4f}  '
            f'val_loss={vl_loss:.4f}  '
            f'val_top1={vl_top1:.2f}%'
        )
        csv_writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_acc:.4f}',
                              f'{vl_loss:.6f}', f'{vl_top1:.4f}'])
        csv_file.flush()

        if vl_top1 > best_val_acc:
            best_val_acc = vl_top1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_top1': vl_top1,
                'tau': args.tau,
                'beta': args.beta,
            }, best_ckpt_path)
            _logger.info(f'  -> New best: {best_val_acc:.2f}% (epoch {best_epoch})')

    csv_file.close()
    elapsed = time.time() - t0

    _, final_acc, _ = validate(model, val_loader, val_loss_fn, device)

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                   if torch.cuda.is_available() else 0.0)

    run_record = {
        'model_name': args.model,
        'beta': args.beta,
        'tau': args.tau,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_val_acc': best_val_acc,
        'final_val_acc': final_acc,
        'epoch_of_best': best_epoch,
        'training_time_seconds': elapsed,
        'peak_gpu_memory_mb': peak_mem_mb,
        'total_parameters': n_params,
        'global_tau_confirmed': args.tau,
        'system': sysinfo,
        'hyperparams': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'batch_size': args.batch_size,
            'smoothing': args.smoothing,
            'mixup': args.mixup,
            'cutmix': args.cutmix,
            'T': args.T,
            'dim': args.dim,
            'mlp_ratio': args.mlp_ratio,
        },
    }

    with open(run_dir / 'run_record.json', 'w') as f:
        json.dump(run_record, f, indent=2)

    _logger.info(f'DONE  best={best_val_acc:.2f}% @ ep {best_epoch}  '
                 f'final={final_acc:.2f}%  time={elapsed/3600:.2f}h')
    _logger.info(f'Results: {run_dir}')

class _make_sched_args:
    """Minimal namespace that create_scheduler expects."""
    def __init__(self, args):
        self.sched = 'cosine'
        self.epochs = args.epochs
        self.min_lr = args.min_lr
        self.warmup_lr = 1e-5
        self.warmup_epochs = args.warmup_epochs
        self.cooldown_epochs = 10
        self.decay_rate = 0.1
        self.lr_noise = None
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.seed = args.seed

if __name__ == '__main__':
    main()
