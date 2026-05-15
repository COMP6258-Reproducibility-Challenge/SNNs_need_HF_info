"""
train_plif.py — Priority 1 / Priority 3: Learnable-beta Max-Former / Spikformer
================================================================================
Monkey-patches MultiStepLIFNode with ParametricMultiStepLIFNode BEFORE any
model code is imported so every LIF layer in the network gets one learnable
_raw_tau scalar.

Notes v2 §3 (P1) and §5 (P3):
  - 150 epochs, seed=0, CIFAR-100
  - Checkpoint every 25 epochs; resumes automatically from latest checkpoint
  - Beta trajectory logged every 5 epochs
  - Output dirs:
      results/p1_maxformer_learnable_b0/   (max_former model)
      results/p3_spikformer_learnable_b0/  (spikformer model)

Outputs per run:
  run.json                  — summary metrics
  epoch_log.csv             — epoch, train_loss, train_top1, val_loss, val_top1, val_top5
  beta_trajectory.csv       — epoch, layer_index, layer_name, beta, tau
  learned_betas_final.csv   — per-layer initial/final/best-epoch beta
  best.pt                   — checkpoint at best val_top1
  final.pt                  — checkpoint at end of training
  checkpoint_epoch_NNN.pt   — resumable checkpoints every 25 epochs
"""

import argparse as _ap
import sys as _sys

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--init-tau', type=float, default=2.0)
_pre.add_argument('--model', type=str, default='max_former')
_pre_args, _ = _pre.parse_known_args()
_INIT_TAU = _pre_args.init_tau

import spikingjelly.clock_driven.neuron as _sj_n
from plif_wrapper import ParametricMultiStepLIFNode

_sj_n.MultiStepLIFNode = ParametricMultiStepLIFNode

import os
import csv
import json
import time
import logging
import subprocess
from pathlib import Path

import yaml
import numpy as np
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
from timm.data import create_dataset
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from timm.utils import AverageMeter, accuracy

import max_former
import spikformer as _spf

from plif_wrapper import count_extra_params, collect_betas

try:
    from loader import create_loader
except ImportError:
    from timm.data import create_loader

_logger = logging.getLogger('train')

_RUN_DIR_MAP = {
    'max_former': 'p1_maxformer_learnable_b0',
    'spikformer': 'p3_spikformer_learnable_b0',
}

def get_run_dir(results_root: str, model_name: str, seed: int) -> Path:
    subdir = _RUN_DIR_MAP.get(model_name, f'{model_name}_learnable_b{seed}')
    return Path(results_root) / subdir

def parse_args():
    p = _ap.ArgumentParser(description='P1/P3: learnable-beta LIF training')
    p.add_argument('--config', type=str, default='')
    p.add_argument('--init-tau', type=float, default=2.0)
    p.add_argument('--model', type=str, default='max_former')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-path', type=str, default='')
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--checkpoint-interval', type=int, default=25)
    p.add_argument('--beta-log-interval', type=int, default=5)

    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--weight-decay', type=float, default=0.06)
    p.add_argument('--warmup-epochs', type=int, default=20)
    p.add_argument('--min-lr', type=float, default=1e-5)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.75)
    p.add_argument('--cutmix', type=float, default=0.5)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=False)

    p.add_argument('--num-classes', type=int, default=100)
    p.add_argument('--time-step', type=int, default=4, dest='T')
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--layer', type=int, default=4)
    p.add_argument('--mlp-ratio', type=int, default=4)

    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            attr = k.replace('-', '_').replace(' ', '_')
            if hasattr(args, attr):
                setattr(args, attr, v)
        if 'amp' in cfg:
            args.amp = bool(cfg['amp'])
        if 'time_step' in cfg:
            args.T = int(cfg['time_step'])
        if 'mlp_ratio' in cfg:
            args.mlp_ratio = int(cfg['mlp_ratio'])

    return args

def build_loaders(args):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    print('[data] creating train dataset...', flush=True)
    train_ds = create_dataset('torch/cifar100', root=args.data_path,
                               split='train', is_training=True, download=True)
    print('[data] creating val dataset...', flush=True)
    val_ds = create_dataset('torch/cifar100', root=args.data_path,
                               split='validation', is_training=False, download=True)
    print('[data] datasets ready', flush=True)

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=None, prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=args.smoothing, num_classes=args.num_classes,
        )

    try:
        print(f'[data] building train loader (workers={args.workers})...', flush=True)
        train_loader = create_loader(
            train_ds, input_size=(3, 32, 32), batch_size=args.batch_size,
            is_training=True, use_prefetcher=True, mean=mean, std=std,
            num_workers=args.workers, distributed=False, crop_pct=1.0,
            interpolation='bicubic', pin_memory=True,
        )
        print('[data] train loader ready', flush=True)
        print(f'[data] building val loader (workers={args.workers})...', flush=True)
        val_loader = create_loader(
            val_ds, input_size=(3, 32, 32), batch_size=args.batch_size,
            is_training=False, use_prefetcher=True, mean=mean, std=std,
            num_workers=args.workers, distributed=False, pin_memory=True,
        )
        print('[data] val loader ready', flush=True)
    except Exception as exc:
        print(f'[data] create_loader failed ({exc}), falling back to plain DataLoader',
              flush=True)
        import torchvision.transforms as T
        t_tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
        t_vl = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(args.data_path, train=True,
                                          transform=t_tr, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(args.data_path, train=False,
                                          transform=t_vl, download=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        mixup_fn = None

    return train_loader, val_loader, mixup_fn

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, mixup_fn=None):
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
        n = inputs.size(0)
        loss_m.update(loss.item(), n)
        if mixup_fn is None:
            acc1, = accuracy(outputs, targets, topk=(1,))
            top1_m.update(acc1.item(), n)

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
        n = inputs.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(acc1.item(), n)
        top5_m.update(acc5.item(), n)

    return loss_m.avg, top1_m.avg, top5_m.avg

def find_latest_checkpoint(run_dir: Path):
    """Return (path, epoch) of the latest epoch checkpoint, or (None, -1)."""
    ckpts = sorted(run_dir.glob('checkpoint_epoch_*.pt'))
    if not ckpts:
        return None, -1
    latest = ckpts[-1]
    try:
        epoch = int(latest.stem.split('_')[-1])
    except ValueError:
        return None, -1
    return latest, epoch

def save_checkpoint(run_dir: Path, epoch: int, model, optimizer, scheduler,
                    scaler, best_val_top1: float, best_epoch: int,
                    best_betas: list, init_betas: list):
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'best_val_top1': best_val_top1,
        'best_epoch': best_epoch,
        'best_betas': best_betas,
        'init_betas': init_betas,
    }
    path = run_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(ckpt, path)
    _logger.info(f'[ckpt] saved {path.name}')
    old_ckpts = sorted(run_dir.glob('checkpoint_epoch_*.pt'))[:-2]
    for old in old_ckpts:
        old.unlink(missing_ok=True)

def main():
    args = parse_args()

    run_dir = get_run_dir(args.results_dir, args.model, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'train.log'),
            logging.StreamHandler(),
        ],
    )
    _logger.info(f'=== Notes-v2 P1/P3: Learnable-beta {args.model} ===')
    _logger.info(f'model={args.model}  init_tau={args.init_tau}  seed={args.seed}  '
                 f'epochs={args.epochs}  run_dir={run_dir}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

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
            sj_ver = 'unknown'

    try:
        git_hash = subprocess.check_output(
            ['git', '-C', str(_MAXFORMER_DIR.parent.parent), 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

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
    n_extra_params = count_extra_params(model)
    default_beta = 1.0 - 1.0 / args.init_tau
    print(f'[init] params={n_params:,}  plif_layers={n_extra_params}  '
          f'init_tau={args.init_tau:.4f}  default_beta={default_beta:.4f}', flush=True)
    _logger.info(f'Total params: {n_params:,}  |  Extra PLIF scalars: {n_extra_params}')

    init_betas = collect_betas(model)
    beta_strs = [str(round(d["beta"], 4)) for d in init_betas]
    _logger.info(f'Initial betas: {beta_strs}')

    print('[init] building data loaders...', flush=True)
    train_loader, val_loader, mixup_fn = build_loaders(args)

    train_loss_fn = (SoftTargetCrossEntropy() if mixup_fn is not None
                     else LabelSmoothingCrossEntropy(smoothing=args.smoothing))
    val_loss_fn = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model, opt='adamw', lr=args.lr, weight_decay=args.weight_decay)

    class _SchedArgs:
        sched = 'cosine'
        epochs = args.epochs
        min_lr = args.min_lr
        warmup_lr = 1e-5
        warmup_epochs = args.warmup_epochs
        cooldown_epochs = 10
        decay_rate = 0.1
        lr_noise = None
        lr_noise_pct = 0.67
        lr_noise_std = 1.0
        seed = args.seed

    scheduler, _ = create_scheduler(_SchedArgs(), optimizer)
    scaler = GradScaler() if args.amp else None

    resume_path, resume_epoch = find_latest_checkpoint(run_dir)
    start_epoch = 0
    best_val_top1 = 0.0
    best_val_top5 = 0.0
    best_epoch = 0
    best_betas = None

    if resume_path is not None:
        print(f'[resume] loading {resume_path.name}', flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_top1 = ckpt.get('best_val_top1', 0.0)
        best_val_top5 = ckpt.get('best_val_top5', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        best_betas = ckpt.get('best_betas', None)
        init_betas = ckpt.get('init_betas', init_betas)
        _logger.info(f'Resumed from epoch {resume_epoch}, continuing at epoch {start_epoch}')
    else:
        print('[init] no checkpoint found — starting from scratch', flush=True)

    btraj_path = run_dir / 'beta_trajectory.csv'
    btraj_mode = 'a' if resume_path is not None else 'w'
    btraj_file = open(btraj_path, btraj_mode, newline='')
    btraj_writer = csv.writer(btraj_file)
    if btraj_mode == 'w':
        btraj_writer.writerow(['epoch', 'layer_index', 'layer_name', 'beta', 'tau'])

    def _log_betas(epoch):
        for d in collect_betas(model):
            btraj_writer.writerow([epoch, d['layer_index'], d['layer_name'],
                                    f"{d['beta']:.6f}", f"{d['tau']:.6f}"])
        btraj_file.flush()

    epoch_log_path = run_dir / 'epoch_log.csv'
    elog_mode = 'a' if resume_path is not None else 'w'
    elog_file = open(epoch_log_path, elog_mode, newline='')
    elog_writer = csv.writer(elog_file)
    if elog_mode == 'w':
        elog_writer.writerow(['epoch', 'train_loss', 'train_top1',
                               'val_loss', 'val_top1', 'val_top5'])

    if start_epoch == 0:
        _log_betas(0)

    best_ckpt_path = run_dir / 'best.pt'
    t0 = time.time()

    print(f'[train] starting from epoch {start_epoch}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        print(f'[train] epoch {epoch}/{args.epochs-1} — training...', flush=True)

        tr_loss, tr_top1 = train_one_epoch(
            model, train_loader, optimizer, train_loss_fn,
            device, scaler=scaler, mixup_fn=mixup_fn)
        vl_loss, vl_top1, vl_top5 = validate(model, val_loader, val_loss_fn, device)

        _logger.info(
            f'Epoch {epoch:3d}/{args.epochs-1}  '
            f'train_loss={tr_loss:.4f}  train_top1={tr_top1:.2f}  '
            f'val_top1={vl_top1:.2f}%  val_top5={vl_top5:.2f}%'
        )
        elog_writer.writerow([epoch, f'{tr_loss:.6f}', f'{tr_top1:.4f}',
                               f'{vl_loss:.6f}', f'{vl_top1:.4f}', f'{vl_top5:.4f}'])
        elog_file.flush()

        if (epoch + 1) % args.beta_log_interval == 0:
            _log_betas(epoch + 1)

        if vl_top1 > best_val_top1:
            best_val_top1 = vl_top1
            best_val_top5 = vl_top5
            best_epoch = epoch
            best_betas = collect_betas(model)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_top1': vl_top1,
                'val_top5': vl_top5,
                'layer_betas': best_betas,
            }, best_ckpt_path)
            _logger.info(f'  -> New best: {best_val_top1:.2f}% top1 / '
                         f'{best_val_top5:.2f}% top5 (epoch {best_epoch})')

        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(run_dir, epoch, model, optimizer, scheduler,
                            scaler, best_val_top1, best_epoch, best_betas, init_betas)

    elog_file.close()
    btraj_file.close()
    elapsed = time.time() - t0

    print('[eval] final validation...', flush=True)
    _, final_top1, final_top5 = validate(model, val_loader, val_loss_fn, device)
    final_betas = collect_betas(model)

    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'val_top1': final_top1,
        'val_top5': final_top5,
        'layer_betas': final_betas,
    }, run_dir / 'final.pt')

    default_beta = 1.0 - 1.0 / args.init_tau
    init_map = {d['layer_index']: d['beta'] for d in init_betas}
    best_map = {d['layer_index']: d['beta'] for d in (best_betas or [])}
    with open(run_dir / 'learned_betas_final.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer_index', 'layer_name', 'initial_beta',
                    'final_beta', 'best_epoch_beta', 'delta_from_default'])
        for d in final_betas:
            idx = d['layer_index']
            init_b = init_map.get(idx, default_beta)
            best_b = best_map.get(idx, d['beta'])
            delta = d['beta'] - default_beta
            w.writerow([idx, d['layer_name'], f'{init_b:.6f}',
                        f"{d['beta']:.6f}", f'{best_b:.6f}', f'{delta:.6f}'])

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                   if torch.cuda.is_available() else 0.0)

    run_json = {
        'model_name': args.model,
        'seed': args.seed,
        'epochs': args.epochs,
        'default_beta_used': default_beta,
        'init_tau': args.init_tau,
        'best_val_top1': best_val_top1,
        'best_val_top5': best_val_top5,
        'final_val_top1': final_top1,
        'final_val_top5': final_top5,
        'epoch_of_best': best_epoch,
        'training_time_seconds': elapsed,
        'peak_gpu_memory_mb': peak_mem_mb,
        'total_parameters': n_params,
        'n_extra_learnable_b_params': n_extra_params,
        'system': {
            'gpu': gpu_name, 'cuda': torch.version.cuda,
            'pytorch': torch.__version__, 'spikingjelly': sj_ver,
            'git_hash': git_hash,
        },
    }
    with open(run_dir / 'run.json', 'w') as f:
        json.dump(run_json, f, indent=2)

    _logger.info(
        f'DONE  best={best_val_top1:.2f}% top1 @ epoch {best_epoch}  '
        f'final={final_top1:.2f}% top1  time={elapsed/3600:.2f}h'
    )
    print(f'[done] best={best_val_top1:.2f}%  final={final_top1:.2f}%  '
          f'time={elapsed/3600:.2f}h  outputs={run_dir}', flush=True)

if __name__ == '__main__':
    main()
