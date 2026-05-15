"""
aggregate.py — Collect per-run JSONs from Exp A and B into summary CSVs
=========================================================================
Run after all training jobs complete.

Usage:
    python aggregate.py --expA-dir results/expA --expB-dir results/expB

Produces:
  results/expA/summary.csv      — one row per (model, beta, seed)
  results/expA/aggregated.csv   — mean+std per (model, beta)
  results/expB/summary.csv      — one row per (model, seed)
  results/expB/learned_betas.csv   — final beta per layer per run
  results/expB/beta_trajectories.csv — combined beta trajectory across all runs
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--expA-dir', type=str, default='results/expA')
    p.add_argument('--expB-dir', type=str, default='results/expB')
    return p.parse_args()

def aggregate_expA(expA_dir: Path):
    expA_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for json_path in sorted(expA_dir.rglob('run_record.json')):
        try:
            with open(json_path) as f:
                d = json.load(f)
            records.append({
                'model': d['model_name'],
                'beta': d['beta'],
                'tau': d['tau'],
                'seed': d['seed'],
                'best_val_acc': d['best_val_acc'],
                'final_val_acc': d['final_val_acc'],
                'epoch_of_best': d['epoch_of_best'],
                'train_seconds': d.get('training_time_seconds', float('nan')),
                'n_params': d.get('total_parameters', -1),
            })
        except Exception as e:
            print(f'WARN: could not parse {json_path}: {e}', file=sys.stderr)

    if not records:
        print('No run_record.json files found under', expA_dir, file=sys.stderr)
        return

    summary_path = expA_dir / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'model', 'beta', 'tau', 'seed',
            'best_val_acc', 'final_val_acc', 'epoch_of_best',
            'train_seconds', 'n_params',
        ])
        w.writeheader()
        for r in sorted(records, key=lambda x: (x['model'], x['beta'], x['seed'])):
            w.writerow(r)
    print(f'Written: {summary_path}  ({len(records)} runs)')

    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r['model'], r['beta'], r['tau'])].append(r['best_val_acc'])

    agg_path = expA_dir / 'aggregated.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'beta', 'tau', 'mean_best_acc', 'std_best_acc', 'n_seeds'])
        for (model, beta, tau), accs in sorted(groups.items()):
            arr = np.array(accs)
            w.writerow([
                model, f'{beta:.4f}', f'{tau:.4f}',
                f'{arr.mean():.4f}', f'{arr.std():.4f}', len(arr),
            ])
    print(f'Written: {agg_path}  ({len(groups)} (model,beta) groups)')

    expected_models = ['spikformer', 'max_former']
    expected_betas = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.875, 0.95, 0.99]
    expected_seeds = [0, 1, 2]

    run_set = {(r['model'], round(r['beta'], 4), r['seed']) for r in records}
    missing = []
    for m in expected_models:
        for b in expected_betas:
            for s in expected_seeds:
                if (m, round(b, 4), s) not in run_set:
                    missing.append((m, b, s))
    if missing:
        print(f'\nMISSING RUNS ({len(missing)}):')
        for m, b, s in missing:
            print(f'  {m}  beta={b}  seed={s}')
    else:
        print('All 60 expected runs present.')

def aggregate_expB(expB_dir: Path):
    expB_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for json_path in sorted(expB_dir.rglob('run_record.json')):
        try:
            with open(json_path) as f:
                d = json.load(f)
            records.append(d)
        except Exception as e:
            print(f'WARN: could not parse {json_path}: {e}', file=sys.stderr)

    if not records:
        print('No run_record.json files found under', expB_dir, file=sys.stderr)
        return

    summary_path = expB_dir / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'best_val_acc', 'final_val_acc',
                    'epoch_of_best', 'n_extra_plif_params'])
        for d in sorted(records, key=lambda x: (x['model_name'], x['seed'])):
            w.writerow([
                d['model_name'], d['seed'],
                f"{d['best_val_acc']:.4f}", f"{d['final_val_acc']:.4f}",
                d['epoch_of_best'], d.get('n_extra_plif_params', '?'),
            ])
    print(f'Written: {summary_path}  ({len(records)} runs)')

    lb_path = expB_dir / 'learned_betas.csv'
    with open(lb_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'layer_index', 'layer_name',
                    'final_beta', 'final_tau', 'best_epoch_beta'])
        for d in sorted(records, key=lambda x: (x['model_name'], x['seed'])):
            final_betas = d.get('final_betas', [])
            best_betas = d.get('best_epoch_betas', [])
            for i, fb in enumerate(final_betas):
                bb_val = best_betas[i]['beta'] if i < len(best_betas) else float('nan')
                w.writerow([
                    d['model_name'], d['seed'],
                    fb['layer_index'], fb['layer_name'],
                    f"{fb['beta']:.6f}", f"{fb['tau']:.6f}",
                    f'{bb_val:.6f}',
                ])
    print(f'Written: {lb_path}')

    bt_path = expB_dir / 'beta_trajectories.csv'
    with open(bt_path, 'w', newline='') as out_f:
        w = csv.writer(out_f)
        w.writerow(['model', 'seed', 'epoch', 'layer_index', 'layer_name',
                    'beta', 'tau'])
        for json_path in sorted(expB_dir.rglob('run_record.json')):
            run_dir = json_path.parent
            traj_path = run_dir / 'beta_trajectory.csv'
            model_seed = f'{run_dir.parent.name}/{run_dir.name}'
            if traj_path.exists():
                with open(traj_path) as tf:
                    reader = csv.DictReader(tf)
                    for row in reader:
                        model_name = run_dir.parent.name
                        seed_str = run_dir.name.replace('seed_', '')
                        w.writerow([model_name, seed_str,
                                    row['epoch'], row['layer_index'],
                                    row['layer_name'], row['beta'], row['tau']])
            else:
                print(f'WARN: no beta_trajectory.csv for {model_seed}',
                      file=sys.stderr)
    print(f'Written: {bt_path}')

def main():
    args = parse_args()
    print('=== Aggregating Experiment A ===')
    aggregate_expA(Path(args.expA_dir))
    print('\n=== Aggregating Experiment B ===')
    aggregate_expB(Path(args.expB_dir))
    print('\nDone.')

if __name__ == '__main__':
    main()
