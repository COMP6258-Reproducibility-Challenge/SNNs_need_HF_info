"""
compare_operators.py  —  Experiment 2.2: Compare all token-mixer operators
===========================================================================
Parses training logs from Exp 1.1 (avg, max) and Exp 2.2 (min, lap, learned_hp),
extracts the best top-1 accuracy for each, and produces:

  comparison_table.txt   — formatted text table for the report
  operator_accuracy.png  — bar chart of best accuracy per operator
  learning_curves.png    — val accuracy vs epoch for all 5 runs

The table and bar chart directly address reviewer weakness W1:
  "Theory is shallow — does high-pass filtering consistently improve accuracy?"

If the results show:
  Avg < Min, Lap, LearnedHP ≈ Max  → any HP operator works; Fourier theory holds
  Avg < Max >> Lap, LearnedHP      → Max-Pool is special, not just HP filtering
  Avg ≈ Min < Lap < LearnedHP < Max → increasing degree of learned HP matters

Usage:
  python compare_operators.py \\
    --exp11-avg-log  /scratch/$USER/maxformer_repro/AvgVsMax/avg/train.log \\
    --exp11-max-log  /scratch/$USER/maxformer_repro/AvgVsMax/max/train.log \\
    --min-log        /scratch/$USER/maxformer_repro/Operators_min/train.log \\
    --lap-log        /scratch/$USER/maxformer_repro/Operators_lap/train.log \\
    --learned-hp-log /scratch/$USER/maxformer_repro/Operators_learned_hp/train.log \\
    --output         ./comparison_results
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
OPERATORS = [
    ('AvgFormer',       'exp11_avg_log',   'royalblue',      'Avg-Pool (baseline)\n~76.73% expected'),
    ('MaxFormer-lite',  'exp11_max_log',   'darkorange',     'Max-Pool (paper)\n~79.12% expected'),
    ('MinFormer',       'min_log',         'mediumpurple',   'Min-Pool (Exp 2.2)'),
    ('LaplacianFormer', 'lap_log',         'crimson',        'Fixed Laplacian (Exp 2.2)'),
    ('LearnedHPFormer', 'learned_hp_log',  'mediumseagreen', 'Learned HP conv (Exp 2.2)'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp11-avg-log',  type=Path, default=None)
    p.add_argument('--exp11-max-log',  type=Path, default=None)
    p.add_argument('--min-log',        type=Path, default=None)
    p.add_argument('--lap-log',        type=Path, default=None)
    p.add_argument('--learned-hp-log', type=Path, default=None)
    p.add_argument('--output',         type=Path, default=Path('./comparison_results'))
    p.add_argument('--dpi',            type=int,  default=200)
    return p.parse_args()


# ── Log parsing ───────────────────────────────────────────────────────────────
def parse_log(log_path: Path) -> dict:
    """
    Extract best top-1 accuracy and per-epoch validation accuracy curve.
    Handles two formats from timm's train.py:
      "*** Best metric: 79.12 (epoch 387)"
      "Test: [...] Acc@1:  79.1200 ..."
    """
    if log_path is None or not log_path.exists():
        return {'best': None, 'curve': [], 'epochs': []}

    text = log_path.read_text(errors='replace')

    # Best metric line
    best = None
    m = re.search(r'\*\*\* Best metric:\s*([\d.]+)', text)
    if m:
        best = float(m.group(1))

    # Per-epoch validation curve — "Test: [epoch/total] ... Acc@1: XX.XXXX (XX.XXXX)"
    # timm logs per-batch; we grab last Test line per epoch by looking for the end-of-epoch summary
    # Format: "Test: [  391/391]  ...  Acc@1:  79.1200 ( 79.1200)"
    epoch_accs = []
    epochs_seen = []
    # Also look for "Train: X " at start to track epoch
    train_ep_pattern = re.compile(r'^Train: (\d+) \[', re.MULTILINE)
    test_summary_pattern = re.compile(
        r'Test.*?Acc@1:\s*([\d.]+)\s*\(\s*([\d.]+)\s*\)', re.MULTILINE
    )

    # Find all per-epoch summaries (last test line before next "Train:" line)
    train_starts = [m.start() for m in train_ep_pattern.finditer(text)]
    if train_starts:
        boundaries = train_starts + [len(text)]
        for i, start in enumerate(train_starts):
            chunk = text[start:boundaries[i+1]]
            # Find last Test summary in this epoch chunk
            test_matches = list(test_summary_pattern.finditer(chunk))
            if test_matches:
                epoch_acc = float(test_matches[-1].group(2))  # avg, not last batch
                epoch_accs.append(epoch_acc)
                # Extract epoch number
                ep_m = train_ep_pattern.search(chunk)
                epochs_seen.append(int(ep_m.group(1)) if ep_m else i)

    # Fallback: if best not found from summary, take max of curve
    if best is None and epoch_accs:
        best = max(epoch_accs)

    return {'best': best, 'curve': epoch_accs, 'epochs': epochs_seen}


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_bar(results: dict, output_dir: Path, dpi: int):
    names, accs, colors = [], [], []
    for name, arg_key, color, _ in OPERATORS:
        r = results.get(name)
        if r is None or r['best'] is None:
            continue
        names.append(name)
        accs.append(r['best'])
        colors.append(color)

    if not names:
        print("  [SKIP] No valid results for bar chart.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    # Reference lines
    ax.axhline(76.73, color='royalblue', ls='--', lw=1.2, alpha=0.6, label='AvgFormer baseline (76.73%)')
    ax.axhline(79.12, color='darkorange', ls='--', lw=1.2, alpha=0.6, label='MaxFormer target (79.12%)')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(min(accs) - 2.0, max(accs) + 1.5)
    ax.set_ylabel('Best top-1 accuracy (%) on CIFAR-100', fontsize=11)
    ax.set_title('Operator Ablation: Does High-Pass Filtering Consistently Help?\n'
                 '(Exp 2.2 — addresses reviewer weakness W1)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()

    path = output_dir / 'operator_accuracy.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(results: dict, output_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for name, arg_key, color, label in OPERATORS:
        r = results.get(name)
        if r is None or not r['curve']:
            continue
        epochs = r['epochs'] if r['epochs'] else list(range(len(r['curve'])))
        ax.plot(epochs, r['curve'],
                color=color, label=f"{name} (best={r['best']:.2f}%)",
                linewidth=1.5, alpha=0.85)
        any_plotted = True

    if not any_plotted:
        return

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation top-1 accuracy (%)', fontsize=11)
    ax.set_title('Training Curves — All Token-Mixer Operators\n'
                 '(Exp 2.2: do alternative HP operators converge differently?)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = output_dir / 'learning_curves.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def write_table(results: dict, output_dir: Path):
    lines = [
        "=" * 65,
        "  EXPERIMENT 2.2 — TOKEN MIXER OPERATOR COMPARISON",
        "  CIFAR-100 Top-1 Accuracy",
        "=" * 65,
        f"  {'Operator':<22}  {'Model Name':<22}  {'Best Acc':>8}",
        "  " + "-" * 61,
    ]

    ref_avg = None
    ref_max = None
    for name, arg_key, color, desc in OPERATORS:
        r = results.get(name)
        acc_str = f"{r['best']:.2f}%" if (r and r['best'] is not None) else "N/A"
        delta = ""
        if r and r['best'] is not None:
            if name == 'AvgFormer':
                ref_avg = r['best']
                delta = "  (baseline)"
            elif ref_avg is not None:
                d = r['best'] - ref_avg
                delta = f"  ({'+' if d>=0 else ''}{d:.2f}% vs Avg)"
                if name == 'MaxFormer-lite':
                    ref_max = r['best']
        model_key = arg_key.replace('_log', '').replace('exp11_', 'pool_former_')
        lines.append(f"  {name:<22}  {model_key:<22}  {acc_str:>8}{delta}")

    lines += [
        "  " + "-" * 61,
        "",
        "  INTERPRETATION (W1):",
        "  Paper claim: replacing Avg-Pool with Max-Pool works because",
        "  Max-Pool is a high-pass filter, counteracting LIF low-pass effect.",
    ]

    vals = {n: results[n]['best'] for n, *_ in OPERATORS if results.get(n) and results[n]['best']}
    if vals:
        hp_ops = {k: v for k, v in vals.items() if k not in ('AvgFormer',)}
        avg_val = vals.get('AvgFormer')
        max_val = vals.get('MaxFormer-lite')
        all_hp_beat_avg = avg_val and all(v > avg_val for v in hp_ops.values())
        if all_hp_beat_avg:
            lines.append("  → ALL high-pass operators beat AvgFormer: supports HF theory (W1).")
        elif avg_val:
            below = [k for k, v in hp_ops.items() if v <= avg_val]
            lines.append(f"  → {below} did NOT beat AvgFormer: weakens the universal HF claim.")

        if avg_val and max_val:
            lap_val = vals.get('LaplacianFormer')
            if lap_val:
                if abs(lap_val - max_val) < 0.5:
                    lines.append("  → LaplacianFormer ≈ MaxFormer-lite: fixed HP operator is sufficient.")
                elif lap_val > max_val:
                    lines.append("  → LaplacianFormer > MaxFormer-lite: Laplacian is a better HP filter.")
                else:
                    lines.append(f"  → LaplacianFormer ({lap_val:.2f}%) < MaxFormer-lite ({max_val:.2f}%): Max-Pool has advantages beyond HP filtering.")

    lines += ["=" * 65, ""]
    text = '\n'.join(lines)
    path = output_dir / 'comparison_table.txt'
    path.write_text(text)
    print(f"  Saved: {path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    log_map = {
        'AvgFormer':       args.exp11_avg_log,
        'MaxFormer-lite':  args.exp11_max_log,
        'MinFormer':       args.min_log,
        'LaplacianFormer': args.lap_log,
        'LearnedHPFormer': args.learned_hp_log,
    }

    print("[2.2] Parsing training logs...")
    results = {}
    for name, log_path in log_map.items():
        r = parse_log(log_path)
        results[name] = r
        status = f"best={r['best']:.2f}%, {len(r['curve'])} epochs" if r['best'] else "not found"
        print(f"  {name:<22}: {status}")

    print("\n[2.2] Generating outputs...")
    write_table(results, args.output)
    plot_bar(results, args.output, args.dpi)
    plot_curves(results, args.output, args.dpi)

    print(f"\n[2.2] All outputs saved to: {args.output}")
    print("       Use comparison_table.txt and operator_accuracy.png in your report.")


if __name__ == '__main__':
    main()
