"""
plot_beta_sweep.py  —  Experiment 3.2: LIF τ sweep analysis (W1)
=================================================================
Reads summary.csv files from all 4 β runs, produces accuracy-vs-β plots
and a spectral analysis summary.

Usage:
  python plot_beta_sweep.py \\
      --base-dir /scratch/$USER/maxformer_repro \\
      --output   /path/to/Beta_Sweep/figures

Expected directory structure (on Iridis scratch):
  base-dir/
    Beta_Sweep/b02/Beta_Sweep_b02/summary.csv    (τ=1.25, β≈0.20)
    Beta_Sweep/b05/Beta_Sweep_b05/summary.csv    (τ=2.00, β=0.50)
    Beta_Sweep/b075/Beta_Sweep_b075/summary.csv  (τ=4.00, β=0.75)
    Beta_Sweep/b09/Beta_Sweep_b09/summary.csv    (τ=10.0, β=0.90)

Or pass individual CSV paths:
  python plot_beta_sweep.py \\
      --csv-b02  .../summary.csv \\
      --csv-b05  .../summary.csv \\
      --csv-b075 .../summary.csv \\
      --csv-b09  .../summary.csv \\
      --output   ./figures
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Beta / tau configuration ──────────────────────────────────────────────────

BETA_CONFIGS = [
    # (tag,   tau,   beta,  color)
    ('02',   1.25,  0.200, '#2166AC'),
    ('05',   2.00,  0.500, '#4DAC26'),
    ('075',  4.00,  0.750, '#F4A582'),
    ('09',  10.00,  0.900, '#D6604D'),
]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_summary(path: str):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def best_top1(rows):
    vals = []
    for r in rows:
        try:
            vals.append(float(r['eval_top1']))
        except (KeyError, ValueError):
            pass
    return max(vals) if vals else float('nan')


def top1_curve(rows):
    vals = []
    for r in rows:
        try:
            vals.append(float(r['eval_top1']))
        except (KeyError, ValueError):
            vals.append(float('nan'))
    return np.array(vals)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='',
                    help='Base scratch directory containing the Beta_Sweep/ folder')
    ap.add_argument('--csv-b02',  default='')
    ap.add_argument('--csv-b05',  default='')
    ap.add_argument('--csv-b075', default='')
    ap.add_argument('--csv-b09',  default='')
    ap.add_argument('--output',   default='figures')
    ap.add_argument('--dpi',      type=int, default=200)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Resolve CSV paths ─────────────────────────────────────────────────────
    csv_paths = {
        '02':  args.csv_b02,
        '05':  args.csv_b05,
        '075': args.csv_b075,
        '09':  args.csv_b09,
    }

    if args.base_dir:
        base = Path(args.base_dir)
        fallbacks = {
            '02':  base / 'Beta_Sweep' / 'b02'  / 'Beta_Sweep_b02'  / 'summary.csv',
            '05':  base / 'Beta_Sweep' / 'b05'  / 'Beta_Sweep_b05'  / 'summary.csv',
            '075': base / 'Beta_Sweep' / 'b075' / 'Beta_Sweep_b075' / 'summary.csv',
            '09':  base / 'Beta_Sweep' / 'b09'  / 'Beta_Sweep_b09'  / 'summary.csv',
        }
        for tag, fb in fallbacks.items():
            if not csv_paths[tag]:
                csv_paths[tag] = str(fb)

    # ── Load data ─────────────────────────────────────────────────────────────
    loaded = {}
    for tag, tau, beta, color in BETA_CONFIGS:
        if not csv_paths.get(tag):
            continue
        rows = read_summary(csv_paths[tag])
        if rows:
            loaded[tag] = rows
            print(f"  β={beta:.2f} (τ={tau:.2f}): {len(rows)} epochs, best={best_top1(rows):.2f}%")
        else:
            print(f"  β={beta:.2f} (τ={tau:.2f}): NOT FOUND — {csv_paths[tag]}")

    if len(loaded) < 2:
        print("ERROR: Need at least 2 β values to plot. Check paths.", file=sys.stderr)
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────────
    ref_best = best_top1(loaded.get('05', []))  # β=0.5 is paper default

    print("\n" + "=" * 62)
    print("  EXP 3.2 — LIF τ SWEEP RESULTS (W1)")
    print("=" * 62)
    print(f"  {'β':>6}  {'τ':>6}  {'Best top-1':>12}  {'Δ vs β=0.5':>12}  {'LP/HP'}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*6}")

    results = []
    for tag, tau, beta, color in BETA_CONFIGS:
        if tag not in loaded:
            continue
        best = best_top1(loaded[tag])
        delta = best - ref_best if not np.isnan(ref_best) else float('nan')
        lp_hp = 'HIGH-PASS' if beta < 0.5 else ('reference' if beta == 0.5 else 'low-pass')
        delta_str = f"{delta:+.2f}%" if not np.isnan(delta) else "  —"
        print(f"  {beta:>6.2f}  {tau:>6.2f}  {best:>11.2f}%  {delta_str:>12}  {lp_hp}")
        results.append((tag, tau, beta, best, delta, color))

    # W1 verdict
    print()
    betas_present  = [r[2] for r in results if not np.isnan(r[3])]
    accs_present   = [r[3] for r in results if not np.isnan(r[3])]
    if len(betas_present) >= 3:
        corr = np.corrcoef(betas_present, accs_present)[0, 1]
        print(f"  Pearson correlation (β, accuracy) = {corr:.3f}")
        if corr < -0.7:
            print("  W1 VERDICT: Strong negative correlation → LIF low-pass IS the bottleneck ✓")
        elif corr < -0.3:
            print("  W1 VERDICT: Moderate negative correlation → partial support for LIF theory")
        else:
            print("  W1 VERDICT: Weak/no correlation → LIF τ may not be the primary factor")
    print("=" * 62)

    # Save text summary
    with open(out / 'beta_sweep_summary.txt', 'w') as f:
        f.write("EXP 3.2 — LIF τ SWEEP RESULTS (W1)\n")
        f.write("=" * 62 + "\n")
        f.write(f"{'β':>6}  {'τ':>6}  {'Best top-1':>12}  {'Δ vs β=0.5':>12}\n")
        for tag, tau, beta, best, delta, _ in results:
            delta_str = f"{delta:+.2f}%" if not np.isnan(delta) else "  —"
            f.write(f"{beta:>6.2f}  {tau:>6.2f}  {best:>11.2f}%  {delta_str:>12}\n")
        if len(betas_present) >= 3:
            corr = np.corrcoef(betas_present, accs_present)[0, 1]
            f.write(f"\nPearson corr(β, acc) = {corr:.3f}\n")

    # ── Figure 1: Accuracy vs β ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    b_vals = [r[2] for r in results if not np.isnan(r[3])]
    a_vals = [r[3] for r in results if not np.isnan(r[3])]
    c_vals = [r[5] for r in results if not np.isnan(r[3])]

    ax.plot(b_vals, a_vals, 'k-', linewidth=1.0, alpha=0.4, zorder=1)
    for bv, av, cv in zip(b_vals, a_vals, c_vals):
        ax.scatter(bv, av, color=cv, s=80, zorder=3, edgecolors='black', linewidths=0.7)
        ax.annotate(f'{av:.2f}%', (bv, av), textcoords='offset points',
                    xytext=(5, 4), fontsize=8)

    ax.set_xlabel('LIF decay factor β  (= 1 − 1/τ)', fontsize=11)
    ax.set_ylabel('Best Top-1 Accuracy (%) on CIFAR-100', fontsize=10)
    ax.set_title('Exp 3.2 — LIF time constant τ sweep\n(W1: Does stronger low-pass hurt accuracy?)',
                 fontsize=10)
    ax.set_xlim(-0.05, 1.0)
    ax.grid(alpha=0.3)

    # Shade the "low-pass zone"
    ax.axvspan(0.5, 1.0, alpha=0.06, color='red', label='High τ (low-pass)')
    ax.axvspan(0.0, 0.5, alpha=0.06, color='blue', label='Low τ (high-pass)')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Paper default (β=0.5)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    p = out / 'accuracy_vs_beta.png'
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    print(f"\n  Saved: {p}")

    # ── Figure 2: Learning curves for all β ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, tau, beta, best, delta, color in results:
        if tag not in loaded:
            continue
        curve = top1_curve(loaded[tag])
        epochs = np.arange(1, len(curve) + 1)
        ax.plot(epochs, curve, label=f'β={beta:.2f} (τ={tau:.1f})',
                color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Exp 3.2 — Training curves for all τ values', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out / 'beta_training_curves.png'
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Figure 3: τ vs accuracy (x-axis = τ instead of β) ────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    tau_vals = [r[1] for r in results if not np.isnan(r[3])]
    a_vals2  = [r[3] for r in results if not np.isnan(r[3])]
    c_vals2  = [r[5] for r in results if not np.isnan(r[3])]

    ax.plot(tau_vals, a_vals2, 'k-', linewidth=1.0, alpha=0.4, zorder=1)
    for tv, av, cv in zip(tau_vals, a_vals2, c_vals2):
        ax.scatter(tv, av, color=cv, s=80, zorder=3, edgecolors='black', linewidths=0.7)
        ax.annotate(f'{av:.2f}%', (tv, av), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)

    ax.set_xlabel('LIF membrane time constant τ', fontsize=11)
    ax.set_ylabel('Best Top-1 Accuracy (%)', fontsize=10)
    ax.set_title('Exp 3.2 — Effect of LIF time constant on accuracy\n(larger τ = stronger low-pass filter)',
                 fontsize=10)
    ax.axvline(2.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Paper default (τ=2)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out / 'accuracy_vs_tau.png'
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved: {p}")

    print(f"\n  All figures → {out}/")
    print("  Key figures for report:")
    print("    accuracy_vs_beta.png    — main W1 evidence")
    print("    accuracy_vs_tau.png     — same but τ on x-axis (more intuitive)")


if __name__ == '__main__':
    main()
