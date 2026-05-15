# Reproducibility Study: Spiking Neural Networks Need High-Frequency Information

Reproducibility study of **"Spiking Neural Networks Need High-Frequency Information"**
(Fang et al., 2025 — arXiv:2505.18608) for the COMP6258 Differentiable Programming and Deep Learning module.

**Authors:** Anna Obure, Kaylan Bakrania, Nabil Al-Shanteer

**Report:** See `Reproducibility_Challenge (4).pdf`

---

## Overview

This repository contains code and results for three extensions of the original paper:

| Experiment | Folder | What it tests |
|---|---|---|
| Baseline reproduction | `HighfreqOp_BetaSweep_Baseline/` | Avg-pool vs Max-pool, Full MaxFormer on CIFAR-10/100 |
| β sweep + learnable β | `learnable_beta/` | Fixed β sensitivity and per-layer learnable β |
| Alternative HF operators | `HighfreqOp_BetaSweep_Baseline/` | MinPool, Laplacian, LearnedHP vs MaxFormer |
| Temporal analysis (SHD) | `SHD/` | Whether HF claims extend to auditory neuromorphic data |

All experiments were run on the University of Southampton Iridis HPC cluster using SLURM.
The baseline and operator experiments use the authors' released codebase as a starting point.
All extensions (learnable-β, alternative operators, SHD analysis) were implemented by us.

---

## Repository Structure
HighfreqOp_BetaSweep_Baseline/   Baseline reproduction and operator experiments
experiments/                 SLURM scripts and code
iridis_results/              Results (CSVs, logs)
report_figures/              Final figures used in the report
report_figures.ipynb         Notebook to regenerate figures
README.md                    Detailed instructions for these experiments
learnable_beta/                  Learnable β extension
configs/                     YAML training configs
results/                     Outputs from each run
slurm/                       SLURM job scripts
train_plif.py                Training with learnable β
train_beta.py                Training with fixed β (sweep baseline)
plif_wrapper.py              ParametricMultiStepLIFNode implementation
README.md                    Detailed instructions for these experiments
SHD/                             Temporal analysis on Spiking Heidelberg Digits
train.py                     Main training script
models.py                    SNN and ANN model definitions
dataset.py                   SHD data loading
analyse_spectrum.py          Fourier spectrum analysis
aggregate_results.py         Results aggregation
shd.yaml                     Hyperparameters
README.md                    Detailed instructions for these experiments
report/                          Final report PDF

---

## Results Summary

### Baseline Reproduction

| Model | Dataset | Paper (%) | Ours (%) |
|---|---|---|---|
| AvgFormer | CIFAR-100 | 76.73 | 73.82 |
| MaxFormer-lite | CIFAR-100 | 79.12 | 76.38 |
| Full MaxFormer | CIFAR-10 | 97.04 | 96.64 |
| Full MaxFormer | CIFAR-100 | 82.65 | 81.44 |

### β Sweep (MaxFormer, CIFAR-100)

| β | Best Top-1 (%) |
|---|---|
| 0.20 | 78.02 |
| 0.50 | 78.86 |
| 0.75 | 77.72 |
| 0.90 | 73.52 |
| **Learnable (per-layer)** | **79.35** |

### Alternative Operators (CIFAR-100)

| Operator | Best Top-1 (%) | Δ vs AvgFormer |
|---|---|---|
| AvgFormer | 74.59 | — |
| MinPool | 74.94 | +0.35 |
| MaxFormer-lite | 76.38 | +1.79 |
| Laplacian | 77.27 | +2.68 |
| LearnedHP | 77.86 | +3.27 |

### SHD Temporal Analysis (mean over 5 seeds)

| Model | Mean Acc. (%) |
|---|---|
| SNN-Avg | 92.75 |
| SNN-Max | 93.12 |
| ANN-Avg | 71.93 |
| ANN-Max | 71.70 |

---

## Requirements

```bash
conda create -n maxformer python=3.10
pip install torch torchvision timm spikingjelly tonic
```

## Running Experiments

See the README in each subfolder for detailed instructions on running
experiments locally or on a SLURM cluster.
