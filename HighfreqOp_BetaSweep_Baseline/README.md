# MaxFormer Reproducibility

Reproducibility study of **"Spiking Neural Networks Need High-Frequency Information"**
(Fang et al., NeurIPS 2025 — arXiv:2505.18608) for the DPDL course at the University of Southampton.

## What's in here

experiments/        SLURM scripts and code for each experiment
iridis_results/     Results downloaded from Iridis (CSVs, logs, checkpoints)
report_figures/     Static copies of the final report figures
report_figures.ipynb  Notebook that reads results and generates all figures
MaxFormer/          Original authors' codebase (cloned from their repo)

### Experiments

| Folder | What it tests |
|---|---|
| `AvgVsMax` | Avg-pool vs Max-pool token mixer (main claim) |
| `FullMax` | Full MaxFormer on CIFAR-10 and CIFAR-100 |
| `Operators` | Alternative HF operators — MinPool, Laplacian, Learned HP |
| `Beta_Sweep` | LIF decay constant β sensitivity (β = 0.2, 0.5, 0.75, 0.9) |

## Running experiments on Iridis

Each experiment folder has one or more SLURM scripts. From the cluster:

```bash
cd ~/Repo/experiments/AvgVsMax
mkdir -p logs
sbatch slurm_avg.sh
sbatch slurm_max.sh
```

Results land in `/scratch/$USER/maxformer_repro/` and get copied back to `iridis_results/` via scp.

## Requirements

The conda environment is `maxformer` on Iridis. To set it up locally:

```bash
conda create -n maxformer python=3.10
pip install torch torchvision timm spikingjelly
```
