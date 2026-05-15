# Learnable Beta

Extension of the MaxFormer reproducibility study exploring **learnable per-layer LIF decay constants (β)**
for the DPDL course at the University of Southampton.

## What's in here

```
configs/            YAML training configs (model, data, optimiser)
results/            Outputs from each run (checkpoints, logs, CSVs)
slurm/              SLURM job scripts for Iridis
train_plif.py       Training with learnable β (one scalar per LIF layer)
train_beta.py       Training with fixed global β (sweep baseline)
plif_wrapper.py     ParametricMultiStepLIFNode implementation
analyze_spectra.py  Spectral analysis of trained models
aggregate.py        Results aggregation
```

### Experiments

| Folder | What it tests |
|---|---|
| `p1_maxformer_learnable_b0` | Max-Former with learnable β, 150 epochs |
| `p2_spectra` | Spectral analysis of P1 |
| `p3_spikformer_learnable_b0` | Spikformer with learnable β, 150 epochs |
| `p4_spectra` | Spectral analysis of P3 |

## Running on Iridis

```bash
sbatch slurm/p1_maxformer.sh
sbatch slurm/p3_spikformer.sh
```

## Requirements

The conda environment is `maxformer` on Iridis. To set it up locally:

```bash
conda create -n maxformer python=3.10
pip install torch torchvision timm spikingjelly
```
