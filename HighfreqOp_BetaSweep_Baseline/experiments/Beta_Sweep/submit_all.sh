#!/bin/bash
# Submit all 4 β-sweep jobs.
# Only 2 nodes available concurrently — SLURM will queue β=0.75 and β=0.90
# and start them automatically when slots open.
#
# Run from experiments/Beta_Sweep/:
#   bash submit_all.sh

set -euo pipefail
mkdir -p logs

echo "[3.2] Submitting all 4 beta-sweep jobs..."

JOB_B02=$(sbatch  --parsable slurm_beta_02.sh)
JOB_B05=$(sbatch  --parsable slurm_beta_05.sh)
JOB_B075=$(sbatch --parsable slurm_beta_075.sh)
JOB_B09=$(sbatch  --parsable slurm_beta_09.sh)

echo ""
echo "Submitted:"
echo "  β=0.20 (τ=1.25) : job ${JOB_B02}"
echo "  β=0.50 (τ=2.00) : job ${JOB_B05}   [paper default]"
echo "  β=0.75 (τ=4.00) : job ${JOB_B075}"
echo "  β=0.90 (τ=10.0) : job ${JOB_B09}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "When all complete, run:"
echo "  python plot_beta_sweep.py \\"
echo "      --base-dir /scratch/\${USER}/maxformer_repro \\"
echo "      --output   \${SLURM_SUBMIT_DIR}/figures_beta"
