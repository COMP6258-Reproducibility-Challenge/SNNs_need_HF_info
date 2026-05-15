#!/bin/bash
#SBATCH --job-name=FullMax_c10
#SBATCH --nodes=1
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=logs/FullMax_c10_%j.out
#SBATCH --error=logs/FullMax_c10_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kb6g21@soton.ac.uk

# =============================================================================
# Experiment 1.2 - Full MaxFormer on CIFAR-10
# Expected: 97.04% top-1  (paper Table 2)
# Run from experiments/FullMax/:
#   mkdir -p logs && sbatch slurm_cifar10.sh
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "  Exp 1.2 - Full MaxFormer CIFAR-10"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time   : $(date)"
echo "========================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${HOME}/Repo/experiments/MaxFormer"
DATA_DIR="/scratch/${USER}/maxformer_repro/data"
EXP_DIR="/scratch/${USER}/maxformer_repro/FullMax_c10"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CIFAR_DIR="${REPO_DIR}/cifar10-100"

# Clean up old checkpoints to avoid FileExistsError
rm -rf "${EXP_DIR}"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load conda/python3

for cuda_mod in cuda/12.4.0 cuda/12.2.0 cuda/11.8.0 cuda/12.5.1 cuda/12.6.3; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[1.2] Loaded: $cuda_mod"; break
    fi
done

export PATH="/home/kb6g21/.conda/envs/maxformer/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cp "${SCRIPT_DIR}/cifar10.yaml" "${CIFAR_DIR}/"

# ── Training ──────────────────────────────────────────────────────────────────
echo "[1.2] Training MaxFormer CIFAR-10 (target: 97.04%)..."
echo "      Start: $(date)"

cd "${CIFAR_DIR}"
python train.py \
    --experiment "FullMax_c10" \
    --config "${CIFAR_DIR}/cifar10.yaml" \
    --data-path "${DATA_DIR}" \
    --output "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "[1.2] Training complete: $(date)"

echo ""
echo "========================================================"
echo "  Exp 1.2 CIFAR-10 COMPLETE - $(date)"
echo "  Checkpoint: ${EXP_DIR}/FullMax_c10/best.pth.tar"
echo "  Summary CSV: ${EXP_DIR}/FullMax_c10/summary.csv"
echo "========================================================"
