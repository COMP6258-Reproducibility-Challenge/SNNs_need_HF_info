#!/bin/bash
#SBATCH --job-name=Operators_min
#SBATCH --nodes=1
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --output=logs/Operators_min_%j.out
#SBATCH --error=logs/Operators_min_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kb6g21@soton.ac.uk

# =============================================================================
#  MinPool token mixer on CIFAR-100
# Expected accuracy: between 76.73% (AvgFormer) and 79.12% (MaxFormer-lite)
# Run from experiments/Operators/:
#   mkdir -p logs && sbatch slurm_min.sh
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "MinPool token mixer"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time   : $(date)"
echo "========================================================"

REPO_DIR="${HOME}/Repo/experiments/MaxFormer"
DATA_DIR="/scratch/${USER}/maxformer_repro/data"
EXP_DIR="/scratch/${USER}/maxformer_repro/Operators_min"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CIFAR_DIR="${REPO_DIR}/cifar10-100"

rm -rf "${EXP_DIR}"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

module purge
module load conda/python3

for cuda_mod in cuda/12.4.0 cuda/12.2.0 cuda/11.8.0 cuda/12.5.1 cuda/12.6.3; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[2.2-min] Loaded: $cuda_mod"; break
    fi
done

export PATH="/home/kb6g21/.conda/envs/maxformer/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ── Inject scripts ────────────────────────────────────────────────────────────
cp "${SCRIPT_DIR}/alt_operator_former.py" "${CIFAR_DIR}/"
cp "${SCRIPT_DIR}/cifar100_min.yaml"      "${CIFAR_DIR}/"

# Also inject pooling_former.py so train.py patch works uniformly
cp "${SCRIPT_DIR}/../AvgVsMax/pooling_former.py" "${CIFAR_DIR}/" 2>/dev/null || true

# ── Patch train.py ────────────────────────────────────────────────────────────
python - <<PYEOF
import os
train_path = os.path.join('${CIFAR_DIR}', 'train.py')
with open(train_path) as f:
    content = f.read()
needs_patch = 'import alt_operator_former' not in content
if needs_patch:
    content = content.replace(
        'from max_resnet import max_resnet18',
        'from max_resnet import max_resnet18\nimport alt_operator_former  # Exp 2.2'
    )
    with open(train_path, 'w') as f:
        f.write(content)
    print('  train.py patched for alt_operator_former.')
else:
    print('  train.py already patched.')
PYEOF

# ── Sanity check ──────────────────────────────────────────────────────────────
cd "${CIFAR_DIR}"
python alt_operator_former.py

# ── Training ──────────────────────────────────────────────────────────────────
echo ""
echo "[2.2-min] Training MinPool former (expected ~77–79%)..."
echo "          Start: $(date)"

python train.py \
    --experiment "Operators_min" \
    --config     "${CIFAR_DIR}/cifar100_min.yaml" \
    --data-path  "${DATA_DIR}" \
    --output     "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "[2.2-min] Training done: $(date)"

# Copy log back
cp "${EXP_DIR}/train.log" "${SCRIPT_DIR}/log_min_${SLURM_JOB_ID}.txt"

echo ""
echo "========================================================"
echo "  Exp 2.2 MinPool COMPLETE — $(date)"
echo "  Checkpoint: ${EXP_DIR}/Operators_min/best.pth.tar"
echo "========================================================"
