#!/bin/bash
#SBATCH --job-name=AvgVsMax_avg
#SBATCH --nodes=1
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=logs/AvgVsMax_avg_%j.out
#SBATCH --error=logs/AvgVsMax_avg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kb6g21@soton.ac.uk

# =============================================================================
#  AvgFormer (Avg-Pool token mixer) on CIFAR-100
#
# Architecture: 3-stage multi-scale (Embed_Orig → Embed_Max → Embed_Max)
#               with Block_Avg throughout — expected top-1 ~76.73%
#
# Run from experiments/AvgVsMax/:
#   mkdir -p logs && sbatch slurm_avg.sh
# Submit alongside slurm_max.sh so both run on separate GPU nodes in parallel.
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "   AvgFormer (Avg-Pool token mixer)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time   : $(date)"
echo "========================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${HOME}/Repo/experiments/MaxFormer"
DATA_DIR="/scratch/${USER}/maxformer_repro/data"
EXP_DIR="/scratch/${USER}/maxformer_repro/AvgVsMax/avg"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CIFAR_DIR="${REPO_DIR}/cifar10-100"

# Clear previous results so summary.csv starts fresh
rm -rf "${EXP_DIR}"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load conda/python3

for cuda_mod in cuda/12.4.0 cuda/12.2.0 cuda/11.8.0 cuda/12.5.1 cuda/12.6.3; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[1.1-avg] Loaded: $cuda_mod"; break
    fi
done

export PATH="/home/kb6g21/.conda/envs/maxformer/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ── Inject updated pooling_former.py ─────────────────────────────────────────
echo "[1.1-avg] Copying pooling_former.py (3-stage multi-scale architecture)..."
cp "${SCRIPT_DIR}/pooling_former.py" "${CIFAR_DIR}/pooling_former.py"
cp "${SCRIPT_DIR}/cifar100_avg.yaml" "${CIFAR_DIR}/"

# ── Patch train.py (idempotent) ───────────────────────────────────────────────
python - <<PYEOF
import os
train_path = os.path.join('${CIFAR_DIR}', 'train.py')
with open(train_path) as f:
    content = f.read()
if 'import pooling_former' not in content:
    content = content.replace(
        'from max_resnet import max_resnet18',
        'from max_resnet import max_resnet18\nimport pooling_former  # Exp 1.1'
    )
    with open(train_path, 'w') as f:
        f.write(content)
    print('  train.py patched.')
else:
    print('  train.py already patched.')
PYEOF

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "[1.1-avg] Sanity check (should print shapes + param count)..."
cd "${CIFAR_DIR}"
python pooling_former.py

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "[1.1-avg] Training AvgFormer (pool_former_avg)..."
echo "          Start: $(date)"

python train.py \
    --experiment "AvgVsMax_avg" \
    --config     "${CIFAR_DIR}/cifar100_avg.yaml" \
    --data-path  "${DATA_DIR}" \
    --output     "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "[1.1-avg] Training done: $(date)"

cp "${EXP_DIR}/train.log" "${SCRIPT_DIR}/log_avg_${SLURM_JOB_ID}.txt"

echo ""
echo "========================================================"
echo "  Exp 1.1 AvgFormer COMPLETE — $(date)"
echo "  Checkpoint : ${EXP_DIR}/AvgVsMax_avg/best.pth.tar"
echo "  Log copy   : ${SCRIPT_DIR}/log_avg_${SLURM_JOB_ID}.txt"
echo "========================================================"
