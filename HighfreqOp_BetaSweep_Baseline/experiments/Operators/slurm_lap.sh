#!/bin/bash
#SBATCH --job-name=Operators_lap
#SBATCH --nodes=1
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --output=logs/Operators_lap_%j.out
#SBATCH --error=logs/Operators_lap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kb6g21@soton.ac.uk

# =============================================================================
# Fixed Laplacian token mixer on CIFAR-100
# Key test: does a fixed (non-trainable) high-pass filter match Max-Pool?
# If yes → supports the Fourier interpretation (W1).
# Run from experiments/Operators/:
#   mkdir -p logs && sbatch slurm_lap.sh
# =============================================================================

set -euo pipefail

echo "========================================================"
echo "  Fixed Laplacian token mixer"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time   : $(date)"
echo "========================================================"

REPO_DIR="${HOME}/Repo/experiments/MaxFormer"
DATA_DIR="/scratch/${USER}/maxformer_repro/data"
EXP_DIR="/scratch/${USER}/maxformer_repro/Operators_lap"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CIFAR_DIR="${REPO_DIR}/cifar10-100"

rm -rf "${EXP_DIR}"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

module purge
module load conda/python3

for cuda_mod in cuda/12.4.0 cuda/12.2.0 cuda/11.8.0 cuda/12.5.1 cuda/12.6.3; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[2.2-lap] Loaded: $cuda_mod"; break
    fi
done

export PATH="/home/kb6g21/.conda/envs/maxformer/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cp "${SCRIPT_DIR}/alt_operator_former.py" "${CIFAR_DIR}/"
cp "${SCRIPT_DIR}/cifar100_lap.yaml"      "${CIFAR_DIR}/"
cp "${SCRIPT_DIR}/../AvgVsMax/pooling_former.py" "${CIFAR_DIR}/" 2>/dev/null || true

python - <<PYEOF
import os
train_path = os.path.join('${CIFAR_DIR}', 'train.py')
with open(train_path) as f:
    content = f.read()
if 'import alt_operator_former' not in content:
    content = content.replace(
        'from max_resnet import max_resnet18',
        'from max_resnet import max_resnet18\nimport alt_operator_former  # Exp 2.2'
    )
    with open(train_path, 'w') as f:
        f.write(content)
    print('  train.py patched.')
else:
    print('  train.py already patched.')
PYEOF

cd "${CIFAR_DIR}"
python alt_operator_former.py

echo ""
echo "[2.2-lap] Training Laplacian former..."
echo "          Start: $(date)"

python train.py \
    --experiment "Operators_lap" \
    --config     "${CIFAR_DIR}/cifar100_lap.yaml" \
    --data-path  "${DATA_DIR}" \
    --output     "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "[2.2-lap] Training done: $(date)"
cp "${EXP_DIR}/train.log" "${SCRIPT_DIR}/log_lap_${SLURM_JOB_ID}.txt"

echo ""
echo "========================================================"
echo "  Exp 2.2 Laplacian COMPLETE — $(date)"
echo "  Checkpoint: ${EXP_DIR}/Operators_lap/best.pth.tar"
echo "========================================================"
