#!/bin/bash
#SBATCH --job-name=Beta_Sweep_b02
#SBATCH --nodes=1
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --partition=ecsstudents_l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/Beta_Sweep_b02_%j.out
#SBATCH --error=logs/Beta_Sweep_b02_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kb6g21@soton.ac.uk

# =============================================================================
# Experiment 3.2 — β sweep: β≈0.20 (τ=1.25)
# Shortest time constant — highest-pass LIF neuron.
# Run from experiments/Beta_Sweep/:
#   mkdir -p logs && sbatch slurm_beta_02.sh
# =============================================================================

set -euo pipefail

BETA_TAG="02"
TAU="1.25"
BETA="0.20"
MODEL="pool_former_beta_02"

echo "========================================================"
echo "  Exp 3.2 — β sweep  β=${BETA} (τ=${TAU})"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time   : $(date)"
echo "========================================================"

REPO_DIR="${HOME}/Repo/experiments/MaxFormer"
DATA_DIR="/scratch/${USER}/maxformer_repro/data"
EXP_DIR="/scratch/${USER}/maxformer_repro/Beta_Sweep/b${BETA_TAG}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
CIFAR_DIR="${REPO_DIR}/cifar10-100"

rm -rf "${EXP_DIR}"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

module purge
module load conda/python3
for cuda_mod in cuda/12.4.0 cuda/12.2.0 cuda/11.8.0 cuda/12.5.1 cuda/12.6.3; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[3.2-b${BETA_TAG}] Loaded: $cuda_mod"; break
    fi
done

export PATH="/home/kb6g21/.conda/envs/maxformer/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ── Inject scripts ────────────────────────────────────────────────────────────
cp "${SCRIPT_DIR}/beta_sweep_former.py"       "${CIFAR_DIR}/"
cp "${SCRIPT_DIR}/../AvgVsMax/pooling_former.py" "${CIFAR_DIR}/" 2>/dev/null || true
cp "${SCRIPT_DIR}/cifar100_beta_${BETA_TAG}.yaml" "${CIFAR_DIR}/"

# ── Patch train.py (idempotent) ───────────────────────────────────────────────
export CIFAR_DIR="${CIFAR_DIR}"
python - <<'PYEOF'
import os, sys
train_path = os.path.join(os.environ['CIFAR_DIR'], 'train.py')
with open(train_path) as f:
    content = f.read()
if 'import beta_sweep_former' in content:
    print('  train.py already patched for beta_sweep_former.')
    sys.exit(0)
anchors = [
    'import ann_maxformer  # Exp 3.1',
    'import alt_operator_former  # Exp 2.2',
    'import pooling_former',
    'from max_resnet import max_resnet18',
]
for anchor in anchors:
    if anchor in content:
        content = content.replace(anchor, anchor + '\nimport beta_sweep_former  # Exp 3.2', 1)
        with open(train_path, 'w') as f:
            f.write(content)
        print('  train.py patched for beta_sweep_former.')
        sys.exit(0)
print('  ERROR: Could not find anchor.', file=sys.stderr)
sys.exit(1)
PYEOF

# ── Sanity check ──────────────────────────────────────────────────────────────
cd "${CIFAR_DIR}"
python beta_sweep_former.py

# ── Training ──────────────────────────────────────────────────────────────────
echo ""
echo "[3.2] Training ${MODEL} (τ=${TAU}, β=${BETA}, 200 epochs)..."
echo "      Start: $(date)"

python train.py \
    --experiment "Beta_Sweep_b${BETA_TAG}" \
    --config     "${CIFAR_DIR}/cifar100_beta_${BETA_TAG}.yaml" \
    --data-path  "${DATA_DIR}" \
    --output     "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "[3.2-b${BETA_TAG}] Training done: $(date)"

# ── Copy results back ─────────────────────────────────────────────────────────
cp "${EXP_DIR}/train.log" "${SCRIPT_DIR}/log_b${BETA_TAG}_${SLURM_JOB_ID}.txt"
[ -f "${EXP_DIR}/Beta_Sweep_b${BETA_TAG}/summary.csv" ] && \
    cp "${EXP_DIR}/Beta_Sweep_b${BETA_TAG}/summary.csv" "${SCRIPT_DIR}/summary_b${BETA_TAG}_${SLURM_JOB_ID}.csv" || true

echo ""
echo "========================================================"
echo "  Exp 3.2 β=${BETA} COMPLETE — $(date)"
echo "  Log: ${SCRIPT_DIR}/log_b${BETA_TAG}_${SLURM_JOB_ID}.txt"
echo "========================================================"
