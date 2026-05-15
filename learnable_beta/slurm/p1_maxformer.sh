#!/bin/bash
#SBATCH --job-name=p1_maxformer
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/p1_maxformer_%j.out
#SBATCH --error=logs/p1_maxformer_%j.err
#SBATCH --requeue

# Priority 1: Learnable-beta Max-Former on CIFAR-100, 150 epochs, seed=0

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${HOME}/data"
mkdir -p "${SCRIPT_DIR}/logs"

echo "[p1] job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate /iridisfs/scratch/nas1u21/.conda/envs/maxformer
export PATH="/iridisfs/scratch/nas1u21/.conda/envs/maxformer/bin:$PATH"

echo "[p1] python: $(which python)"
echo "[p1] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

cd "${SCRIPT_DIR}"

python train_plif.py \
    --config  configs/cifar100_maxformer.yaml \
    --model   max_former \
    --seed    0 \
    --data-path   "${DATA_DIR}" \
    --results-dir "${SCRIPT_DIR}/results" \
    --checkpoint-interval 25 \
    --beta-log-interval   5

echo "[p1] finished at $(date)"
