#!/bin/bash
#SBATCH --job-name=p2_spectra
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/p2_spectra_%j.out
#SBATCH --error=logs/p2_spectra_%j.err

# Priority 2: Spectral analysis on P1's best.pt (run after P1 finishes)

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${HOME}/data"
P1_CKPT="${SCRIPT_DIR}/results/p1_maxformer_learnable_b0/best.pt"
mkdir -p "${SCRIPT_DIR}/logs"

echo "[p2] job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

if [ ! -f "${P1_CKPT}" ]; then
    echo "[p2] ERROR: P1 checkpoint not found at ${P1_CKPT}"
    exit 1
fi

module purge
module load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate /iridisfs/scratch/nas1u21/.conda/envs/maxformer
export PATH="/iridisfs/scratch/nas1u21/.conda/envs/maxformer/bin:$PATH"

cd "${SCRIPT_DIR}"

python analyze_spectra.py \
    --checkpoint  "${P1_CKPT}" \
    --model       max_former \
    --source-run  p1 \
    --data-path   "${DATA_DIR}" \
    --output-dir  results/p2_spectra

echo "[p2] finished at $(date)"
