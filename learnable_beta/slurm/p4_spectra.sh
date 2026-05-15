#!/bin/bash
#SBATCH --job-name=p4_spectra
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/p4_spectra_%j.out
#SBATCH --error=logs/p4_spectra_%j.err

# Priority 4: Spectral analysis on P3's best.pt (run after P3 finishes)

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${HOME}/data"
P3_CKPT="${SCRIPT_DIR}/results/p3_spikformer_learnable_b0/best.pt"
mkdir -p "${SCRIPT_DIR}/logs"

echo "[p4] job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

if [ ! -f "${P3_CKPT}" ]; then
    echo "[p4] ERROR: P3 checkpoint not found at ${P3_CKPT}"
    exit 1
fi

module purge
module load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate /iridisfs/scratch/nas1u21/.conda/envs/maxformer
export PATH="/iridisfs/scratch/nas1u21/.conda/envs/maxformer/bin:$PATH"

cd "${SCRIPT_DIR}"

python analyze_spectra.py \
    --checkpoint  "${P3_CKPT}" \
    --model       spikformer \
    --source-run  p3 \
    --data-path   "${DATA_DIR}" \
    --output-dir  results/p4_spectra

echo "[p4] finished at $(date)"
