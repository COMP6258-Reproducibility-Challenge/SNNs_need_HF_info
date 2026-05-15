#!/bin/bash
# submit_all.sh — Submit all three Exp 2.2 jobs in parallel.
# Run from experiments/Operators/:
#   mkdir -p logs && bash submit_all.sh
#
# Each job runs independently (~13h each on a separate GPU node).
# All three can run simultaneously if enough L4 nodes are available.

set -euo pipefail

mkdir -p logs

echo "Submitting Exp 2.2 — Alternative High-Frequency Operators"
echo "==========================================================="

JOB_MIN=$(sbatch --parsable slurm_min.sh)
echo "  MinPool        job ID: ${JOB_MIN}"

JOB_LAP=$(sbatch --parsable slurm_lap.sh)
echo "  Laplacian      job ID: ${JOB_LAP}"

JOB_LHP=$(sbatch --parsable slurm_learned_hp.sh)
echo "  Learned-HP     job ID: ${JOB_LHP}"

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After all complete, run compare_operators.py:"
echo "  python compare_operators.py \\"
echo "    --exp11-avg-log /scratch/\$USER/maxformer_repro/AvgVsMax/avg/train.log \\"
echo "    --exp11-max-log /scratch/\$USER/maxformer_repro/AvgVsMax/max/train.log \\"
echo "    --min-log       /scratch/\$USER/maxformer_repro/Operators_min/train.log \\"
echo "    --lap-log       /scratch/\$USER/maxformer_repro/Operators_lap/train.log \\"
echo "    --learned-hp-log /scratch/\$USER/maxformer_repro/Operators_learned_hp/train.log \\"
echo "    --output        ./comparison_results"
echo ""
echo "Job IDs: min=${JOB_MIN}  lap=${JOB_LAP}  learned_hp=${JOB_LHP}"
echo "Save these for --dependency in later experiments (e.g. Exp 2.1 FFT on new ckpts)."
