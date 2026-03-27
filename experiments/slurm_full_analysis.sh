#!/bin/bash
#SBATCH --job-name=subleq-full
#SBATCH --output=logs/full_analysis_%j.out
#SBATCH --error=logs/full_analysis_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=08:00:00
#SBATCH --account=naiss2025-5-243

# Full analysis after all seeds have trained.
# Runs Phases 2-4 on all seeds, then generates report.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP

# Phase 2: Probe all seeds (including seeds 1-4)
echo "=== Phase 2: Probing all seeds ==="
python3 phase2_probe_trained.py \
    --n-examples 5000 \
    --n-steps 1000 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results 2>&1
echo "Phase 2 done: $(date)"

# Phase 3: Patching all seeds
echo "=== Phase 3: Patching all seeds ==="
python3 phase3_patching.py \
    --n-pairs 500 \
    --n-patch-per-type 200 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results 2>&1
echo "Phase 3 done: $(date)"

# Phase 4: Failure analysis all seeds
echo "=== Phase 4: Failures all seeds ==="
python3 phase4_failures.py \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results 2>&1
echo "Phase 4 done: $(date)"

# Generate report
echo "=== Generating report ==="
python3 generate_report.py \
    --results-dir $EXP/results \
    --output $REPO/research_report.md 2>&1
echo "Report done: $(date)"

echo "All done: $(date)"
