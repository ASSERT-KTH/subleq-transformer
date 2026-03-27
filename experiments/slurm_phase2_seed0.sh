#!/bin/bash
#SBATCH --job-name=subleq-p2s0
#SBATCH --output=logs/phase2_seed0_%j.out
#SBATCH --error=logs/phase2_seed0_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=03:00:00
#SBATCH --account=naiss2025-5-243

# Re-run Phase 2/3/4 on seed 0 (after fixing syntax error)

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP

# Phase 2: Probe seed 0 only
echo "=== Phase 2: Probe seed 0 ==="
python3 phase2_probe_trained.py \
    --seed 0 \
    --n-examples 5000 \
    --n-steps 1000 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results 2>&1
echo "Phase 2 done: $(date)"

# Phase 3: Patching on seed 0
echo "=== Phase 3: Patching seed 0 ==="
python3 phase3_patching.py \
    --seed 0 \
    --ckpt $REPO/round2_trained/checkpoints/best_model.pt \
    --n-pairs 500 \
    --n-patch-per-type 200 \
    --output-dir $EXP/results 2>&1
echo "Phase 3 done: $(date)"

# Phase 4: Failure analysis on seed 0
echo "=== Phase 4: Failures seed 0 ==="
python3 phase4_failures.py \
    --seed 0 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results 2>&1
echo "Phase 4 done: $(date)"

# Generate partial report
echo "=== Generating partial report ==="
python3 generate_report.py \
    --results-dir $EXP/results \
    --output $REPO/research_report_partial.md 2>&1

echo "All done: $(date)"
