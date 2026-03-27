#!/bin/bash
#SBATCH --job-name=subleq-analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=06:00:00
#SBATCH --account=naiss2025-5-243

# Run all analysis phases after training is complete.
# Phases 1, 2, 3, 4 in sequence.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Starting analysis on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

cd $EXP

# Phase 1: Oracle probing
echo "=== Phase 1: Oracle ==="
python phase1_oracle.py
echo "Phase 1 done"; date

# Phase 2: Probe trained models
echo "=== Phase 2: Probing trained models ==="
python phase2_probe_trained.py \
    --n-examples 5000 \
    --n-steps 1000 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results
echo "Phase 2 done"; date

# Phase 3: Activation patching
echo "=== Phase 3: Activation patching ==="
python phase3_patching.py \
    --n-pairs 500 \
    --n-patch-per-type 100 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results
echo "Phase 3 done"; date

# Phase 4: Failure analysis
echo "=== Phase 4: Failure cases ==="
python phase4_failures.py \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results
echo "Phase 4 done"; date

echo "All analysis phases complete!"
date
