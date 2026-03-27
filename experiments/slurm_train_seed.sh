#!/bin/bash
#SBATCH --job-name=subleq-train
#SBATCH --output=logs/train_seed%a_%j.out
#SBATCH --error=logs/train_seed%a_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --account=naiss2025-5-243

# Train seeds 1-4 of the SUBLEQ transformer
# Seed 0 already exists in round2_trained/checkpoints/best_model.pt

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

cd $REPO/round2_trained

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Starting training for seed $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

python -u $EXP/train_seeds.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --output-dir $EXP/checkpoints \
    --total-steps 80000 \
    --batch-size 256 \
    --data-size 50000

echo "Training done for seed $SLURM_ARRAY_TASK_ID"
date
