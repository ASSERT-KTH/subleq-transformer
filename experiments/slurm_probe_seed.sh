#!/bin/bash
#SBATCH --job-name=subleq-probe
#SBATCH --output=logs/probe_seed%a_%j.out
#SBATCH --error=logs/probe_seed%a_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Run probing for a specific seed (can run in parallel with other seeds)

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

SEED=$SLURM_ARRAY_TASK_ID
echo "Probing seed $SEED on $(hostname)"
date

cd $EXP

python phase2_probe_trained.py \
    --seed $SEED \
    --n-examples 5000 \
    --n-steps 1000 \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results

echo "Probing done for seed $SEED"
date
