#!/bin/bash
#SBATCH --job-name=subleq-p1r
#SBATCH --output=logs/phase1_rerun_%j.out
#SBATCH --error=logs/phase1_rerun_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=01:00:00
#SBATCH --account=naiss2025-5-243

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
python3 -u phase1_oracle.py 2>&1
echo "Phase 1 rerun done: $(date)"
