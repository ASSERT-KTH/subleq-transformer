#!/bin/bash
#SBATCH --job-name=subleq-constrained
#SBATCH --output=logs/constrained_%j.out
#SBATCH --error=logs/constrained_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=06:00:00
#SBATCH --account=naiss2025-5-243

# Train constrained transformer (oracle footprint: d=32, 4L, 8H, d_ff=64, ReLU)
# Two variants: with and without LayerNorm. Seeds 0-2 each.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
mkdir -p logs checkpoints

# Variant: with LayerNorm (closer to trained model style)
for SEED in 0 1 2; do
    echo "=== Training constrained LN seed ${SEED} ==="
    python3 train_constrained.py \
        --variant ln \
        --seed $SEED \
        --total-steps 80000 \
        --batch-size 256 \
        --lr 3e-4 \
        --ckpt-base $EXP/checkpoints 2>&1
    echo "LN seed ${SEED} done: $(date)"
done

# Variant: without LayerNorm (closer to oracle style)
for SEED in 0 1 2; do
    echo "=== Training constrained no_ln seed ${SEED} ==="
    python3 train_constrained.py \
        --variant no_ln \
        --seed $SEED \
        --total-steps 80000 \
        --batch-size 256 \
        --lr 3e-4 \
        --ckpt-base $EXP/checkpoints 2>&1
    echo "no_ln seed ${SEED} done: $(date)"
done

echo "All constrained training done: $(date)"
ls -lh $EXP/checkpoints/
