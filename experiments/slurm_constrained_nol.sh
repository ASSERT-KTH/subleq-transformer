#!/bin/bash
#SBATCH --job-name=subleq-constrained-nol
#SBATCH --output=logs/constrained_nol_%j.out
#SBATCH --error=logs/constrained_nol_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=04:00:00
#SBATCH --account=naiss2025-5-243

# Train constrained transformer without LayerNorm (no_ln variant, seeds 0-2).
# Re-run after fixing Identity.__init__ to accept d_model arg.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
mkdir -p logs checkpoints

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

echo "All no_ln training done: $(date)"
ls -lh $EXP/checkpoints/constrained_no_ln_*/
