#!/bin/bash
#SBATCH -A naiss2025-5-243
#SBATCH -J subleq_scaled_train
#SBATCH -t 06:00:00
#SBATCH --gpus-per-node=V100:1
#SBATCH -N 1
#SBATCH -o logs/scaled_train_%j.out
#SBATCH -e logs/scaled_train_%j.err

set -e
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

cd /mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer/experiments
mkdir -p logs

echo "=== Training capacity-sweep models ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# d=32, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=32 seed=${seed} ---"
    python3 train_scaled.py --d-model 32 --seed ${seed}
done

# d=64, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=64 seed=${seed} ---"
    python3 train_scaled.py --d-model 64 --seed ${seed}
done

# d=128, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=128 seed=${seed} ---"
    python3 train_scaled.py --d-model 128 --seed ${seed}
done

echo ""
echo "=== All training complete ==="
