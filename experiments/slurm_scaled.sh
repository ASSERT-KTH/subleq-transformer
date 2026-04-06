#!/bin/bash
#SBATCH -A naiss2025-5-243
#SBATCH -J subleq_scaled_train
#SBATCH -t 06:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -N 1
#SBATCH -o logs/scaled_train_%j.out
#SBATCH -e logs/scaled_train_%j.err

set -e
module load Python/3.11.3-GCCcore-12.3.0
source /cephyr/users/andreafo/Alvis/subleq-venv/bin/activate

cd /mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer/experiments
mkdir -p logs

echo "=== Training capacity-sweep models ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# d=32, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=32 seed=${seed} ---"
    python train_scaled.py --d-model 32 --seed ${seed} --epochs 60 --lr 3e-4
done

# d=64, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=64 seed=${seed} ---"
    python train_scaled.py --d-model 64 --seed ${seed} --epochs 60 --lr 3e-4
done

# d=128, seeds 0-2
for seed in 0 1 2; do
    echo ""
    echo "--- d=128 seed=${seed} ---"
    python train_scaled.py --d-model 128 --seed ${seed} --epochs 60 --lr 3e-4
done

echo ""
echo "=== All training complete ==="
