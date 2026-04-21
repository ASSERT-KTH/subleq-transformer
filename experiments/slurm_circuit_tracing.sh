#!/bin/bash
#SBATCH --job-name=subleq-circuits
#SBATCH --output=logs/circuit_tracing_%j.out
#SBATCH --error=logs/circuit_tracing_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=04:00:00
#SBATCH --account=naiss2025-5-243

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

mkdir -p $REPO/logs

cd $EXP

# ── Unit tests first ──────────────────────────────────────────────────────────
echo "=== Running circuit tracing unit tests ==="
python3 -m pytest circuit_tracing/tests/ -v --tb=short 2>&1
echo "Tests done: $(date)"

# ── Phase 8: Circuit tracing ──────────────────────────────────────────────────
echo "=== Phase 8: Circuit tracing ==="
python3 phase8_circuit_tracing.py \
    --ckpt-dir  $EXP/checkpoints \
    --output-dir $EXP/results \
    --figures-dir $REPO/figures \
    --n-examples 1000 \
    --seeds 0 1 2 \
    --models constrained_ln \
    --device cuda \
    2>&1

echo "Phase 8 done: $(date)"

# ── Also run on base trained model if checkpoints exist ───────────────────────
if ls $EXP/checkpoints/seed0/best_model.pt 2>/dev/null; then
    echo "=== Phase 8: Base trained model ==="
    python3 phase8_circuit_tracing.py \
        --ckpt-dir  $EXP/checkpoints \
        --output-dir $EXP/results \
        --figures-dir $REPO/figures \
        --n-examples 1000 \
        --seeds 0 1 2 \
        --models base_trained \
        --device cuda \
        --skip-hardcoded \
        2>&1
    echo "Base model done: $(date)"
fi

echo "All done: $(date)"
