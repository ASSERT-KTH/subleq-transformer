#!/bin/bash
#SBATCH --job-name=subleq-constrained-patch
#SBATCH --output=logs/constrained_patch_%j.out
#SBATCH --error=logs/constrained_patch_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=02:00:00
#SBATCH --account=naiss2025-5-243

# Activation patching on constrained-LN (seeds 0-2), then regenerate
# figures (fig10 now 4-panel, new fig11 patching comparison) and report.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
mkdir -p logs results figures

echo "=== Phase 3 (constrained-LN): Activation Patching ==="
python3 phase3_patch_constrained.py \
    --n-pairs 500 \
    --n-patch-per-type 200 2>&1
echo "Constrained patching done: $(date)"

echo "=== Generating Figures ==="
python3 generate_figures.py \
    --results-dir $EXP/results \
    --output-dir $EXP/figures 2>&1
echo "Figures done: $(date)"

echo "=== Updating Report ==="
python3 generate_report.py \
    --results-dir $EXP/results \
    --figures-dir $EXP/figures \
    --output $REPO/research_report.md 2>&1
echo "Report done: $(date)"

echo "All done: $(date)"
ls -lh $EXP/results/phase3_constrained_ln*.json
ls -lh $EXP/figures/fig10* $EXP/figures/fig11*
