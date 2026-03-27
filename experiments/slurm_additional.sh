#!/bin/bash
#SBATCH --job-name=subleq-additional
#SBATCH --output=logs/additional_%j.out
#SBATCH --error=logs/additional_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=06:00:00
#SBATCH --account=naiss2025-5-243

# Additional analyses: oracle patching, localization, training dynamics,
# failure trace, figure generation.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
mkdir -p logs figures

# Phase 5: Oracle activation patching
echo "=== Phase 5: Oracle Activation Patching ==="
python3 phase5_oracle_patch.py 2>&1
echo "Phase 5 done: $(date)"

# Phase 6: Dimensional localization + training dynamics + failure trace
echo "=== Phase 6: Additional Analyses ==="
python3 phase6_additional.py \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results \
    --n-examples 3000 2>&1
echo "Phase 6 done: $(date)"

# Generate all figures (figs 1-9)
echo "=== Generating Figures ==="
python3 generate_figures.py \
    --results-dir $EXP/results \
    --output-dir $EXP/figures 2>&1
echo "Figures done: $(date)"

# Update research report
echo "=== Updating Report ==="
python3 generate_report.py \
    --results-dir $EXP/results \
    --figures-dir $EXP/figures \
    --output $REPO/research_report.md 2>&1
echo "Report done: $(date)"

echo "All done: $(date)"
# Print figure listing
ls -lh $EXP/figures/
