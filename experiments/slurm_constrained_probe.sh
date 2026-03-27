#!/bin/bash
#SBATCH --job-name=subleq-constrained-probe
#SBATCH --output=logs/constrained_probe_%j.out
#SBATCH --error=logs/constrained_probe_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=02:00:00
#SBATCH --account=naiss2025-5-243

# Probe constrained models (ln and no_ln, seeds 0-2),
# then regenerate figures and report.

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

echo "=== Phase 2 (constrained): Probe all constrained models ==="
python3 phase2_probe_constrained.py \
    --n-data 5000 \
    --n-steps 500 2>&1
echo "Constrained probing done: $(date)"

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
ls -lh $EXP/results/phase2_constrained_*.json
ls -lh $EXP/figures/
