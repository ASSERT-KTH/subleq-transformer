#!/bin/bash
#SBATCH --job-name=subleq-trace
#SBATCH --output=logs/trace_%j.out
#SBATCH --error=logs/trace_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=00:20:00
#SBATCH --account=naiss2025-5-243

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

echo "Start: $(date)"
cd $EXP

python3 phase6_additional.py \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results \
    --n-examples 3000 \
    --skip-localization \
    --skip-dynamics 2>&1

python3 generate_figures.py \
    --results-dir $EXP/results \
    --output-dir $EXP/figures 2>&1

python3 generate_report.py \
    --results-dir $EXP/results \
    --figures-dir $EXP/figures \
    --output $REPO/research_report.md 2>&1

echo "Done: $(date)"
