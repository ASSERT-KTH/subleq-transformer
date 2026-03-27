#!/bin/bash
#SBATCH --job-name=subleq-analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --gpus-per-node=V100:1
#SBATCH --time=04:00:00
#SBATCH --account=naiss2025-5-243

# Re-run patching (focused metric), held-out probes, fix Fig 9 + Fig 6, update report.

set -e
REPO=/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer
EXP=$REPO/experiments

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

cd $EXP
mkdir -p logs figures results

# Phase 3: Activation patching (focused metric, all 5 seeds)
echo "=== Phase 3: Activation Patching (focused metric) ==="
python3 phase3_patching.py \
    --ckpt-dir $EXP/checkpoints \
    --output-dir $EXP/results \
    --n-pairs 500 \
    --n-patch-per-type 200 2>&1
echo "Phase 3 done: $(date)"

# Phase 5: Oracle patching (focused metric)
echo "=== Phase 5: Oracle Patching (focused metric) ==="
python3 phase5_oracle_patch.py 2>&1
echo "Phase 5 done: $(date)"

# Phase 2 (held-out): Distribution generalization of probes
echo "=== Phase 2 (held-out): Held-Out Probes ==="
python3 phase2_heldout.py \
    --n-train 5000 \
    --n-heldout 1000 \
    --n-steps 500 2>&1
echo "Phase 2 (held-out) done: $(date)"

# Regenerate all figures (includes Fig 9 fix + Fig 6 threshold fix)
echo "=== Generating Figures ==="
python3 generate_figures.py \
    --results-dir $EXP/results \
    --output-dir $EXP/figures 2>&1
echo "Figures done: $(date)"

# Update report
echo "=== Updating Report ==="
python3 generate_report.py \
    --results-dir $EXP/results \
    --figures-dir $EXP/figures \
    --output $REPO/research_report.md 2>&1
echo "Report done: $(date)"

echo "All done: $(date)"
ls -lh $EXP/figures/
