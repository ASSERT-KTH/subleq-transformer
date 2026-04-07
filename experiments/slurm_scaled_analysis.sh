#!/bin/bash
#SBATCH -A naiss2025-5-243
#SBATCH -J subleq_scaled_analysis
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=V100:1
#SBATCH -N 1
#SBATCH -o logs/scaled_analysis_%j.out
#SBATCH -e logs/scaled_analysis_%j.err

set -e
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

cd /mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer/experiments
mkdir -p logs

echo "=== Phase 2 (scaled): Linear probing all capacity models ==="
python3 phase2_probe_scaled.py --n-data 5000 --n-steps 500

echo ""
echo "=== Phase 3 (scaled): Activation patching all capacity models ==="
python3 phase3_patch_scaled.py --n-pairs 500 --n-patch 200

echo ""
echo "=== Phase 7: Distributional metrics & hypothesis tests ==="
python3 phase7_metrics.py --n-data 2000 --n-rsa 500 --n-perm 1000

echo ""
echo "=== Generating figures ==="
python3 generate_figures.py

echo ""
echo "=== Generating report ==="
python3 generate_report.py

echo ""
echo "=== Analysis pipeline complete ==="
ls -lh results/phase2_scaled_summary.json results/phase3_scaled_summary.json results/phase7_metrics.json
