#!/usr/bin/env python3
"""
Phase 2 (constrained): Probe constrained models (ln and no_ln variants, seeds 0-2).

Uses the same probing infrastructure as phase2_probe_trained.py.
Constrained models share the same forward pass interface as round2 (MiniSUBLEQTransformer),
so get_r2_residuals_batched works directly on them.

Output format matches phase2_summary.json:
  {targets, n_seeds, seeds, probe_means: {target: {layer_str: {mean, std, n}}}}

Saves:
  results/phase2_constrained_ln.json
  results/phase2_constrained_no_ln.json
  results/phase2_constrained_summary.json  (combined for figure generation)
"""

import os
import sys
import json
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from extract_residuals import get_r2_residuals_batched, generate_metadata_dataset
from constrained_model import load_constrained_model
from probe import probe_all_layers

TARGETS = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']
CKPT_BASE = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')


def probe_one_model(ckpt_path, inputs, metadata, device='cpu', n_steps=500):
    """Load constrained model, extract residuals, run probe battery. Returns per-layer metrics."""
    print(f"  Loading {ckpt_path}...")
    model, config = load_constrained_model(ckpt_path, device)
    n_layers = config.get('n_layers', 4)
    print(f"  d_model={config.get('d_model', 32)}, n_layers={n_layers}, "
          f"layer_norm={config.get('layer_norm', True)}")

    print("  Extracting residuals...")
    residuals, logits = get_r2_residuals_batched(model, inputs, device=device)

    print("  Running probes (pos0 only)...")
    results = probe_all_layers(residuals, metadata, n_steps=n_steps, device=device)

    # Extract per-layer, per-target best metric at pos0
    layer_metrics = {}  # target -> {layer_idx -> metric}
    for layer_idx, layer_res in results.items():
        pos0 = layer_res.get('pos0', {})
        for tname in TARGETS:
            if tname not in layer_metrics:
                layer_metrics[tname] = {}
            m = pos0.get(tname, {}).get('metric', float('nan'))
            layer_metrics[tname][layer_idx] = m

    return layer_metrics, n_layers


def summarize_seeds(all_seed_metrics):
    """
    Given list of per-seed dicts {target: {layer_idx: metric}},
    compute mean/std across seeds per layer.
    Returns {target: {layer_str: {mean, std, n}}}
    """
    targets = list(all_seed_metrics[0].keys())
    layer_idxs = sorted(all_seed_metrics[0][targets[0]].keys())

    probe_means = {}
    for tname in targets:
        probe_means[tname] = {}
        for lidx in layer_idxs:
            vals = [s[tname][lidx] for s in all_seed_metrics
                    if lidx in s[tname] and not np.isnan(s[tname][lidx])]
            probe_means[tname][str(lidx)] = {
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                'n': len(vals),
            }
    return probe_means


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-data', type=int, default=5000)
    parser.add_argument('--n-steps', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 2 (constrained): Probing constrained models")
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nGenerating {args.n_data} random-state examples...")
    inputs, metadata = generate_metadata_dataset(n=args.n_data, seed=42)
    print(f"  Generated {len(metadata)} examples")

    combined = {}  # variant -> {seeds: [...], probe_means: ...}

    for variant in ['ln', 'no_ln']:
        print(f"\n{'='*60}")
        print(f"Variant: constrained_{variant}")
        print(f"{'='*60}")

        all_seed_metrics = []
        seed_list = []

        for seed in [0, 1, 2]:
            ckpt_dir = os.path.join(CKPT_BASE, f'constrained_{variant}_seed{seed}')
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
            if not os.path.exists(ckpt_path):
                print(f"  MISSING: {ckpt_path} — skipping seed {seed}")
                continue

            print(f"\n  Seed {seed}:")
            layer_metrics, n_layers = probe_one_model(
                ckpt_path, inputs, metadata, device=device, n_steps=args.n_steps)
            all_seed_metrics.append(layer_metrics)
            seed_list.append(seed)

            # Print per-seed summary
            for tname in TARGETS:
                best_layer = max(layer_metrics[tname], key=layer_metrics[tname].get)
                best_val = layer_metrics[tname][best_layer]
                print(f"    {tname}: best L{best_layer} = {best_val:.3f}")

        if not all_seed_metrics:
            print(f"  No checkpoints found for variant {variant}")
            continue

        probe_means = summarize_seeds(all_seed_metrics)
        result = {
            'variant': variant,
            'targets': TARGETS,
            'n_seeds': len(seed_list),
            'seeds': seed_list,
            'probe_means': probe_means,
        }

        out_path = os.path.join(RESULTS_DIR, f'phase2_constrained_{variant}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # Print summary table
        print(f"\n  Summary ({variant}) — mean across seeds (best layer):")
        for tname in TARGETS:
            pm = probe_means[tname]
            best_l = max(pm.keys(), key=lambda k: pm[k]['mean']
                         if not np.isnan(pm[k]['mean']) else -999)
            print(f"    {tname:<15}: L{best_l} = {pm[best_l]['mean']:.3f} ± {pm[best_l]['std']:.3f}")

        combined[variant] = result

    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, 'phase2_constrained_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined summary saved: {summary_path}")


if __name__ == '__main__':
    main()
