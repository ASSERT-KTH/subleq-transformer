#!/usr/bin/env python3
"""
Phase 3 (constrained): Activation patching on constrained-LN model (seeds 0-2).

Uses identical focused metric as phase3_patching.py / phase5_oracle_patch.py:
  effect = clip01((logit_patched[p, token_A[p]] - logit_B[p, token_A[p]])
                 / (logit_A[p, token_A[p]] - logit_B[p, token_A[p]]))
averaged over changed positions.

Only probes constrained-LN (which learned the task). Skips no_ln.

Saves:
  results/phase3_constrained_ln.pkl
  results/phase3_constrained_ln_summary.json
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from subleq import step, generate_random_state, encode
from subleq.interpreter import MEM_SIZE, VALUE_MIN, VALUE_MAX, clamp
from subleq.tokenizer import get_changed_positions
from constrained_model import load_constrained_model

# Reuse contrast pair generation from phase3
from phase3_patching import generate_contrast_pairs, activation_patch_r2


CKPT_BASE = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')


def run_patching_experiment(model, pairs, device='cpu', n_pairs=200):
    n = min(n_pairs, len(pairs))
    all_effects = []
    for i in range(n):
        effects = activation_patch_r2(model, pairs[i], device=device)
        all_effects.append(effects)
        if (i + 1) % 50 == 0:
            print(f"    Patched {i+1}/{n} pairs")
    return torch.stack(all_effects).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pairs', type=int, default=500)
    parser.add_argument('--n-patch-per-type', type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 3 (constrained-LN): Activation Patching")
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pair_types = ['mem_a', 'mem_b', 'branch']
    print(f"\nGenerating contrast pairs ({args.n_pairs} per type)...")
    all_pairs = {ptype: generate_contrast_pairs(ptype, n=args.n_pairs, seed=42)
                 for ptype in pair_types}

    all_results = {}

    for seed in [0, 1, 2]:
        ckpt_path = os.path.join(CKPT_BASE, f'constrained_ln_seed{seed}', 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"  MISSING: {ckpt_path} — skipping")
            continue

        print(f"\n=== Patching constrained-LN seed {seed} ===")
        model, config = load_constrained_model(ckpt_path, device)
        n_layers = config.get('n_layers', 4)
        print(f"  d_model={config.get('d_model', 32)}, n_layers={n_layers}")

        seed_effects = {}
        for ptype in pair_types:
            print(f"  Pair type: {ptype}")
            mean_effects = run_patching_experiment(
                model, all_pairs[ptype], device=device,
                n_pairs=args.n_patch_per_type)
            seed_effects[ptype] = mean_effects.numpy()

            print(f"  Effect heatmap (max per layer):")
            for l in range(n_layers + 1):
                row_max = mean_effects[l].max().item()
                row_argmax = mean_effects[l].argmax().item()
                print(f"    L{l}: max={row_max:.4f} at pos {row_argmax}")

        all_results[seed] = {
            'ckpt': ckpt_path,
            'config': config,
            'effects': seed_effects,
        }

    if not all_results:
        print("No results — aborting")
        return

    pkl_path = os.path.join(RESULTS_DIR, 'phase3_constrained_ln.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'results': all_results, 'pair_types': pair_types}, f)
    print(f"\nSaved: {pkl_path}")

    # Build mean effects across seeds and summary JSON
    mean_across_seeds = {}
    for ptype in pair_types:
        stacked = np.stack([all_results[s]['effects'][ptype]
                            for s in all_results if ptype in all_results[s]['effects']])
        mean_across_seeds[ptype] = stacked.mean(axis=0)  # (n_layers+1, seq_len)

    json_summary = {}
    sample_config = next(iter(all_results.values()))['config']
    n_layers = sample_config.get('n_layers', 4)
    for ptype, effects in mean_across_seeds.items():
        effects_t = torch.tensor(effects)
        json_summary[ptype] = {
            f'layer_{l}': {
                'max': float(effects_t[l].max()),
                'argmax': int(effects_t[l].argmax()),
                'mean': float(effects_t[l].mean()),
            }
            for l in range(effects_t.shape[0])
        }

    json_summary['_meta'] = {
        'n_seeds': len(all_results),
        'seeds': list(all_results.keys()),
        'n_layers': n_layers,
        'pair_types': pair_types,
    }

    json_path = os.path.join(RESULTS_DIR, 'phase3_constrained_ln_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== Summary: mean effects (max over positions, mean over seeds) ===")
    for ptype in pair_types:
        effects_t = torch.tensor(mean_across_seeds[ptype])
        print(f"  {ptype}:")
        for l in range(n_layers + 1):
            print(f"    L{l}: max={effects_t[l].max():.4f} at pos {effects_t[l].argmax()}")


if __name__ == '__main__':
    main()
