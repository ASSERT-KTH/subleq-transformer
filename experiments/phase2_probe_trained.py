#!/usr/bin/env python3
"""
Phase 2: Probe all trained model seeds, compare with oracle.

For each seed × layer × target, train a linear probe.
Produces:
- Comparison table: oracle vs trained
- Dimensional localization curves
- Training dynamics (probe at each checkpoint)
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from extract_residuals import load_r2_model, generate_metadata_dataset, get_r2_residuals_batched
from probe import (train_regression_probe, train_classification_probe,
                   probe_all_layers, localization_analysis)


def run_full_probe_battery(model, inputs, metadata, device='cpu', n_steps=1000):
    """Run full probe battery on a model."""
    print("  Extracting residuals...")
    residuals, logits = get_r2_residuals_batched(model, inputs, device=device)

    print("  Running probes...")
    results = probe_all_layers(residuals, metadata, n_steps=n_steps, device=device)
    return results, residuals


def run_localization_analysis(residuals, metadata, layer_idx, device='cpu'):
    """Run localization analysis on best layer."""
    N = len(metadata)
    n_val = int(N * 0.2)
    n_train = N - n_val

    targets = {
        'pc': (np.array([m['pc'] for m in metadata], dtype=np.float32), False),
        'mem_a': (np.array([m['mem_a'] for m in metadata], dtype=np.float32), False),
        'mem_b': (np.array([m['mem_b'] for m in metadata], dtype=np.float32), False),
        'delta': (np.array([m['delta'] for m in metadata], dtype=np.float32), False),
        'branch_taken': (np.array([m['branch_taken'] for m in metadata], dtype=np.int64), True),
    }

    resid = residuals[layer_idx]  # (N, seq_len, d_model)
    X = resid[:, 0, :]  # Use position 0

    localization_results = {}
    for tname, (y_arr, is_cls) in targets.items():
        X_train = torch.tensor(X[:n_train])
        X_val = torch.tensor(X[n_train:])
        y_tr = torch.tensor(y_arr[:n_train])
        y_vl = torch.tensor(y_arr[n_train:])

        # Get weights from full probe
        if is_cls:
            _, weights = train_classification_probe(X_train, y_tr, X_val, y_vl,
                                                    n_steps=500, device=device)
        else:
            _, weights = train_regression_probe(X_train, y_tr, X_val, y_vl,
                                               n_steps=500, device=device)

        loc_results = localization_analysis(X_train, y_tr, X_val, y_vl, weights,
                                            n_steps=500, device=device,
                                            is_classification=is_cls)
        localization_results[tname] = loc_results

    return localization_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Specific seed to probe (or None=all)')
    parser.add_argument('--n-examples', type=int, default=5000)
    parser.add_argument('--n-steps', type=int, default=1000)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 2: Probing trained models")
    print(f"Device: {device}")

    # Set directories
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(script_dir, 'checkpoints')
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'results')
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all seed checkpoints
    # Seed 0: original best_model.pt from round2_trained
    seed0_ckpt = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')

    seeds_to_run = []
    if args.seed is None or args.seed == 0:
        if os.path.exists(seed0_ckpt):
            seeds_to_run.append((0, seed0_ckpt))

    for seed in range(1, 5):
        if args.seed is not None and args.seed != seed:
            continue
        ckpt_path = os.path.join(args.ckpt_dir, f'seed{seed}_final.pt')
        if os.path.exists(ckpt_path):
            seeds_to_run.append((seed, ckpt_path))
        else:
            print(f"  WARNING: seed {seed} checkpoint not found at {ckpt_path}")

    if not seeds_to_run:
        print("No checkpoints found! Run train_seeds.py first.")
        return

    print(f"Found {len(seeds_to_run)} seeds: {[s for s, _ in seeds_to_run]}")

    # Generate shared dataset
    print(f"Generating {args.n_examples} probe examples...")
    inputs, metadata = generate_metadata_dataset(n=args.n_examples, seed=42)

    all_seed_results = {}
    for seed_id, ckpt_path in seeds_to_run:
        print(f"\n=== Seed {seed_id} ({ckpt_path}) ===")
        model, config = load_r2_model(ckpt_path, device)
        n_layers = config.get('n_layers', 6)
        d_model = config.get('d_model', 256)
        print(f"  Model: d_model={d_model}, n_layers={n_layers}")

        results, residuals = run_full_probe_battery(model, inputs, metadata,
                                                     device=device, n_steps=args.n_steps)

        # Find best layer for each target
        best_layers = {}
        for tname in ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']:
            best = -1
            best_layer = -1
            for layer in sorted(results.keys()):
                m = results[layer].get('pos0', {}).get(tname, {}).get('metric', -1)
                if m > best:
                    best = m
                    best_layer = layer
            best_layers[tname] = (best_layer, best)
            print(f"  Best layer for {tname}: L{best_layer} ({best:.3f})")

        # Localization analysis at best layer for pc
        best_pc_layer = best_layers['pc'][0]
        if best_pc_layer >= 0:
            print(f"  Running localization analysis at L{best_pc_layer}...")
            loc = run_localization_analysis(residuals, metadata, best_pc_layer, device=device)
        else:
            loc = {}

        all_seed_results[seed_id] = {
            'ckpt': ckpt_path,
            'config': config,
            'probe_results': results,
            'best_layers': best_layers,
            'localization': loc,
        }

    # Also run checkpoint (training dynamics) probing if available
    print("\n=== Training dynamics (checkpoint probing) ===")
    dynamics_results = {}
    for frac in [10, 25, 50, 75, 100]:
        for seed_id in range(1, 5):
            ckpt_path = os.path.join(args.ckpt_dir, f'seed{seed_id}_frac{frac:03d}.pt')
            if os.path.exists(ckpt_path):
                if frac not in dynamics_results:
                    dynamics_results[frac] = {}
                print(f"  Probing seed {seed_id} at {frac}% of training...")
                model, config = load_r2_model(ckpt_path, device)
                results, _ = run_full_probe_battery(model, inputs, metadata,
                                                     device=device, n_steps=500)
                dynamics_results[frac][seed_id] = results

    # Print comparison table
    print("\n=== Phase 2 Results Summary ===")
    header_suffix = "".join(f" S{s}:L{s}" for s, _ in seeds_to_run)
    print(f"{'Quantity':<15} {'Layer':>6}" + header_suffix)

    # Save all results
    output = {
        'all_seed_results': all_seed_results,
        'dynamics_results': dynamics_results,
        'n_examples': args.n_examples,
    }
    out_path = os.path.join(args.output_dir, 'phase2_probe_trained.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nResults saved to {out_path}")

    # Print readable summary
    print("\n=== Probe accuracy by layer (mean across seeds, position 0) ===")
    targets = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']
    n_layers = 7  # 0=embed + 6 transformer layers

    for tname in targets:
        row = f"{tname:<15}"
        for layer in range(n_layers):
            metrics = []
            for seed_id, seed_data in all_seed_results.items():
                m = seed_data['probe_results'].get(layer, {}).get('pos0', {}).get(tname, {}).get('metric', None)
                if m is not None:
                    metrics.append(m)
            if metrics:
                mean_m = np.mean(metrics)
                std_m = np.std(metrics)
                row += f" {mean_m:.3f}±{std_m:.3f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # Save JSON summary
    json_summary = {
        'targets': targets,
        'n_seeds': len(all_seed_results),
        'seeds': list(all_seed_results.keys()),
        'probe_means': {},
    }
    for tname in targets:
        json_summary['probe_means'][tname] = {}
        for layer in range(n_layers):
            metrics = []
            for seed_data in all_seed_results.values():
                m = seed_data['probe_results'].get(layer, {}).get('pos0', {}).get(tname, {}).get('metric', None)
                if m is not None:
                    metrics.append(m)
            if metrics:
                json_summary['probe_means'][tname][str(layer)] = {
                    'mean': float(np.mean(metrics)),
                    'std': float(np.std(metrics)),
                    'n': len(metrics),
                }
    with open(os.path.join(args.output_dir, 'phase2_summary.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary saved.")


if __name__ == '__main__':
    main()
