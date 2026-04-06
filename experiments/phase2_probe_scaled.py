#!/usr/bin/env python3
"""
Phase 2 (scaled): Probe all capacity-sweep models.

Models:
  constrained-LN  (d=32, 4L, ReLU)           — seeds 0-2
  scaled-d32      (d=32, 6L, GELU, LN)       — seeds 0-2
  scaled-d64      (d=64, 6L, GELU, LN)       — seeds 0-2
  scaled-d128     (d=128, 6L, GELU, LN)      — seeds 0-2
  trained-d256    (d=256, 6L, GELU, LN)      — seeds 0-4  (existing checkpoints)

Output: results/phase2_scaled_summary.json
  {model_key: {d, n_layers, n_seeds, seeds, probe_means: {target: {layer: {mean,std,n}}}}}
"""

import os, sys, json
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from extract_residuals import get_r2_residuals_batched, generate_metadata_dataset, load_r2_model
from constrained_model import load_constrained_model
from probe import probe_all_layers

TARGETS = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']
CKPT_BASE = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')


def best_metric(probe_means, tname):
    tdata = probe_means.get(tname, {})
    vals = [v['mean'] for v in tdata.values() if isinstance(v, dict)
            and not np.isnan(v.get('mean', float('nan')))]
    return max(vals) if vals else float('nan')


def probe_one(ckpt_path, inputs, metadata, device, n_steps=500, is_constrained=False):
    if is_constrained:
        model, config = load_constrained_model(ckpt_path, device)
    else:
        model, config = load_r2_model(ckpt_path, device)

    residuals, _ = get_r2_residuals_batched(model, inputs, device=device)
    results = probe_all_layers(residuals, metadata, n_steps=n_steps, device=device)

    layer_metrics = {}  # target -> {layer_idx -> metric}
    for layer_idx, layer_res in results.items():
        pos0 = layer_res.get('pos0', {})
        for tname in TARGETS:
            layer_metrics.setdefault(tname, {})[layer_idx] = pos0.get(tname, {}).get('metric', float('nan'))

    return layer_metrics, config


def summarize_seeds(all_seed_metrics):
    targets = list(all_seed_metrics[0].keys())
    layer_idxs = sorted(all_seed_metrics[0][targets[0]].keys())
    probe_means = {}
    for tname in targets:
        probe_means[tname] = {}
        for li in layer_idxs:
            vals = [s[tname][li] for s in all_seed_metrics
                    if li in s[tname] and not np.isnan(s[tname].get(li, float('nan')))]
            probe_means[tname][str(li)] = {
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'std':  float(np.std(vals))  if len(vals) > 1 else 0.0,
                'n':    len(vals),
            }
    return probe_means


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-data',  type=int, default=5000)
    parser.add_argument('--n-steps', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 2 (scaled): device={device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Generating {args.n_data} round2 inputs...")
    inputs, metadata = generate_metadata_dataset(n=args.n_data, seed=42)

    # ── Model registry ────────────────────────────────────────────────────────
    models_to_probe = []

    # constrained-LN (d=32, 4L, ReLU)
    for s in range(3):
        p = os.path.join(CKPT_BASE, f'constrained_ln_seed{s}', 'best_model.pt')
        if os.path.exists(p):
            models_to_probe.append(('constrained_ln', s, p, True))

    # scaled models (d=32, 64, 128)
    for d in [32, 64, 128]:
        for s in range(3):
            p = os.path.join(CKPT_BASE, f'scaled_d{d}_seed{s}', 'best_model.pt')
            if os.path.exists(p):
                models_to_probe.append((f'scaled_d{d}', s, p, False))

    # trained-d256: seed 0 is in round2_trained/, seeds 1-4 in checkpoints/
    p0 = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')
    if os.path.exists(p0):
        models_to_probe.append(('trained_d256', 0, p0, False))
    for s in range(1, 5):
        p = os.path.join(CKPT_BASE, f'seed{s}_final.pt')
        if os.path.exists(p):
            models_to_probe.append(('trained_d256', s, p, False))

    # ── Probe ────────────────────────────────────────────────────────────────
    per_model_seeds = {}   # model_key -> list of (seed, layer_metrics)

    for model_key, seed, ckpt_path, is_constrained in models_to_probe:
        print(f"\n  {model_key} seed={seed}: {ckpt_path}")
        try:
            lm, config = probe_one(ckpt_path, inputs, metadata, device,
                                   n_steps=args.n_steps, is_constrained=is_constrained)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        per_model_seeds.setdefault(model_key, []).append((seed, lm, config))

        for tname in TARGETS:
            best_l = max(lm[tname], key=lm[tname].get) if lm[tname] else '?'
            bv = lm[tname].get(best_l, float('nan'))
            print(f"    {tname}: best L{best_l} = {bv:.3f}")

    # ── Summarize ─────────────────────────────────────────────────────────────
    summary = {}
    for model_key, seed_entries in per_model_seeds.items():
        seeds = [e[0] for e in seed_entries]
        all_lm = [e[1] for e in seed_entries]
        config = seed_entries[0][2]
        probe_means = summarize_seeds(all_lm)

        summary[model_key] = {
            'd_model':    config.get('d_model', '?'),
            'n_layers':   config.get('n_layers', '?'),
            'n_seeds':    len(seeds),
            'seeds':      seeds,
            'probe_means': probe_means,
        }

        print(f"\n{model_key} (d={config.get('d_model')}, {config.get('n_layers')}L) summary:")
        for tname in TARGETS:
            b = best_metric(probe_means, tname)
            print(f"  {tname}: best = {b:.3f}")

    out = os.path.join(RESULTS_DIR, 'phase2_scaled_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
