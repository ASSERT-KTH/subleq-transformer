#!/usr/bin/env python3
"""
Phase 3 (scaled): Focused activation patching on all capacity-sweep models.

Runs seed 0 for each model variant (speed: ~5 min each).
d=256 seed 0 already exists in phase3_patching.pkl — reloaded here.

Output: results/phase3_scaled_summary.json
  {model_key: {ptype: {layer_L: {max, argmax, mean}}, peak_layer, peak_max}}
"""

import os, sys, json, pickle
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from extract_residuals import load_r2_model
from constrained_model import load_constrained_model
from phase3_patching import generate_contrast_pairs, activation_patch_r2

CKPT_BASE   = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')
PAIR_TYPES  = ['mem_a', 'mem_b', 'branch']


def run_patching(model, pairs_by_type, device, n_patch=200):
    effects = {}
    for ptype, pairs in pairs_by_type.items():
        n = min(n_patch, len(pairs))
        all_e = []
        for i in range(n):
            all_e.append(activation_patch_r2(model, pairs[i], device=device))
            if (i+1) % 50 == 0:
                print(f"    {ptype}: {i+1}/{n}")
        effects[ptype] = torch.stack(all_e).mean(dim=0).numpy()
    return effects


def summarize_effects(effects, n_layers):
    out = {}
    for ptype, arr in effects.items():
        et = torch.tensor(arr)
        ptype_summary = {}
        for l in range(n_layers + 1):
            ptype_summary[f'layer_{l}'] = {
                'max':    float(et[l].max()),
                'argmax': int(et[l].argmax()),
                'mean':   float(et[l].mean()),
            }
        # Peak layer for this pair type
        peak_l = max(range(n_layers+1), key=lambda l: ptype_summary[f'layer_{l}']['max'])
        ptype_summary['peak_layer'] = peak_l
        ptype_summary['peak_max']   = ptype_summary[f'layer_{peak_l}']['max']
        out[ptype] = ptype_summary
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pairs',       type=int, default=500)
    parser.add_argument('--n-patch',       type=int, default=200)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 3 (scaled): Activation patching | device={device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nGenerating contrast pairs ({args.n_pairs}/type)...")
    pairs_by_type = {pt: generate_contrast_pairs(pt, n=args.n_pairs, seed=42)
                     for pt in PAIR_TYPES}

    summary = {}

    # ── Model list (seed 0 only) ──────────────────────────────────────────────
    models_to_patch = [
        ('constrained_ln',  os.path.join(CKPT_BASE, 'constrained_ln_seed0', 'best_model.pt'),  True),
        ('scaled_d32',      os.path.join(CKPT_BASE, 'scaled_d32_seed0',     'best_model.pt'),  False),
        ('scaled_d64',      os.path.join(CKPT_BASE, 'scaled_d64_seed0',     'best_model.pt'),  False),
        ('scaled_d128',     os.path.join(CKPT_BASE, 'scaled_d128_seed0',    'best_model.pt'),  False),
    ]

    for model_key, ckpt_path, is_constrained in models_to_patch:
        if not os.path.exists(ckpt_path):
            print(f"  MISSING: {ckpt_path} — skipping {model_key}")
            continue

        print(f"\n=== {model_key}: {ckpt_path} ===")
        if is_constrained:
            model, config = load_constrained_model(ckpt_path, device)
        else:
            model, config = load_r2_model(ckpt_path, device)
        n_layers = config.get('n_layers', 6)
        print(f"  d={config.get('d_model')}, n_layers={n_layers}")

        effects = run_patching(model, pairs_by_type, device, n_patch=args.n_patch)
        model_summary = summarize_effects(effects, n_layers)
        model_summary['_meta'] = {
            'ckpt': ckpt_path, 'd_model': config.get('d_model'),
            'n_layers': n_layers, 'seed': 0,
        }
        summary[model_key] = model_summary

        for ptype in PAIR_TYPES:
            pl = model_summary[ptype]['peak_layer']
            pm = model_summary[ptype]['peak_max']
            pa = model_summary[ptype][f'layer_{pl}']['argmax']
            print(f"  {ptype}: peak L{pl} pos{pa} = {pm:.4f}")

    # ── Reuse existing d=256 seed 0 from phase3_patching.pkl ──────────────────
    pkl_path = os.path.join(RESULTS_DIR, 'phase3_patching.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            d256_data = pickle.load(f)
        seed0_res = d256_data.get('results', {}).get(0)
        if seed0_res:
            n_layers = seed0_res.get('config', {}).get('n_layers', 6)
            effects256 = seed0_res.get('effects', {})
            s256 = summarize_effects(effects256, n_layers)
            s256['_meta'] = {
                'ckpt': 'round2_trained/checkpoints/best_model.pt',
                'd_model': seed0_res.get('config', {}).get('d_model', 256),
                'n_layers': n_layers, 'seed': 0,
            }
            summary['trained_d256'] = s256
            print(f"\ntrained_d256 (from existing pkl):")
            for pt in PAIR_TYPES:
                pl = s256[pt]['peak_layer']
                print(f"  {pt}: peak L{pl} = {s256[pt]['peak_max']:.4f}")

    out = os.path.join(RESULTS_DIR, 'phase3_scaled_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
