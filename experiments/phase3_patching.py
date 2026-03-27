#!/usr/bin/env python3
"""
Phase 3: Activation patching for causal circuit comparison.

Builds contrast pairs (differ in exactly one causally relevant way),
runs activation patching at every (layer, position), produces heatmaps.

Metric: for each contrast pair, focus only on output positions that
differ between A and B (PC bytes + mem[b] bytes). At those positions
measure the logit of A's correct token:

  effect(patch_layer, patch_pos) =
      mean over changed positions p of:
          clip01( (logit_patched[p, token_A[p]] - logit_B[p, token_A[p]])
                / (logit_A[p, token_A[p]] - logit_B[p, token_A[p]]) )

This equals 0 when patching has no effect and 1 when it fully restores A's output.
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from subleq import step, generate_random_state, encode, decode
from subleq.interpreter import MEM_SIZE, VALUE_MIN, VALUE_MAX, clamp
from subleq.tokenizer import get_changed_positions
from extract_residuals import load_r2_model, get_r2_residuals


def generate_contrast_pairs(pair_type='mem_a', n=1000, seed=42):
    """
    Generate contrast pairs differing in exactly one causally relevant way.

    pair_type:
      'mem_a' - same PC, same mem[b], different mem[a]
      'mem_b' - same PC, same mem[a], different mem[b]
      'branch' - same PC, same mem[a], mem[b] chosen to flip branch direction
    """
    random.seed(seed)
    pairs = []
    attempts = 0

    while len(pairs) < n and attempts < n * 50:
        attempts += 1

        n_instr = random.randint(1, 8)
        mem_A, pc = generate_random_state(n_instr)

        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a_addr = mem_A[pc]
        b_addr = mem_A[pc + 1]
        c_addr = mem_A[pc + 2]

        if not (0 <= a_addr < MEM_SIZE and 0 <= b_addr < MEM_SIZE):
            continue

        new_mem_A, new_pc_A, halted_A = step(mem_A, pc)
        if halted_A:
            continue

        mem_B = list(mem_A)

        if pair_type == 'mem_a':
            orig_val = mem_A[a_addr]
            new_val = random.randint(VALUE_MIN, VALUE_MAX)
            while new_val == orig_val:
                new_val = random.randint(VALUE_MIN, VALUE_MAX)
            mem_B[a_addr] = new_val

        elif pair_type == 'mem_b':
            orig_val = mem_A[b_addr]
            new_val = random.randint(VALUE_MIN, VALUE_MAX)
            while new_val == orig_val:
                new_val = random.randint(VALUE_MIN, VALUE_MAX)
            mem_B[b_addr] = new_val

        elif pair_type == 'branch':
            mem_a_val = mem_A[a_addr]
            mem_b_val = mem_A[b_addr]
            new_val_A = clamp(mem_b_val - mem_a_val)
            branch_A = new_val_A <= 0

            if branch_A:
                new_mb = random.randint(max(VALUE_MIN, mem_a_val + 1),
                                        min(VALUE_MAX, mem_a_val + 50))
            else:
                new_mb = random.randint(max(VALUE_MIN, mem_a_val - 50),
                                        min(VALUE_MAX, mem_a_val))
            if new_mb == mem_b_val:
                continue
            mem_B[b_addr] = new_mb
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")

        new_mem_B, new_pc_B, halted_B = step(mem_B, pc)
        if halted_B:
            continue

        # Keep pairs where outputs differ
        out_A = encode(new_mem_A, new_pc_A)
        out_B = encode(new_mem_B, new_pc_B)
        if (out_A == out_B).all():
            continue

        pairs.append({
            'mem_A': list(mem_A),
            'mem_B': list(mem_B),
            'pc': pc,
            'a_addr': a_addr,
            'b_addr': b_addr,
            'c_addr': c_addr,
            'new_mem_A': list(new_mem_A),
            'new_mem_B': list(new_mem_B),
            'new_pc_A': new_pc_A,
            'new_pc_B': new_pc_B,
            'pair_type': pair_type,
        })

    print(f"Generated {len(pairs)}/{n} {pair_type} contrast pairs ({attempts} attempts)")
    return pairs


def activation_patch_r2(model, pair, device='cpu'):
    """
    Focused activation patching on round2 model.

    For each (layer, position), patch activations from A into B's forward pass
    and measure how much logit of A's correct tokens improves at changed positions.

    Returns:
        effects: (n_layers+1, seq_len) tensor, each entry in [0, 1]
    """
    model.eval()

    mem_A = pair['mem_A']
    mem_B = pair['mem_B']
    pc = pair['pc']
    new_mem_A = pair['new_mem_A']
    new_pc_A = pair['new_pc_A']
    new_mem_B = pair['new_mem_B']
    new_pc_B = pair['new_pc_B']

    inp_A = encode(mem_A, pc).unsqueeze(0).to(device)
    inp_B = encode(mem_B, pc).unsqueeze(0).to(device)
    out_A = encode(new_mem_A, new_pc_A).to(device)  # (seq_len,) ground-truth tokens for A
    out_B = encode(new_mem_B, new_pc_B).to(device)

    seq_len = inp_A.shape[1]
    n_layers = len(model.layers)
    S = seq_len

    # Positions that change in one step (same for A and B since they share pc, b_addr)
    chg_candidates = get_changed_positions(mem_A, pc)
    # Keep only positions where A's and B's outputs actually differ
    chg_pos = [p for p in chg_candidates if out_A[p].item() != out_B[p].item()]

    if not chg_pos:
        return torch.zeros(n_layers + 1, seq_len)

    # Cache A's residuals
    with torch.no_grad():
        cache_A = {}
        B_size = 1
        tok = model.token_emb(inp_A)
        pos = model.pos_emb(model.pos_indices[:, :S].expand(B_size, -1))
        typ = model.type_emb(model.type_indices[:, :S].expand(B_size, -1))
        h = tok + pos + typ
        cache_A[0] = h.clone()
        for i, layer in enumerate(model.layers):
            h = layer(h)
            cache_A[i + 1] = h.clone()
        logits_A = model.output_head(model.final_norm(cache_A[n_layers]))  # (1, S, V)

        # B baseline logits
        tok_B = model.token_emb(inp_B)
        pos_B = model.pos_emb(model.pos_indices[:, :S].expand(B_size, -1))
        typ_B = model.type_emb(model.type_indices[:, :S].expand(B_size, -1))
        h_B = tok_B + pos_B + typ_B
        for layer in model.layers:
            h_B = layer(h_B)
        logits_B = model.output_head(model.final_norm(h_B))  # (1, S, V)

    # Precompute denominators: logit_A[p, token_A_p] - logit_B[p, token_A_p]
    denom_baseline = {}  # p -> (logit_B_p, gap)
    valid_pos = []
    for p in chg_pos:
        t = out_A[p].item()
        lA = logits_A[0, p, t].item()
        lB = logits_B[0, p, t].item()
        gap = lA - lB
        if abs(gap) > 0.01:
            denom_baseline[p] = (lB, gap)
            valid_pos.append(p)

    if not valid_pos:
        return torch.zeros(n_layers + 1, seq_len)

    effects = torch.zeros(n_layers + 1, seq_len)

    for patch_layer in range(n_layers + 1):
        for patch_pos in range(seq_len):
            with torch.no_grad():
                tok_B = model.token_emb(inp_B)
                pos_B = model.pos_emb(model.pos_indices[:, :S].expand(1, -1))
                typ_B = model.type_emb(model.type_indices[:, :S].expand(1, -1))
                h = tok_B + pos_B + typ_B

                if patch_layer == 0:
                    h[:, patch_pos, :] = cache_A[0][:, patch_pos, :]

                for i, layer in enumerate(model.layers):
                    h = layer(h)
                    if patch_layer == i + 1:
                        h[:, patch_pos, :] = cache_A[i + 1][:, patch_pos, :]

                logits_patched = model.output_head(model.final_norm(h))  # (1, S, V)

                # Compute focused effect
                pos_effects = []
                for p in valid_pos:
                    t = out_A[p].item()
                    lB, gap = denom_baseline[p]
                    lP = logits_patched[0, p, t].item()
                    eff = (lP - lB) / gap
                    pos_effects.append(max(0.0, min(1.0, eff)))

                effects[patch_layer, patch_pos] = float(np.mean(pos_effects))

    return effects


def run_patching_experiment(model, pairs, device='cpu', n_pairs=200):
    """Run patching on a sample of contrast pairs."""
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
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-pairs', type=int, default=500,
                        help='Number of contrast pairs per type')
    parser.add_argument('--n-patch-per-type', type=int, default=200,
                        help='Number of pairs to patch per type')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 3: Activation Patching (focused metric)")
    print(f"Device: {device}")

    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'results')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(script_dir, 'checkpoints')

    # Find checkpoints
    if args.ckpt is not None:
        ckpt_paths = [(args.seed, args.ckpt)]
    else:
        ckpt_paths = []
        seed0_ckpt = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')
        if os.path.exists(seed0_ckpt):
            ckpt_paths.append((0, seed0_ckpt))
        for seed in range(1, 5):
            cp = os.path.join(args.ckpt_dir, f'seed{seed}_final.pt')
            if os.path.exists(cp):
                ckpt_paths.append((seed, cp))

    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"\nGenerating contrast pairs ({args.n_pairs} per type)...")
    pair_types = ['mem_a', 'mem_b', 'branch']
    all_pairs = {}
    for ptype in pair_types:
        all_pairs[ptype] = generate_contrast_pairs(ptype, n=args.n_pairs, seed=args.seed * 100)

    all_results = {}
    for seed_id, ckpt_path in ckpt_paths:
        print(f"\n=== Patching seed {seed_id} ===")
        model, config = load_r2_model(ckpt_path, device)
        n_layers = config.get('n_layers', 6)

        seed_effects = {}
        for ptype in pair_types:
            print(f"  Pair type: {ptype}")
            mean_effects = run_patching_experiment(model, all_pairs[ptype], device=device,
                                                   n_pairs=args.n_patch_per_type)
            seed_effects[ptype] = mean_effects.numpy()

            print(f"  Effect heatmap (max per layer):")
            for l in range(n_layers + 1):
                row_max = mean_effects[l].max().item()
                row_argmax = mean_effects[l].argmax().item()
                print(f"    L{l}: max={row_max:.4f} at pos {row_argmax}")

        all_results[seed_id] = {
            'ckpt': ckpt_path,
            'config': config,
            'effects': seed_effects,
        }

    out_path = os.path.join(args.output_dir, 'phase3_patching.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'results': all_results, 'pair_types': pair_types}, f)
    print(f"\nResults saved to {out_path}")

    json_summary = {}
    for seed_id, seed_data in all_results.items():
        json_summary[str(seed_id)] = {}
        for ptype, effects in seed_data['effects'].items():
            effects_t = torch.tensor(effects)
            json_summary[str(seed_id)][ptype] = {
                f'layer_{l}': {
                    'max': float(effects_t[l].max()),
                    'argmax': int(effects_t[l].argmax()),
                    'mean': float(effects_t[l].mean()),
                }
                for l in range(effects_t.shape[0])
            }
    with open(os.path.join(args.output_dir, 'phase3_summary.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)
    print("JSON summary saved.")


if __name__ == '__main__':
    main()
