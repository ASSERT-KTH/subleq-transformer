#!/usr/bin/env python3
"""
Phase 3: Activation patching for causal circuit comparison.

Builds contrast pairs (differ in exactly one causally relevant way),
runs activation patching at every (layer, position), produces heatmaps.
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

        # Generate base state
        n_instr = random.randint(1, 8)
        mem_A, pc = generate_random_state(n_instr)

        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a_addr = mem_A[pc]
        b_addr = mem_A[pc + 1]
        c_addr = mem_A[pc + 2]

        if not (0 <= a_addr < MEM_SIZE and 0 <= b_addr < MEM_SIZE):
            continue

        # Get step A
        new_mem_A, new_pc_A, halted_A = step(mem_A, pc)
        if halted_A:
            continue

        mem_B = list(mem_A)  # Start as copy

        if pair_type == 'mem_a':
            # Change mem[a] to a different value
            orig_val = mem_A[a_addr]
            new_val = random.randint(VALUE_MIN, VALUE_MAX)
            while new_val == orig_val:
                new_val = random.randint(VALUE_MIN, VALUE_MAX)
            mem_B[a_addr] = new_val

        elif pair_type == 'mem_b':
            # Change mem[b] to a different value
            orig_val = mem_A[b_addr]
            new_val = random.randint(VALUE_MIN, VALUE_MAX)
            while new_val == orig_val:
                new_val = random.randint(VALUE_MIN, VALUE_MAX)
            mem_B[b_addr] = new_val

        elif pair_type == 'branch':
            # Choose mem[b] to flip branch direction
            mem_a_val = mem_A[a_addr]
            mem_b_val = mem_A[b_addr]
            new_val_A = clamp(mem_b_val - mem_a_val)
            branch_A = new_val_A <= 0

            # Find mem_b such that branch flips
            # branch_B = (mem_b - mem_a <= 0) = !branch_A
            # If branch_A=True (took branch), we need new_val > 0
            # so mem_b > mem_a, i.e., mem_b >= mem_a + 1
            if branch_A:
                # Make branch NOT taken: need mem_b - mem_a > 0
                # So mem_b > mem_a
                new_mb = random.randint(max(VALUE_MIN, mem_a_val + 1),
                                        min(VALUE_MAX, mem_a_val + 50))
            else:
                # Make branch TAKEN: need mem_b - mem_a <= 0
                # So mem_b <= mem_a
                new_mb = random.randint(max(VALUE_MIN, mem_a_val - 50),
                                        min(VALUE_MAX, mem_a_val))
            if new_mb == mem_b_val:
                continue
            mem_B[b_addr] = new_mb
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")

        # Get step B
        new_mem_B, new_pc_B, halted_B = step(mem_B, pc)
        if halted_B:
            continue

        # Only keep pairs where outputs differ (otherwise uninformative)
        if new_mem_A == new_mem_B and new_pc_A == new_pc_B:
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


def activation_patch_r2(model, inp_A, inp_B, device='cpu'):
    """
    Run activation patching on round2 model.

    For each (layer, position), patch activations from input_A into
    the forward pass of input_B, and measure effect on output.

    Returns:
        effects: (n_layers+1, seq_len) tensor of patching effects
    """
    model.eval()
    inp_A = inp_A.unsqueeze(0).to(device)  # (1, seq_len)
    inp_B = inp_B.unsqueeze(0).to(device)

    seq_len = inp_A.shape[1]
    n_layers = len(model.layers)

    # Cache all residuals from input_A
    with torch.no_grad():
        cache_A = {}
        B_size, S = inp_A.shape
        tok = model.token_emb(inp_A)
        pos = model.pos_emb(model.pos_indices[:, :S].expand(B_size, -1))
        typ = model.type_emb(model.type_indices[:, :S].expand(B_size, -1))
        h = tok + pos + typ
        cache_A[0] = h.clone()  # embedding

        for i, layer in enumerate(model.layers):
            h = layer(h)
            cache_A[i + 1] = h.clone()

        # Get unpatched output for B
        tok_B = model.token_emb(inp_B)
        pos_B = model.pos_emb(model.pos_indices[:, :S].expand(B_size, -1))
        typ_B = model.type_emb(model.type_indices[:, :S].expand(B_size, -1))
        h_B_orig = tok_B + pos_B + typ_B
        for layer in model.layers:
            h_B_orig = layer(h_B_orig)
        h_B_norm = model.final_norm(h_B_orig)
        logits_B_unpatched = model.output_head(h_B_norm)  # (1, seq_len, vocab)

        # Get unpatched output for A (target)
        h_A_final = cache_A[n_layers]
        h_A_norm = model.final_norm(h_A_final)
        logits_A = model.output_head(h_A_norm)

    # For each (layer, position), patch and measure effect
    effects = torch.zeros(n_layers + 1, seq_len)

    # Compute total logit distance between unpatched A and B outputs
    # Use KL divergence or L2 on logits at the changed positions
    # For simplicity: L1 distance in logits summed across all positions
    with torch.no_grad():
        logit_dist_total = (logits_A - logits_B_unpatched).abs().sum().item()
        if logit_dist_total < 1e-6:
            return effects  # No difference, uninformative pair

    for patch_layer in range(n_layers + 1):
        for patch_pos in range(seq_len):
            with torch.no_grad():
                # Rerun B's forward with patch at (patch_layer, patch_pos)
                tok_B = model.token_emb(inp_B)
                pos_B = model.pos_emb(model.pos_indices[:, :S].expand(1, -1))
                typ_B = model.type_emb(model.type_indices[:, :S].expand(1, -1))
                h = tok_B + pos_B + typ_B

                if patch_layer == 0:
                    # Patch at embedding layer
                    h[:, patch_pos, :] = cache_A[0][:, patch_pos, :]

                for i, layer in enumerate(model.layers):
                    h = layer(h)
                    if patch_layer == i + 1:
                        # Patch at this layer
                        h[:, patch_pos, :] = cache_A[i + 1][:, patch_pos, :]

                h_norm = model.final_norm(h)
                logits_patched = model.output_head(h_norm)

                # Measure effect: how much does patching shift logits toward A?
                shift_toward_A = (logits_A - logits_patched).abs().sum().item()
                # Effect = (distance to B decreases) / total distance
                # Or: logit shift = |logits_A - logits_B_unpatched| - |logits_A - logits_patched|
                # Normalized by total distance
                shift = (logit_dist_total - shift_toward_A) / (logit_dist_total + 1e-8)
                effects[patch_layer, patch_pos] = shift

    return effects


def run_patching_experiment(model, pairs, device='cpu', n_pairs=200):
    """Run patching on a sample of contrast pairs."""
    n = min(n_pairs, len(pairs))
    all_effects = []

    for i in range(n):
        pair = pairs[i]
        inp_A = encode(pair['mem_A'], pair['pc'])
        inp_B = encode(pair['mem_B'], pair['pc'])

        effects = activation_patch_r2(model, inp_A, inp_B, device=device)
        all_effects.append(effects)

        if (i + 1) % 50 == 0:
            print(f"    Patched {i+1}/{n} pairs")

    # Average effects
    mean_effects = torch.stack(all_effects).mean(dim=0)
    return mean_effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-pairs', type=int, default=500,
                        help='Number of contrast pairs per type')
    parser.add_argument('--n-patch-per-type', type=int, default=100,
                        help='Number of pairs to patch per type (subset for speed)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 3: Activation Patching")
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

    # Generate contrast pairs
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
            pairs = all_pairs[ptype]
            mean_effects = run_patching_experiment(model, pairs, device=device,
                                                   n_pairs=args.n_patch_per_type)
            seed_effects[ptype] = mean_effects.numpy()

            # Print heatmap summary
            print(f"  Effect heatmap (max per layer):")
            for l in range(n_layers + 1):
                row_max = mean_effects[l].max().item()
                row_argmax = mean_effects[l].argmax().item()
                print(f"    L{l}: max={row_max:.3f} at pos {row_argmax}")

        all_results[seed_id] = {
            'ckpt': ckpt_path,
            'config': config,
            'effects': seed_effects,
        }

    # Save
    out_path = os.path.join(args.output_dir, 'phase3_patching.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'results': all_results, 'pair_types': pair_types}, f)
    print(f"\nResults saved to {out_path}")

    # JSON summary: max patching effect per layer per type
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
