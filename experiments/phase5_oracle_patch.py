#!/usr/bin/env python3
"""
Phase 5: Activation patching on the oracle (round1) model.

Mirrors the phase3 patching procedure on the oracle model
so that patching heatmaps can be compared side-by-side.

Oracle architecture: 4 layers, 8 heads, d_model=32, SEQ_LEN=417.
Token encoding: value + 32768 (VALUE_OFFSET).

Note: The oracle forward pass produces hard one-hot logits by reading DV
(dimension 0) at the final layer. For patching we compare the final DV
values (continuous before rounding) so that fractional shifts are visible.
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))
sys.path.insert(0, script_dir)

from interpreter import step, MEM_SIZE, VALUE_MIN, VALUE_MAX, VALUE_OFFSET, clamp
from programs import make_random_program
from model import HandCodedSUBLEQ, DV


def generate_r1_contrast_pairs(pair_type='mem_a', n=500, seed=42):
    """
    Generate contrast pairs for the oracle (round1) format.

    pair_type:
      'mem_a'  - same PC/mem[b], different mem[a]
      'mem_b'  - same PC/mem[a], different mem[b]
      'branch' - same PC/mem[a], mem[b] chosen to flip branch direction
    """
    random.seed(seed)
    pairs = []
    attempts = 0

    while len(pairs) < n and attempts < n * 50:
        attempts += 1

        n_instr = random.randint(1, 20)
        mem_A, pc = make_random_program(n_instr)

        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a_addr = mem_A[pc]
        b_addr = mem_A[pc + 1]

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
                                        min(VALUE_MAX, mem_a_val + 500))
            else:
                new_mb = random.randint(max(VALUE_MIN, mem_a_val - 500),
                                        min(VALUE_MAX, mem_a_val))
            if new_mb == mem_b_val:
                continue
            mem_B[b_addr] = new_mb
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")

        new_mem_B, new_pc_B, halted_B = step(mem_B, pc)
        if halted_B:
            continue

        if new_mem_A == new_mem_B and new_pc_A == new_pc_B:
            continue

        pairs.append({
            'mem_A': list(mem_A),
            'mem_B': list(mem_B),
            'pc': pc,
            'a_addr': a_addr,
            'b_addr': b_addr,
            'new_mem_A': list(new_mem_A),
            'new_mem_B': list(new_mem_B),
            'new_pc_A': new_pc_A,
            'new_pc_B': new_pc_B,
            'pair_type': pair_type,
        })

    print(f"Generated {len(pairs)}/{n} {pair_type} contrast pairs ({attempts} attempts)")
    return pairs


def encode_r1(mem, pc):
    """Encode oracle (round1) state to token tensor."""
    tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
    return torch.tensor(tokens, dtype=torch.long)


def _oracle_forward_get_dv(model, tokens, device):
    """Run oracle forward, return final hidden state DV dimension (B, T)."""
    tokens = tokens.to(device)
    pos_ids = torch.arange(tokens.shape[1], device=device)
    h = model.tok_emb(tokens) + model.pos_emb(pos_ids)
    for layer in model.layers:
        h = layer(h)
    return h, h[:, :, DV].float()  # (B, T, d_model), (B, T)


def _oracle_forward_with_cache(model, tokens, device):
    """Run oracle forward, cache residuals at each layer."""
    tokens = tokens.to(device)
    pos_ids = torch.arange(tokens.shape[1], device=device)
    cache = {}
    h = model.tok_emb(tokens) + model.pos_emb(pos_ids)
    cache[0] = h.clone()
    for i, layer in enumerate(model.layers):
        h = layer(h)
        cache[i + 1] = h.clone()
    return cache, h[:, :, DV].float()


def activation_patch_r1(model, inp_A, inp_B, device='cpu'):
    """
    Run activation patching on the oracle (round1) model.

    Measures how much patching each (layer, position) shifts
    the final DV output from B toward A.

    Returns:
        effects: (n_layers+1, seq_len) tensor
    """
    model.eval()
    inp_A = inp_A.unsqueeze(0).to(device)
    inp_B = inp_B.unsqueeze(0).to(device)

    seq_len = inp_A.shape[1]    # 417
    n_layers = len(model.layers)  # 4

    with torch.no_grad():
        cache_A, dv_A = _oracle_forward_with_cache(model, inp_A, device)
        _, dv_B_unpatched = _oracle_forward_get_dv(model, inp_B, device)

        dist_total = (dv_A - dv_B_unpatched).abs().sum().item()
        if dist_total < 1e-6:
            return torch.zeros(n_layers + 1, seq_len)

    effects = torch.zeros(n_layers + 1, seq_len)

    for patch_layer in range(n_layers + 1):
        for patch_pos in range(seq_len):
            with torch.no_grad():
                pos_ids = torch.arange(seq_len, device=device)
                h = model.tok_emb(inp_B) + model.pos_emb(pos_ids)

                if patch_layer == 0:
                    h[:, patch_pos, :] = cache_A[0][:, patch_pos, :]

                for i, layer in enumerate(model.layers):
                    h = layer(h)
                    if patch_layer == i + 1:
                        h[:, patch_pos, :] = cache_A[i + 1][:, patch_pos, :]

                dv_patched = h[:, :, DV].float()
                dist_after = (dv_A - dv_patched).abs().sum().item()
                effects[patch_layer, patch_pos] = (
                    (dist_total - dist_after) / (dist_total + 1e-8)
                )

    return effects


def run_oracle_patching(model, pairs, device='cpu', n_pairs=100):
    """Patch a sample of contrast pairs and return mean effects."""
    n = min(n_pairs, len(pairs))
    all_effects = []

    for i in range(n):
        pair = pairs[i]
        inp_A = encode_r1(pair['mem_A'], pair['pc'])
        inp_B = encode_r1(pair['mem_B'], pair['pc'])

        effects = activation_patch_r1(model, inp_A, inp_B, device=device)
        all_effects.append(effects)

        if (i + 1) % 20 == 0:
            print(f"    Patched {i+1}/{n} pairs")

    return torch.stack(all_effects).mean(dim=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 5: Oracle Activation Patching")
    print(f"Device: {device}")

    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("\nLoading oracle model...")
    model = HandCodedSUBLEQ()
    model.to(device)
    model.eval()
    print(f"  Parameters: {model.count_params():,}")

    pair_types = ['mem_a', 'mem_b', 'branch']

    print("\nGenerating contrast pairs...")
    all_pairs = {}
    for ptype in pair_types:
        all_pairs[ptype] = generate_r1_contrast_pairs(ptype, n=500, seed=0)

    print("\nRunning activation patching (oracle)...")
    oracle_effects = {}
    for ptype in pair_types:
        print(f"  Pair type: {ptype}")
        mean_effects = run_oracle_patching(model, all_pairs[ptype], device=device, n_pairs=100)
        oracle_effects[ptype] = mean_effects.numpy()

        print(f"  Effect heatmap (max per layer):")
        for l in range(mean_effects.shape[0]):
            row_max = mean_effects[l].max().item()
            row_argmax = mean_effects[l].argmax().item()
            print(f"    L{l}: max={row_max:.3f} at pos {row_argmax}")

    # Save
    out_pkl = os.path.join(results_dir, 'phase5_oracle_patch.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'oracle_effects': oracle_effects, 'pair_types': pair_types}, f)
    print(f"\nResults saved to {out_pkl}")

    # JSON summary
    json_summary = {}
    for ptype, effects in oracle_effects.items():
        effects_t = torch.tensor(effects)
        json_summary[ptype] = {
            f'layer_{l}': {
                'max': float(effects_t[l].max()),
                'argmax': int(effects_t[l].argmax()),
                'mean': float(effects_t[l].mean()),
            }
            for l in range(effects_t.shape[0])
        }
    with open(os.path.join(results_dir, 'phase5_summary.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)
    print("JSON summary saved.")


if __name__ == '__main__':
    main()
