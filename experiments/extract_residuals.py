#!/usr/bin/env python3
"""
Extract residual stream activations from both models at each layer.

For round2 (trained): hooks into each TransformerBlock output.
For round1 (constructed): hooks into each layer output.

Output: dict of {layer_idx: tensor of shape (N, seq_len, d_model)}
"""

import os
import sys
import random
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))


# ── Round 2 residual extraction ─────────────────────────────────────────────

def get_r2_residuals(model, inputs, device='cpu'):
    """
    Extract residual stream after each transformer layer for round2 model.

    Args:
        model: MiniSUBLEQTransformer
        inputs: LongTensor (N, SEQ_LEN)
        device: 'cuda' or 'cpu'

    Returns:
        dict: layer_idx (0=embed, 1..n_layers=after each block) ->
              tensor (N, SEQ_LEN, d_model)
        logits: (N, SEQ_LEN, VOCAB_SIZE)
    """
    model.eval()
    residuals = {}
    hooks = []

    # Hook into embedding output
    B, S = inputs.shape

    with torch.no_grad():
        # Manual forward pass with hooks
        inputs = inputs.to(device)
        tok = model.token_emb(inputs)
        pos = model.pos_emb(model.pos_indices[:, :S].expand(B, -1))
        typ = model.type_emb(model.type_indices[:, :S].expand(B, -1))
        h = tok + pos + typ  # no dropout in eval mode
        residuals[0] = h.cpu()  # embedding layer

        for i, layer in enumerate(model.layers):
            h = layer(h)
            residuals[i + 1] = h.cpu()

        h_norm = model.final_norm(h)
        logits = model.output_head(h_norm)

    return residuals, logits.cpu()


def get_r2_residuals_batched(model, inputs, device='cpu', batch_size=512):
    """Batched version for large datasets."""
    all_residuals = {}
    all_logits = []
    n = inputs.shape[0]

    for start in range(0, n, batch_size):
        batch = inputs[start:start + batch_size]
        res, logits = get_r2_residuals(model, batch, device)
        all_logits.append(logits)
        for layer_idx, r in res.items():
            if layer_idx not in all_residuals:
                all_residuals[layer_idx] = []
            all_residuals[layer_idx].append(r)

    return {k: torch.cat(v, dim=0) for k, v in all_residuals.items()}, \
           torch.cat(all_logits, dim=0)


# ── Round 1 residual extraction ─────────────────────────────────────────────

def get_r1_residuals(model, inputs, device='cpu'):
    """
    Extract residual stream after each layer for round1 constructed model.

    Args:
        model: HandCodedSUBLEQ
        inputs: LongTensor (N, 417) with token values (value + VALUE_OFFSET)

    Returns:
        dict: layer_idx -> tensor (N, 417, 32)
    """
    model.eval()
    residuals = {}

    with torch.no_grad():
        inputs = inputs.to(device)
        # The round1 model has a different forward structure
        # We need to trace through it manually
        # Get embedding
        h = model.embed(inputs)  # (N, 417, 32)
        residuals[0] = h.cpu()

        for i, (attn, ffn) in enumerate(zip(model.attns, model.ffns)):
            h = h + attn(h)
            h = h + ffn(h)
            residuals[i + 1] = h.cpu()

    return residuals


# ── Generate metadata dataset ────────────────────────────────────────────────

def generate_metadata_dataset(n=10000, seed=42):
    """
    Generate dataset of single SUBLEQ steps with full metadata.
    For the round2 model (32 cells, 8-bit).

    Returns:
        inputs: (N, 33) token tensors
        metadata: list of dicts with pc, a, b, c, mem_a, mem_b, new_val, branch_taken
    """
    random.seed(seed)
    torch.manual_seed(seed)

    from subleq import step, generate_random_state, generate_random_program
    from subleq import encode, MEM_SIZE, VALUE_MIN, VALUE_MAX
    from subleq.programs import generate_random_state

    inputs = []
    metadata = []

    attempts = 0
    while len(inputs) < n and attempts < n * 10:
        attempts += 1
        n_instr = random.randint(1, 8)
        mem, pc = generate_random_state(n_instr)

        # Get instruction fields
        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a_addr = mem[pc]
        b_addr = mem[pc + 1]
        c_addr = mem[pc + 2]

        if not (0 <= a_addr < MEM_SIZE and 0 <= b_addr < MEM_SIZE):
            continue

        mem_a = mem[a_addr]
        mem_b = mem[b_addr]
        new_val = mem_b - mem_a
        # Clamp to 8-bit
        from subleq.interpreter import clamp
        new_val_clamped = clamp(new_val)
        branch_taken = new_val_clamped <= 0

        new_mem, new_pc, halted = step(mem, pc)
        if halted:
            continue

        inp = encode(mem, pc)
        inputs.append(inp)
        metadata.append({
            'pc': pc,
            'a_addr': a_addr,
            'b_addr': b_addr,
            'c_addr': c_addr,
            'mem_a': mem_a,
            'mem_b': mem_b,
            'new_val': new_val_clamped,
            'delta': mem_b - mem_a,  # before clamping
            'branch_taken': int(branch_taken),
            'new_pc': new_pc,
            'memory': list(mem),
        })

    inputs = torch.stack(inputs[:n])
    metadata = metadata[:n]
    print(f"Generated {len(metadata)} metadata examples")
    return inputs, metadata


def load_r2_model(ckpt_path, device='cpu'):
    """Load a round2 model from checkpoint."""
    from subleq import MiniSUBLEQTransformer
    from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, config


if __name__ == '__main__':
    # Quick test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    ckpt = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, config = load_r2_model(ckpt, device)
    print(f"Model loaded: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    inputs, metadata = generate_metadata_dataset(n=1000)
    print(f"Generated {len(metadata)} examples")

    residuals, logits = get_r2_residuals_batched(model, inputs, device=device)
    print("Residual shapes:")
    for layer, r in residuals.items():
        print(f"  Layer {layer}: {r.shape}")
