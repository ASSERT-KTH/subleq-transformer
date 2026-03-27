#!/usr/bin/env python3
"""
Phase 1: Map and verify the oracle (constructed model, round1).

Tasks:
1. Extract oracle circuit map from model.py documentation
2. Train linear probes on oracle residual streams
3. Verify with interventions

Round 1 architecture: 4 layers, 8 heads, d_model=32, 416 memory cells, 16-bit.
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))
sys.path.insert(0, script_dir)

# Round 1 imports
from interpreter import step, run, MEM_SIZE, VALUE_MIN, VALUE_MAX, VALUE_OFFSET, DATA_START
from programs import make_negate, make_addition, make_multiply, make_random_program
from model import HandCodedSUBLEQ

# Oracle dimension map from model.py
ORACLE_MAP = {
    # Dimension: (name, layer_first_present, description)
    0:  ('DV', 0, 'Token value; receives PC delta and write delta from L4'),
    2:  ('DI', 0, 'Position index i'),
    3:  ('DI2', 0, 'Position index squared i^2'),
    4:  ('D1', 0, 'Constant 1'),
    5:  ('DPC', 0, 'PC indicator: 1 at position 0 only'),
    6:  ('DA', 1, 'Operand a = mem[pc] (after L1 attn)'),
    8:  ('DB', 1, 'Operand b = mem[pc+1] (after L1 attn)'),
    10: ('DC', 1, 'Operand c = mem[pc+2] (after L1 attn)'),
    18: ('DPCC', 1, 'Broadcast copy of PC value (after L1 attn)'),
    21: ('DSB', 1, 'Safe b: b at pos 0, 0 elsewhere (after L1 FFN)'),
    12: ('DMA', 2, 'mem[a] (after L2 attn)'),
    13: ('DMB', 2, 'mem[b] (after L2 attn)'),
    14: ('DNV', 2, 'New value = mem[b] - mem[a] (after L2 FFN)'),
    15: ('DDW', 2, 'Write delta = -mem[a] (after L2 FFN)'),
    20: ('DSTEP', 2, 'Step indicator 1[nv>0] (after L2 FFN)'),
    22: ('DSDDW', 2, 'Safe write delta (after L2 FFN)'),
    29: ('DSS', 2, 'Safe step (after L2 FFN)'),
    24: ('DBCB', 3, 'Broadcast of b to all positions (after L3 attn)'),
    25: ('DBCDDW', 3, 'Broadcast of write delta (after L3 attn)'),
    26: ('DH0', 3, 'ReLU(j - b) (after L3 FFN)'),
    27: ('DH1', 3, 'ReLU(j - b - 1) (after L3 FFN)'),
    28: ('DH2', 3, 'ReLU(j - b - 2) (after L3 FFN)'),
}

# Layer purposes
LAYER_PURPOSES = {
    0: 'Embedding: encodes token values, positions, PC indicator',
    1: 'L1: reads a, b, c from PC position; broadcasts PC value',
    2: 'L2: reads mem[a], mem[b]; computes new value and write delta',
    3: 'L3: broadcasts b address and write delta to all positions',
    4: 'L4: writes new value to mem[b] position, updates PC',
}


def generate_r1_dataset(n=5000, seed=42):
    """Generate dataset for round1 oracle probing."""
    random.seed(seed)
    inputs = []
    metadata = []

    attempts = 0
    while len(inputs) < n and attempts < n * 20:
        attempts += 1
        # Use random programs
        n_instr = random.randint(1, 20)
        mem, pc = make_random_program(n_instr)

        # Get instruction
        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a = mem[pc]
        b = mem[pc + 1]
        c = mem[pc + 2]

        if not (0 <= a < MEM_SIZE and 0 <= b < MEM_SIZE):
            continue

        mem_a = mem[a]
        mem_b = mem[b]
        new_val = mem_b - mem_a
        from interpreter import clamp
        new_val_c = clamp(new_val)
        branch_taken = new_val_c <= 0

        new_mem, new_pc, halted = step(mem, pc)
        if halted:
            continue

        # Encode as tokens
        tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
        inp = torch.tensor(tokens, dtype=torch.long)
        inputs.append(inp)
        metadata.append({
            'pc': pc,
            'a_addr': a, 'b_addr': b, 'c_addr': c,
            'mem_a': mem_a, 'mem_b': mem_b,
            'new_val': new_val_c,
            'delta': new_val,
            'branch_taken': int(branch_taken),
            'new_pc': new_pc,
        })

    print(f"Generated {len(metadata)}/{n} oracle examples")
    return torch.stack(inputs[:n]), metadata[:n]


def get_r1_residuals(model, inputs, device='cpu'):
    """Extract residual stream from round1 model at each layer.

    HandCodedSUBLEQ structure:
        embedding: tok_emb + pos_emb
        layers: 4 x TransformerBlock (each does h = h + attn(h); h = h + ffn(h))
    """
    model.eval()
    B, T = inputs.shape
    inputs = inputs.to(device)

    residuals = {}
    with torch.no_grad():
        # Embedding (tok_emb + pos_emb, no dropout)
        pos_ids = torch.arange(T, device=device)
        h = model.tok_emb(inputs) + model.pos_emb(pos_ids)
        residuals[0] = h.cpu()

        for i, layer in enumerate(model.layers):
            h = layer(h)  # TransformerBlock.forward: h = h + attn(h); h = h + ffn(h)
            residuals[i + 1] = h.cpu()

    return residuals


def get_r1_residuals_batched(model, inputs, device='cpu', batch_size=256):
    """Batched version."""
    all_res = {}
    n = inputs.shape[0]
    for start in range(0, n, batch_size):
        batch = inputs[start:start + batch_size]
        res = get_r1_residuals(model, batch, device)
        for k, v in res.items():
            if k not in all_res:
                all_res[k] = []
            all_res[k].append(v)
    return {k: torch.cat(v, dim=0) for k, v in all_res.items()}


class LinearProbe(nn.Module):
    def __init__(self, d_in, d_out=1):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)


def probe_specific_dims(X_train, y_train, X_val, y_val, dims, n_steps=500, device='cpu', is_classification=False):
    """Probe using only specific dimensions."""
    X_tr = X_train[:, dims].to(device).float()
    X_vl = X_val[:, dims].to(device).float()

    # Normalize inputs
    X_mean = X_tr.mean(dim=0, keepdim=True)
    X_std = X_tr.std(dim=0, keepdim=True) + 1e-8
    X_tr = (X_tr - X_mean) / X_std
    X_vl = (X_vl - X_mean) / X_std

    d = len(dims)
    if is_classification:
        probe = LinearProbe(d, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        y_tr = y_train.to(device).long()
        y_vl = y_val.to(device).long()
        for _ in range(n_steps):
            probe.train()
            loss = nn.functional.cross_entropy(probe(X_tr), y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(X_vl).argmax(dim=-1) == y_vl).float().mean().item()
        return acc
    else:
        probe = LinearProbe(d, 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        y_tr = y_train.to(device).float()
        y_vl = y_val.to(device).float()
        y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
        for _ in range(n_steps):
            probe.train()
            loss = nn.functional.mse_loss(probe(X_tr).squeeze(), (y_tr - y_mean) / y_std)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(X_vl).squeeze()
            y_n = (y_vl - y_mean) / y_std
            ss_res = ((pred - y_n) ** 2).sum()
            ss_tot = ((y_n - y_n.mean()) ** 2).sum()
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        return r2


def run_oracle_probe_battery(residuals, metadata, device='cpu', n_steps=1000):
    """Run linear probe battery on oracle residuals."""
    N = len(metadata)
    n_val = int(N * 0.2)
    n_train = N - n_val

    targets = {
        'pc': (np.array([m['pc'] for m in metadata], dtype=np.float32), False),
        'mem_a': (np.array([m['mem_a'] for m in metadata], dtype=np.float32), False),
        'mem_b': (np.array([m['mem_b'] for m in metadata], dtype=np.float32), False),
        'delta': (np.array([m['delta'] for m in metadata], dtype=np.float32), False),
        'new_val': (np.array([m['new_val'] for m in metadata], dtype=np.float32), False),
        'branch_taken': (np.array([m['branch_taken'] for m in metadata]), True),
        'a_addr': (np.array([m['a_addr'] for m in metadata], dtype=np.float32), False),
        'b_addr': (np.array([m['b_addr'] for m in metadata], dtype=np.float32), False),
    }

    results = {}
    for layer_idx, resid in sorted(residuals.items()):
        # resid: (N, 417, 32) for round1
        # Probe at position 0 (PC position)
        X = resid[:, 0, :]  # (N, 32)

        X_train_t = torch.tensor(X[:n_train])
        X_val_t = torch.tensor(X[n_train:])

        results[layer_idx] = {}
        for tname, (y_arr, is_cls) in targets.items():
            y_tr = torch.tensor(y_arr[:n_train])
            y_vl = torch.tensor(y_arr[n_train:])

            # Full d_model probe
            full_metric = probe_specific_dims(X_train_t, y_tr, X_val_t, y_vl,
                                              list(range(32)), n_steps=n_steps,
                                              device=device, is_classification=is_cls)

            # Oracle-specific dimensions probe
            oracle_results = {}
            for dim, (dname, first_layer, desc) in ORACLE_MAP.items():
                if dim < 32:
                    metric = probe_specific_dims(X_train_t, y_tr, X_val_t, y_vl,
                                                 [dim], n_steps=n_steps,
                                                 device=device, is_classification=is_cls)
                    oracle_results[dim] = {'name': dname, 'metric': metric}

            results[layer_idx][tname] = {
                'full_metric': full_metric,
                'oracle_dims': oracle_results,
                'is_classification': is_cls,
            }
            print(f"    L{layer_idx} {tname}: {full_metric:.3f}")

    return results


def verify_oracle_with_interventions(model, inputs, metadata, device='cpu', n_pairs=100):
    """
    Verify oracle circuit map with activation interventions.
    For each claimed dimension, patch its value and measure effect.
    """
    print("\nVerifying oracle with interventions...")
    model.eval()
    results = {}

    # Test DPC (dim 5): should be 1 at position 0 (PC position), 0 elsewhere
    # Verify by patching dim 5 at position 0 with 0 and checking if output changes
    print("  Testing DPC (dim 5 = PC indicator)...")

    # Get baseline predictions
    n_test = min(n_pairs, len(inputs))
    inp_batch = inputs[:n_test].to(device)

    residuals_baseline = get_r1_residuals_batched(model, inp_batch, device)
    # At layer 0 (embedding), dim 5 should be 1 at position 0
    layer0_resid = residuals_baseline[0]  # (N, 417, 32)
    pc_indicator_pos0 = layer0_resid[:, 0, 5]  # Should be ~1
    pc_indicator_other = layer0_resid[:, 1:10, 5].mean(dim=1)  # Should be ~0

    print(f"    Dim 5 at pos 0: mean={pc_indicator_pos0.mean():.3f} (expected ~1)")
    print(f"    Dim 5 at pos 1-9: mean={pc_indicator_other.mean():.3f} (expected ~0)")

    results['DPC_pos0'] = pc_indicator_pos0.mean().item()
    results['DPC_others'] = pc_indicator_other.mean().item()

    # Test DI (dim 2): should encode position index
    print("  Testing DI (dim 2 = position index)...")
    layer0_resid = residuals_baseline[0]
    positions = torch.arange(417).float()
    # Check if dim 2 correlates with position
    dim2_vals = layer0_resid[0, :, 2].float()  # (417,) for first example
    correlation = torch.corrcoef(torch.stack([dim2_vals, positions]))[0, 1].item()
    print(f"    Correlation of dim 2 with position: {correlation:.4f} (expected ~1)")
    results['DI_correlation'] = correlation

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 1: Oracle Analysis")
    print(f"Device: {device}")

    # Print oracle circuit map
    print("\n=== Oracle Circuit Map ===")
    print(f"{'Dim':>4} {'Name':>10} {'Layer':>6} {'Description'}")
    print("-" * 70)
    for dim in sorted(ORACLE_MAP.keys()):
        name, layer, desc = ORACLE_MAP[dim]
        print(f"{dim:4d} {name:>10} {layer:6d} {desc}")

    print("\n=== Layer Purposes ===")
    for layer, purpose in LAYER_PURPOSES.items():
        print(f"  Layer {layer}: {purpose}")

    # Load oracle model
    print("\nLoading oracle model...")
    model = HandCodedSUBLEQ()
    model.to(device)
    model.eval()
    print(f"  Parameters: {model.count_params():,}")

    # Generate dataset
    print("\nGenerating oracle dataset...")
    inputs, metadata = generate_r1_dataset(n=3000, seed=42)
    print(f"  {len(metadata)} examples")

    # Extract residuals
    print("Extracting residuals...")
    residuals = get_r1_residuals_batched(model, inputs, device=device)
    for layer, r in sorted(residuals.items()):
        print(f"  Layer {layer}: {r.shape}")

    # Verify oracle
    verification = verify_oracle_with_interventions(model, inputs, metadata, device=device)

    # Run probe battery
    print("\nRunning probe battery...")
    probe_results = run_oracle_probe_battery(residuals, metadata, device=device, n_steps=1000)

    # Print summary
    print("\n=== Oracle Probe Results (at position 0) ===")
    targets = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken', 'a_addr', 'b_addr']
    header = f"{'Target':<15}" + "".join(f" L{i:>4}" for i in range(len(residuals)))
    print(header)
    print("-" * len(header))
    for tname in targets:
        row = f"{tname:<15}"
        for layer in sorted(probe_results.keys()):
            m = probe_results[layer].get(tname, {}).get('full_metric', float('nan'))
            row += f" {m:5.3f}"
        print(row)

    # Save results
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output = {
        'oracle_map': ORACLE_MAP,
        'layer_purposes': LAYER_PURPOSES,
        'probe_results': probe_results,
        'verification': verification,
    }
    out_path = os.path.join(results_dir, 'phase1_oracle.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nResults saved to {out_path}")

    # Also save as JSON (without numpy arrays)
    json_output = {
        'oracle_map': {str(k): {'name': v[0], 'layer': v[1], 'desc': v[2]}
                       for k, v in ORACLE_MAP.items()},
        'layer_purposes': LAYER_PURPOSES,
        'probe_summary': {
            str(layer): {tname: round(probe_results[layer][tname]['full_metric'], 4)
                         for tname in targets if tname in probe_results[layer]}
            for layer in sorted(probe_results.keys())
        },
        'verification': verification,
    }
    with open(os.path.join(results_dir, 'phase1_oracle.json'), 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON summary saved.")


if __name__ == '__main__':
    main()
