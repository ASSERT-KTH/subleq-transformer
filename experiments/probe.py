#!/usr/bin/env python3
"""
Linear probing for all quantities at all layers.

Probe targets (for round2 model):
  - pc: integer PC value (regression, R²)
  - mem_a: value at address a (regression, R²)
  - mem_b: value at address b (regression, R²)
  - delta: mem[b] - mem[a] before clamping (regression, R²)
  - new_val: mem[b] - mem[a] after clamping (regression, R²)
  - branch_taken: bool (classification, accuracy)
  - a_addr: address a = mem[pc] (regression)
  - b_addr: address b = mem[pc+1] (regression)

For each target × layer × position, we train a linear probe.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))


class LinearProbe(nn.Module):
    """Single linear layer probe (no nonlinearity)."""
    def __init__(self, d_in, d_out=1):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)


def train_regression_probe(X_train, y_train, X_val, y_val, n_steps=1000, lr=1e-2, device='cpu'):
    """
    Train a linear regression probe. Returns R² on val set and probe weights.

    X: (N, d_model)
    y: (N,) float
    """
    d = X_train.shape[1]
    probe = LinearProbe(d, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    # Normalize inputs (important for oracle model which has very large activations)
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    for step in range(n_steps):
        probe.train()
        pred = probe(X_train).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_train_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred_val = probe(X_val).squeeze(-1)
        # R² = 1 - SS_res / SS_tot
        ss_res = ((pred_val - y_val_norm) ** 2).sum()
        ss_tot = ((y_val_norm - y_val_norm.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

    # Get weight magnitudes for localization analysis
    weights = probe.linear.weight.data.cpu().numpy().flatten()
    return r2, weights


def train_classification_probe(X_train, y_train, X_val, y_val, n_steps=1000, lr=1e-2, device='cpu'):
    """
    Train a linear classification probe. Returns accuracy on val set.

    X: (N, d_model)
    y: (N,) long (0 or 1)
    """
    d = X_train.shape[1]
    probe = LinearProbe(d, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).long()

    # Normalize inputs
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    for step in range(n_steps):
        probe.train()
        pred = probe(X_train)
        loss = nn.functional.cross_entropy(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred_val = probe(X_val)
        acc = (pred_val.argmax(dim=-1) == y_val).float().mean().item()

    weights = probe.linear.weight.data.cpu().numpy()
    return acc, weights


def probe_all_layers(residuals, metadata, n_steps=1000, device='cpu', val_frac=0.2):
    """
    Run probe battery on all layers.

    residuals: dict {layer_idx: (N, seq_len, d_model)}
    metadata: list of N dicts

    Returns: dict of results
    """
    N = len(metadata)
    n_val = int(N * val_frac)
    n_train = N - n_val

    # Extract target arrays
    targets = {
        'pc': np.array([m['pc'] for m in metadata], dtype=np.float32),
        'mem_a': np.array([m['mem_a'] for m in metadata], dtype=np.float32),
        'mem_b': np.array([m['mem_b'] for m in metadata], dtype=np.float32),
        'delta': np.array([m['delta'] for m in metadata], dtype=np.float32),
        'new_val': np.array([m['new_val'] for m in metadata], dtype=np.float32),
        'branch_taken': np.array([m['branch_taken'] for m in metadata], dtype=np.int64),
        'a_addr': np.array([m['a_addr'] for m in metadata], dtype=np.float32),
        'b_addr': np.array([m['b_addr'] for m in metadata], dtype=np.float32),
    }

    classification_targets = {'branch_taken'}

    results = {}
    n_layers = max(residuals.keys()) + 1

    for layer_idx, resid in residuals.items():
        # resid: (N, seq_len, d_model)
        print(f"  Layer {layer_idx}/{max(residuals.keys())}...")

        results[layer_idx] = {}

        # Probe at position 0 (PC position) - most natural for PC-related quantities
        # Also probe at aggregated representation (mean across all positions)
        # And position-specific for memory quantities

        for probe_pos in ['pos0', 'mean']:
            if probe_pos == 'pos0':
                X = resid[:, 0, :]  # (N, d_model)
            else:
                X = resid.mean(dim=1)  # (N, d_model)

            X_train = torch.tensor(X[:n_train].numpy())
            X_val = torch.tensor(X[n_train:].numpy())

            probe_results = {}
            for target_name, y in targets.items():
                y_train = torch.tensor(y[:n_train])
                y_val = torch.tensor(y[n_train:])

                if target_name in classification_targets:
                    acc, weights = train_classification_probe(
                        X_train, y_train, X_val, y_val, n_steps=n_steps, device=device)
                    probe_results[target_name] = {'metric': acc, 'type': 'accuracy', 'weights': weights}
                else:
                    r2, weights = train_regression_probe(
                        X_train, y_train, X_val, y_val, n_steps=n_steps, device=device)
                    probe_results[target_name] = {'metric': r2, 'type': 'r2', 'weights': weights}

            results[layer_idx][probe_pos] = probe_results

    return results


def localization_analysis(X_train, y_train, X_val, y_val, weights, n_steps=500, device='cpu', is_classification=False):
    """
    Train probes using only top-k dimensions (by weight magnitude).
    Returns list of (k, metric) pairs.
    """
    d = X_train.shape[1]
    # Rank dimensions by weight magnitude
    if weights.ndim == 1:
        magnitudes = np.abs(weights)
    else:
        magnitudes = np.abs(weights).sum(axis=0)
    ranked = np.argsort(magnitudes)[::-1].copy()  # descending (copy to fix negative stride)

    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, d]
    ks = [k for k in ks if k <= d]

    results = []
    for k in ks:
        top_k_dims = ranked[:k]
        X_tr_k = X_train[:, top_k_dims]
        X_vl_k = X_val[:, top_k_dims]

        if is_classification:
            metric, _ = train_classification_probe(X_tr_k, y_train, X_vl_k, y_val,
                                                    n_steps=n_steps, device=device)
        else:
            metric, _ = train_regression_probe(X_tr_k, y_train, X_vl_k, y_val,
                                                n_steps=n_steps, device=device)
        results.append((k, metric))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output .pkl file')
    parser.add_argument('--n-examples', type=int, default=5000)
    parser.add_argument('--n-steps', type=int, default=1000, help='Probe training steps')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    from extract_residuals import load_r2_model, generate_metadata_dataset, get_r2_residuals_batched

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt}")
    model, config = load_r2_model(args.ckpt, device)
    print(f"Model: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    # Generate data
    print(f"Generating {args.n_examples} examples...")
    inputs, metadata = generate_metadata_dataset(n=args.n_examples, seed=args.seed)

    # Extract residuals
    print("Extracting residuals...")
    residuals, logits = get_r2_residuals_batched(model, inputs, device=device)
    print(f"  {len(residuals)} layers, shape: {residuals[0].shape}")

    # Run probes
    print("Running probes...")
    results = probe_all_layers(residuals, metadata, n_steps=args.n_steps, device=device)

    # Save results
    output = {
        'ckpt': args.ckpt,
        'config': config,
        'n_examples': args.n_examples,
        'results': results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    print(f"Results saved to {args.output}")

    # Print summary table
    print("\n=== Probe Summary ===")
    print(f"{'Target':<15} {'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5':>8} {'L6':>8}")
    for target in ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']:
        row = f"{target:<15}"
        for layer in sorted(results.keys()):
            r = results[layer].get('pos0', {}).get(target, {})
            m = r.get('metric', float('nan'))
            row += f" {m:8.3f}"
        print(row)
