#!/usr/bin/env python3
"""
Phase 2 (held-out): Distribution generalization of linear probes.

Probes trained on random-state execution steps.
Evaluated on execution traces of structured programs NOT seen in the same distribution:
  - fibonacci  (never in training data)
  - countdown  (in training as traces, but different distribution from random states)
  - multiply   (in training as traces)
  - addition   (in training as traces)

Reports R² / accuracy per program type and layer, for seed 0's best_model.
Saves: results/phase2_heldout.json
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from subleq import step, encode
from subleq.interpreter import MEM_SIZE, clamp
from subleq.data import generate_trace_pairs
from subleq.tokenizer import decode as tok_decode
from subleq.programs import (make_fibonacci, make_countdown, make_multiply,
                              make_addition, generate_random_state)
from extract_residuals import load_r2_model, get_r2_residuals_batched, generate_metadata_dataset


TARGETS = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']
IS_CLASSIFICATION = {'branch_taken': True}


# ── Self-contained probe with stored normalization ────────────────────────────

class FittedProbe:
    """Linear probe that stores normalization stats for evaluation on new data."""

    def __init__(self, is_cls):
        self.is_cls = is_cls
        # Set by fit()
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.W = None   # (d_out, d_in)
        self.b = None   # (d_out,)

    def fit(self, X_tr, y_tr, X_vl, y_vl, n_steps=500, lr=1e-2, device='cpu'):
        """Train probe. Returns validation metric (R² or accuracy)."""
        X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_tr).to(device)
        X_vl = torch.tensor(X_vl, dtype=torch.float32).to(device)
        y_vl = torch.tensor(y_vl).to(device)

        # Normalize inputs
        self.X_mean = X_tr.mean(dim=0)
        self.X_std = X_tr.std(dim=0) + 1e-8
        X_tr_n = (X_tr - self.X_mean) / self.X_std
        X_vl_n = (X_vl - self.X_mean) / self.X_std

        d_in = X_tr.shape[1]

        if self.is_cls:
            d_out = int(y_tr.max().item()) + 1
            probe = nn.Linear(d_in, d_out).to(device)
            y_tr = y_tr.long()
            y_vl = y_vl.long()
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            for _ in range(n_steps):
                probe.train()
                loss = nn.functional.cross_entropy(probe(X_tr_n), y_tr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probe.eval()
            with torch.no_grad():
                acc = (probe(X_vl_n).argmax(dim=-1) == y_vl).float().mean().item()
            self.W = probe.weight.data.cpu()
            self.b = probe.bias.data.cpu()
            return acc

        else:
            # Normalize targets
            y_tr_f = y_tr.float()
            self.y_mean = y_tr_f.mean()
            self.y_std = y_tr_f.std() + 1e-8
            y_tr_n = (y_tr_f - self.y_mean) / self.y_std
            y_vl_f = y_vl.float()
            y_vl_n = (y_vl_f - self.y_mean) / self.y_std

            probe = nn.Linear(d_in, 1).to(device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            for _ in range(n_steps):
                probe.train()
                loss = nn.functional.mse_loss(probe(X_tr_n).squeeze(-1), y_tr_n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probe.eval()
            with torch.no_grad():
                pred_n = probe(X_vl_n).squeeze(-1)
                ss_res = ((pred_n - y_vl_n) ** 2).sum()
                ss_tot = ((y_vl_n - y_vl_n.mean()) ** 2).sum()
                r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
            self.W = probe.weight.data.cpu()
            self.b = probe.bias.data.cpu()
            return r2

    def evaluate(self, X, y, device='cpu'):
        """Evaluate on new data (returns R² or accuracy)."""
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y)
        X_n = (X - self.X_mean.to(device)) / self.X_std.to(device)
        W = self.W.to(device)
        b = self.b.to(device)

        with torch.no_grad():
            logits_or_pred = (X_n @ W.T + b)
            if self.is_cls:
                preds = logits_or_pred.argmax(dim=-1).cpu()
                return (preds == y.long()).float().mean().item()
            else:
                pred_n = logits_or_pred.squeeze(-1)
                # Un-normalize for R² in original space
                pred = pred_n * self.y_std.to(device) + self.y_mean.to(device)
                pred = pred.cpu()
                y_f = y.float()
                ss_res = ((y_f - pred) ** 2).sum()
                ss_tot = ((y_f - y_f.mean()) ** 2).sum()
                return (1 - ss_res / (ss_tot + 1e-8)).item()


# ── Data generation ───────────────────────────────────────────────────────────

def compute_step_metadata(mem, pc):
    """Compute probe targets for one SUBLEQ step. Returns None if invalid."""
    if pc < 0 or pc + 2 >= MEM_SIZE:
        return None
    a_addr = mem[pc]
    b_addr = mem[pc + 1]
    if not (0 <= a_addr < MEM_SIZE and 0 <= b_addr < MEM_SIZE):
        return None
    mem_a = mem[a_addr]
    mem_b = mem[b_addr]
    delta = mem_b - mem_a
    new_val = clamp(delta)
    branch_taken = int(new_val <= 0)
    _, _, halted = step(mem, pc)
    if halted:
        return None
    return {
        'pc': pc,
        'mem_a': mem_a,
        'mem_b': mem_b,
        'delta': float(delta),
        'branch_taken': branch_taken,
    }


def generate_program_trace_dataset(prog_gen, n_examples, seed=42):
    """Generate dataset from structured-program execution traces."""
    random.seed(seed)
    inputs = []
    metadata = []

    while len(inputs) < n_examples:
        result = prog_gen()
        mem = list(result[0])
        pc = result[1]

        pairs = generate_trace_pairs(mem, pc, max_steps=50)
        for inp_tensor, out_tensor, _ in pairs:
            if len(inputs) >= n_examples:
                break
            cur_mem, cur_pc = tok_decode(inp_tensor)
            meta = compute_step_metadata(cur_mem, cur_pc)
            if meta is None:
                continue
            inputs.append(inp_tensor)
            metadata.append(meta)

        if len(inputs) == 0 and len(metadata) == 0:
            # Guard against infinite loop on zero-step programs
            break

    if not inputs:
        return None, None
    print(f"  Generated {len(inputs)} steps")
    return torch.stack(inputs[:n_examples]), metadata[:n_examples]


def fibonacci_gen():
    n = random.randint(1, 5)
    mem, pc, _, _ = make_fibonacci(n)
    return mem, pc


def countdown_gen():
    n = random.randint(2, 20)
    mem, pc, _ = make_countdown(n)
    return mem, pc


def multiply_gen():
    a = random.randint(1, 8)
    b = random.randint(1, min(10, 127 // max(a, 1)))
    mem, pc, _ = make_multiply(a, b)
    return mem, pc


def addition_gen():
    a = random.randint(-60, 60)
    b = random.randint(-60, 60)
    mem, pc, _ = make_addition(a, b)
    return mem, pc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(repo_root, 'round2_trained',
                                             'checkpoints', 'best_model.pt'))
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(script_dir, 'results'))
    parser.add_argument('--n-train', type=int, default=5000)
    parser.add_argument('--n-heldout', type=int, default=1000)
    parser.add_argument('--n-steps', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 2 (held-out): Distribution Generalization of Probes")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from {args.ckpt}...")
    model, config = load_r2_model(args.ckpt, device)
    n_layers = config.get('n_layers', 6)
    print(f"  d_model={config.get('d_model')}, n_layers={n_layers}")

    # 1. Random-state data for probe training
    print(f"\nGenerating {args.n_train} random-state examples...")
    train_inputs, train_meta = generate_metadata_dataset(n=args.n_train, seed=42)
    print("Extracting training residuals...")
    train_res, _ = get_r2_residuals_batched(model, train_inputs, device=device)

    # 2. Train probes (store normalization)
    N = len(train_meta)
    n_val = int(N * 0.2)
    n_tr = N - n_val

    target_arrays = {
        'pc':           np.array([m['pc'] for m in train_meta], dtype=np.float32),
        'mem_a':        np.array([m['mem_a'] for m in train_meta], dtype=np.float32),
        'mem_b':        np.array([m['mem_b'] for m in train_meta], dtype=np.float32),
        'delta':        np.array([m['delta'] for m in train_meta], dtype=np.float32),
        'branch_taken': np.array([m['branch_taken'] for m in train_meta], dtype=np.int64),
    }

    # fitted probes: tname -> layer_idx -> FittedProbe
    fitted = {t: {} for t in TARGETS}
    iid_metrics = {t: {} for t in TARGETS}

    print("\nTraining probes on random-state data...")
    for tname in TARGETS:
        is_cls = IS_CLASSIFICATION.get(tname, False)
        y_arr = target_arrays[tname]
        print(f"  {tname}:")
        for layer_idx, resid in sorted(train_res.items()):
            X = resid[:, 0, :].numpy()  # (N, d_model)
            X_tr, X_vl = X[:n_tr], X[n_tr:]
            y_tr, y_vl = y_arr[:n_tr], y_arr[n_tr:]

            probe = FittedProbe(is_cls=is_cls)
            metric = probe.fit(X_tr, y_tr, X_vl, y_vl,
                               n_steps=args.n_steps, device=device)
            fitted[tname][layer_idx] = probe
            iid_metrics[tname][layer_idx] = metric
            print(f"    L{layer_idx}: {metric:.3f}")

    # 3. Evaluate on structured programs
    program_configs = [
        ('fibonacci', fibonacci_gen),
        ('countdown', countdown_gen),
        ('multiply', multiply_gen),
        ('addition', addition_gen),
    ]

    heldout_metrics = {}

    for prog_name, prog_gen in program_configs:
        print(f"\nGenerating {args.n_heldout} steps from {prog_name}...")
        ho_inputs, ho_meta = generate_program_trace_dataset(
            prog_gen, n_examples=args.n_heldout, seed=100 + abs(hash(prog_name)) % 1000)

        if ho_inputs is None or len(ho_meta) == 0:
            print(f"  No data for {prog_name}")
            heldout_metrics[prog_name] = {}
            continue

        print(f"  Extracting residuals ({prog_name})...")
        ho_res, _ = get_r2_residuals_batched(model, ho_inputs, device=device)

        ho_target_arrays = {
            'pc':           np.array([m['pc'] for m in ho_meta], dtype=np.float32),
            'mem_a':        np.array([m['mem_a'] for m in ho_meta], dtype=np.float32),
            'mem_b':        np.array([m['mem_b'] for m in ho_meta], dtype=np.float32),
            'delta':        np.array([m['delta'] for m in ho_meta], dtype=np.float32),
            'branch_taken': np.array([m['branch_taken'] for m in ho_meta], dtype=np.int64),
        }

        prog_metrics = {}
        for tname in TARGETS:
            prog_metrics[tname] = {}
            y_ho = ho_target_arrays[tname]
            for layer_idx, resid in sorted(ho_res.items()):
                X_ho = resid[:, 0, :].numpy()
                probe = fitted[tname].get(layer_idx)
                if probe is None:
                    continue
                metric = probe.evaluate(X_ho, y_ho, device=device)
                prog_metrics[tname][layer_idx] = metric

        heldout_metrics[prog_name] = prog_metrics

        print(f"  {prog_name} metrics (best layer):")
        for tname in TARGETS:
            best_iid = max(iid_metrics[tname].values(), default=0)
            best_ho = max(prog_metrics.get(tname, {0: 0}).values(), default=0)
            print(f"    {tname}: {best_ho:.3f} (iid: {best_iid:.3f})")

    # 4. Save
    # Convert integer layer keys to strings for JSON
    def jsonify(d):
        if isinstance(d, dict):
            return {str(k): jsonify(v) for k, v in d.items()}
        if isinstance(d, (np.floating, float)):
            return float(d)
        return d

    output = {
        'iid_metrics': jsonify(iid_metrics),
        'heldout_metrics': jsonify(heldout_metrics),
        'n_train': args.n_train,
        'n_heldout': args.n_heldout,
        'ckpt': args.ckpt,
        'config': config,
        'targets': TARGETS,
    }
    out_path = os.path.join(args.output_dir, 'phase2_heldout.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n=== Summary: Best probe metric across layers ===")
    header = f"{'Target':<15} {'IID':>8}"
    for prog_name, _ in program_configs:
        header += f" {prog_name[:8]:>10}"
    print(header)
    for tname in TARGETS:
        iid_best = max(iid_metrics[tname].values(), default=0.0)
        row = f"{tname:<15} {iid_best:>8.3f}"
        for prog_name, _ in program_configs:
            best = max(heldout_metrics.get(prog_name, {}).get(tname, {0: 0.0}).values(),
                       default=0.0)
            row += f" {best:>10.3f}"
        print(row)


if __name__ == '__main__':
    main()
