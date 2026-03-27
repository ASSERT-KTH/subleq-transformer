#!/usr/bin/env python3
"""
Phase 6: Additional analyses.

Part 1: Dimensional localization — for each quantity, run localization
  analysis (probe with top-k dimensions) on both oracle and trained models.
  Uses existing phase1_oracle.pkl and phase2_probe_trained.pkl probes.

Part 2: Training dynamics — probe accuracy at each checkpoint fraction
  (10%, 25%, 50%, 75%, 100%) for seeds 1-4.

Part 3: Step-by-step failure trace — for the 10 consistent failure programs,
  run the trained model step-by-step and apply probes at each (layer, step)
  to find where it goes wrong.
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
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))
sys.path.insert(0, script_dir)

from extract_residuals import load_r2_model, generate_metadata_dataset, get_r2_residuals_batched, get_r2_residuals
from probe import (train_regression_probe, train_classification_probe,
                   probe_all_layers, localization_analysis)

# ── Part 1: Dimensional Localization ─────────────────────────────────────────

def run_localization_all_targets(residuals, metadata, device='cpu', n_steps=500):
    """
    For each quantity, run localization analysis at every layer.

    Returns: {target_name: {layer: [(k, metric), ...]}}
    """
    N = len(metadata)
    n_val = int(N * 0.2)
    n_train = N - n_val

    target_defs = {
        'pc':           (np.array([m['pc'] for m in metadata], dtype=np.float32), False),
        'mem_a':        (np.array([m['mem_a'] for m in metadata], dtype=np.float32), False),
        'mem_b':        (np.array([m['mem_b'] for m in metadata], dtype=np.float32), False),
        'delta':        (np.array([m['delta'] for m in metadata], dtype=np.float32), False),
        'branch_taken': (np.array([m['branch_taken'] for m in metadata], dtype=np.int64), True),
    }

    results = {}

    for tname, (y_arr, is_cls) in target_defs.items():
        results[tname] = {}
        for layer_idx, resid in sorted(residuals.items()):
            X = resid[:, 0, :]  # (N, d_model) — probe at position 0

            X_train = torch.tensor(X[:n_train].numpy() if hasattr(X, 'numpy') else X[:n_train])
            X_val   = torch.tensor(X[n_train:].numpy() if hasattr(X, 'numpy') else X[n_train:])
            y_tr = torch.tensor(y_arr[:n_train])
            y_vl = torch.tensor(y_arr[n_train:])

            # First train full probe to get weights
            if is_cls:
                _, weights = train_classification_probe(X_train, y_tr, X_val, y_vl,
                                                        n_steps=n_steps, device=device)
            else:
                _, weights = train_regression_probe(X_train, y_tr, X_val, y_vl,
                                                    n_steps=n_steps, device=device)

            loc = localization_analysis(X_train, y_tr, X_val, y_vl, weights,
                                        n_steps=n_steps, device=device,
                                        is_classification=is_cls)
            results[tname][layer_idx] = loc
            print(f"    {tname} L{layer_idx}: {[(k, f'{m:.3f}') for k, m in loc]}")

    return results


def run_oracle_localization(device='cpu'):
    """Run localization analysis on oracle residuals."""
    from phase1_oracle import generate_r1_dataset, get_r1_residuals_batched
    from model import HandCodedSUBLEQ

    print("Loading oracle model...")
    oracle_model = HandCodedSUBLEQ()
    oracle_model.to(device)
    oracle_model.eval()

    print("Generating oracle dataset (2000 examples)...")
    inputs, metadata = generate_r1_dataset(n=2000, seed=42)

    print("Extracting oracle residuals...")
    residuals = get_r1_residuals_batched(oracle_model, inputs, device=device)

    print("Running oracle localization analysis...")
    loc = run_localization_all_targets(residuals, metadata, device=device)
    return loc


# ── Part 2: Training Dynamics ─────────────────────────────────────────────────

def run_training_dynamics(inputs, metadata, ckpt_dir, device='cpu', n_steps=500):
    """
    For each checkpoint fraction × seed, run probe battery and record
    best probe metric per target.

    Returns: {frac: {seed: {target: {layer: metric}}}}
    """
    fracs = [10, 25, 50, 75, 100]
    seeds = [0, 1, 2, 3, 4]

    results = {}
    for frac in fracs:
        results[frac] = {}
        for seed in seeds:
            if seed == 0:
                # Seed 0 only has final
                if frac != 100:
                    continue
                ckpt_path = os.path.join(repo_root, 'round2_trained',
                                         'checkpoints', 'best_model.pt')
            else:
                ckpt_path = os.path.join(ckpt_dir, f'seed{seed}_frac{frac:03d}.pt')
            if not os.path.exists(ckpt_path):
                print(f"  Skipping frac={frac} seed={seed}: not found ({ckpt_path})")
                continue

            print(f"  Probing seed {seed} at frac {frac}%...")
            model, config = load_r2_model(ckpt_path, device)
            probe_results = probe_all_layers(
                get_r2_residuals_batched(model, inputs, device=device)[0],
                metadata, n_steps=n_steps, device=device
            )

            seed_summary = {}
            for tname in ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']:
                seed_summary[tname] = {}
                for layer in sorted(probe_results.keys()):
                    m = probe_results[layer].get('pos0', {}).get(tname, {}).get('metric', None)
                    if m is not None:
                        seed_summary[tname][layer] = float(m)

            results[frac][seed] = seed_summary
            del model

    return results


# ── Part 3: Step-by-step failure trace ───────────────────────────────────────

def run_failure_trace(model, probes_by_layer, failure_program, device='cpu', max_steps=20):
    """
    Run the model step-by-step on a failing program.
    At each step, extract residuals and apply probes at each layer.

    probes_by_layer: {layer: {tname: (weights_or_none, is_cls, y_mean, y_std, X_mean, X_std)}}
      (simplified: just re-run probe_all_layers on each step's residual)

    Returns: list of {step, pc, layer_results: {layer: {target: model_pred, gt}}}
    """
    from subleq import step as subleq_step, encode, decode, run as subleq_run
    from subleq.interpreter import clamp

    mem, pc = failure_program['memory'], failure_program['pc']
    prog_mem0 = list(mem)

    trace_steps = []
    for s in range(max_steps):
        if pc < 0 or pc + 2 >= len(mem):
            break

        # Ground truth for this step
        a_addr = mem[pc]
        b_addr = mem[pc + 1] if pc + 1 < len(mem) else 0
        mem_a = mem[a_addr] if 0 <= a_addr < len(mem) else 0
        mem_b = mem[b_addr] if 0 <= b_addr < len(mem) else 0
        delta_gt = mem_b - mem_a
        delta_clamped = clamp(delta_gt)
        branch_gt = int(delta_clamped <= 0)

        gt = {
            'pc': pc, 'a_addr': a_addr, 'b_addr': b_addr,
            'mem_a': mem_a, 'mem_b': mem_b,
            'delta': delta_gt, 'branch_taken': branch_gt,
        }

        # Model prediction
        inp = encode(mem, pc).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)  # (1, seq_len, vocab)
        pred_tokens = logits.argmax(dim=-1).squeeze(0)
        pred_mem, pred_pc = decode(pred_tokens)

        # True next state
        true_mem, true_pc, halted = subleq_step(mem, pc)

        step_info = {
            'step': s,
            'pc': pc,
            'gt': gt,
            'pred_pc': pred_pc,
            'true_pc': true_pc,
            'correct': (pred_mem == list(true_mem) and pred_pc == true_pc),
        }

        # Extract residuals at this step for probe analysis
        with torch.no_grad():
            residuals, _ = get_r2_residuals(model, inp, device=device)

        step_probes = {}
        for layer_idx, resid in residuals.items():
            # resid: (1, seq_len, d_model) → take position 0
            x_pos0 = resid[0, 0, :].numpy()  # (d_model,)
            step_probes[layer_idx] = x_pos0

        step_info['residuals_pos0'] = step_probes
        trace_steps.append(step_info)

        # Advance using ground truth (to trace correctly)
        if halted:
            break
        mem = list(true_mem)
        pc = true_pc

    return trace_steps


def apply_probes_to_trace(trace_steps, seed_probes, metadata_for_probes, device='cpu'):
    """
    Given a trace and trained probe weights, decode each residual.

    seed_probes: full probe_results from probe_all_layers
    Returns enriched trace with decoded quantities at each (step, layer).
    """
    # Build simple linear decoders from probe weights
    for step_info in trace_steps:
        layer_decoded = {}
        for layer_idx, x_pos0 in step_info['residuals_pos0'].items():
            decoded = {}
            for tname in ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']:
                probe_info = seed_probes.get(layer_idx, {}).get('pos0', {}).get(tname)
                if probe_info is None:
                    continue
                w = probe_info['weights']  # ndarray
                is_cls = probe_info.get('type') == 'accuracy'
                if is_cls:
                    # w: (2, d), logit = w @ x, argmax
                    logits = w.dot(x_pos0)  # (2,)
                    decoded[tname] = int(np.argmax(logits))
                else:
                    # w: (d,) or (1, d)
                    if w.ndim > 1:
                        w = w.flatten()
                    decoded[tname] = float(w.dot(x_pos0))
            layer_decoded[layer_idx] = decoded
        step_info['decoded'] = layer_decoded
    return trace_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-examples', type=int, default=3000)
    parser.add_argument('--skip-localization', action='store_true')
    parser.add_argument('--skip-dynamics', action='store_true')
    parser.add_argument('--skip-trace', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 6: Additional Analyses")
    print(f"Device: {device}")

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(script_dir, 'checkpoints')
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'results')
    os.makedirs(args.output_dir, exist_ok=True)

    results_dir = args.output_dir

    # ── Part 1: Dimensional Localization ─────────────────────────────────────
    if not args.skip_localization:
        print("\n=== Part 1: Dimensional Localization ===")

        # Oracle localization
        print("\n--- Oracle ---")
        oracle_loc = run_oracle_localization(device=device)

        # Trained model localization (seed 0 = best_model.pt)
        print("\n--- Trained (seed 0) ---")
        seed0_ckpt = os.path.join(repo_root, 'round2_trained',
                                   'checkpoints', 'best_model.pt')
        model0, _ = load_r2_model(seed0_ckpt, device)
        inputs_loc, metadata_loc = generate_metadata_dataset(n=args.n_examples, seed=42)
        residuals0, _ = get_r2_residuals_batched(model0, inputs_loc, device=device)
        trained_loc = run_localization_all_targets(residuals0, metadata_loc, device=device)
        del model0

        localization_output = {
            'oracle': oracle_loc,
            'trained_seed0': trained_loc,
        }
        with open(os.path.join(results_dir, 'phase6_localization.pkl'), 'wb') as f:
            pickle.dump(localization_output, f)

        # JSON-serializable summary
        def loc_to_json(loc_dict):
            out = {}
            for tname, layer_dict in loc_dict.items():
                out[tname] = {}
                for layer, pairs in layer_dict.items():
                    out[tname][str(layer)] = [[int(k), float(m)] for k, m in pairs]
            return out

        json_loc = {
            'oracle': loc_to_json(oracle_loc),
            'trained_seed0': loc_to_json(trained_loc),
        }
        with open(os.path.join(results_dir, 'phase6_localization.json'), 'w') as f:
            json.dump(json_loc, f, indent=2)
        print("Localization results saved.")

    # ── Part 2: Training Dynamics ─────────────────────────────────────────────
    if not args.skip_dynamics:
        print("\n=== Part 2: Training Dynamics ===")

        inputs_dyn, metadata_dyn = generate_metadata_dataset(n=args.n_examples, seed=123)

        dynamics = run_training_dynamics(inputs_dyn, metadata_dyn,
                                         args.ckpt_dir, device=device)

        with open(os.path.join(results_dir, 'phase6_dynamics.pkl'), 'wb') as f:
            pickle.dump(dynamics, f)

        # JSON-friendly
        json_dyn = {}
        for frac, seed_dict in dynamics.items():
            json_dyn[str(frac)] = {}
            for seed, tdict in seed_dict.items():
                json_dyn[str(frac)][str(seed)] = {
                    tname: {str(l): v for l, v in ldict.items()}
                    for tname, ldict in tdict.items()
                }
        with open(os.path.join(results_dir, 'phase6_dynamics.json'), 'w') as f:
            json.dump(json_dyn, f, indent=2)
        print("Dynamics results saved.")

    # ── Part 3: Step-by-step failure trace ───────────────────────────────────
    if not args.skip_trace:
        print("\n=== Part 3: Failure Trace ===")

        # Load phase4 failure data
        phase4_pkl = os.path.join(results_dir, 'phase4_failures.pkl')
        if not os.path.exists(phase4_pkl):
            print(f"  phase4_failures.pkl not found at {phase4_pkl}, skipping trace.")
        else:
            with open(phase4_pkl, 'rb') as f:
                phase4_data = pickle.load(f)

            # Load seed 0 model and its probes
            seed0_ckpt = os.path.join(repo_root, 'round2_trained',
                                       'checkpoints', 'best_model.pt')
            model0, _ = load_r2_model(seed0_ckpt, device)

            # Load phase2 probe results for seed 0
            phase2_pkl = os.path.join(results_dir, 'phase2_probe_trained.pkl')
            if os.path.exists(phase2_pkl):
                with open(phase2_pkl, 'rb') as f:
                    phase2_data = pickle.load(f)
                seed0_probes = (phase2_data.get('all_seed_results', {})
                                .get(0, {}).get('probe_results', {}))
            else:
                seed0_probes = {}

            # Get failure programs from seed 0 of phase4 pickle
            # Structure: {seed_id: {'failures': [{'name', 'mem_init', 'pc_init', ...}]}}
            failure_programs = []
            if isinstance(phase4_data, dict) and 0 in phase4_data:
                seed0_failures = phase4_data[0].get('failures', [])
                # Find programs that failed in all (or most) seeds
                failure_counts = {}
                for seed_id, seed_data in phase4_data.items():
                    for f in seed_data.get('failures', []):
                        name = f.get('name', '')
                        failure_counts[name] = failure_counts.get(name, 0) + 1
                # Pick top consistent failures
                consistent_names = sorted(failure_counts, key=lambda n: -failure_counts[n])[:3]
                for fail in seed0_failures:
                    if fail.get('name') in consistent_names:
                        failure_programs.append({
                            'name': fail['name'],
                            'memory': fail.get('mem_init', []),
                            'pc': fail.get('pc_init', 0),
                        })

            trace_results = {}
            for prog in failure_programs[:3]:
                name = prog.get('name', 'unknown')
                print(f"  Tracing {name}...")
                trace = run_failure_trace(model0, seed0_probes, prog,
                                          device=device, max_steps=15)
                if seed0_probes:
                    trace = apply_probes_to_trace(trace, seed0_probes, None, device=device)
                # Strip large residual arrays from JSON output
                trace_json = []
                for step_info in trace:
                    s = dict(step_info)
                    s.pop('residuals_pos0', None)
                    trace_json.append(s)
                trace_results[name] = trace_json
                print(f"    {len(trace)} steps traced")

            with open(os.path.join(results_dir, 'phase6_failure_trace.json'), 'w') as f:
                json.dump(trace_results, f, indent=2, default=str)
            print("Failure trace results saved.")

            del model0

    print("\nPhase 6 complete.")


if __name__ == '__main__':
    main()
