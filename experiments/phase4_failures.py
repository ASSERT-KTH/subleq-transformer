#!/usr/bin/env python3
"""
Phase 4: Failure case analysis.

For each trained seed:
1. Find all multi-step programs where the model fails
2. Run step-by-step residual stream comparison
3. Identify the first divergence point (step, layer)
4. Use probe results to interpret the divergence
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
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from subleq import (step, run, encode, decode, MEM_SIZE, VALUE_MIN, VALUE_MAX,
                    make_negate, make_addition, make_countdown, make_multiply,
                    make_fibonacci, generate_random_program)
from subleq.interpreter import clamp
from extract_residuals import load_r2_model, get_r2_residuals


def model_step(model, memory, pc, device='cpu'):
    """One model forward pass."""
    from subleq import encode, decode
    inp = encode(memory, pc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred = logits.argmax(dim=-1).squeeze(0)
    new_mem, new_pc = decode(pred)
    return new_mem, new_pc


def run_model_multistep(model, memory, pc, max_steps=200, device='cpu'):
    """Run model iteratively until halt or max_steps."""
    m, p = list(memory), pc
    trace = [(list(m), p)]
    for s in range(max_steps):
        if p < 0 or p + 2 >= len(m):
            break
        m, p = model_step(model, m, p, device)
        trace.append((list(m), p))
    return m, p, trace


def find_failure_cases(model, device='cpu'):
    """Find all multi-step programs where the model fails."""
    failures = []
    successes = []

    test_cases = []

    # Negate
    for val in range(-100, 101):
        mem, pc, result_addr = make_negate(val)
        test_cases.append(('negate', f'negate({val})', mem, pc, result_addr, None))

    # Addition
    test_pairs = [(a, b) for a in range(-50, 51, 5) for b in range(-50, 51, 5)]
    random.shuffle(test_pairs)
    for a, b in test_pairs[:200]:
        mem, pc, result_addr = make_addition(a, b)
        test_cases.append(('addition', f'add({a},{b})', mem, pc, result_addr, None))

    # Countdown
    for n in range(1, 21):
        mem, pc, result_addr = make_countdown(n)
        test_cases.append(('countdown', f'countdown({n})', mem, pc, result_addr, None))

    # Multiply
    for a in range(1, 11):
        for b in range(1, min(11, VALUE_MAX // max(a, 1) + 1)):
            if a * b > VALUE_MAX:
                continue
            mem, pc, result_addr = make_multiply(a, b)
            test_cases.append(('multiply', f'mul({a},{b})', mem, pc, result_addr, None))

    # Fibonacci
    for n in range(1, 7):
        mem, pc, addr_a, addr_b = make_fibonacci(n)
        test_cases.append(('fibonacci', f'fib(n={n})', mem, pc, addr_a, addr_b))

    # Random programs
    random.seed(42)
    for i in range(200):
        n_instr = random.randint(1, 5)
        mem, pc = generate_random_program(n_instr)
        test_cases.append(('random', f'random_{i}', mem, pc, None, None))

    print(f"Testing {len(test_cases)} programs...")
    for prog_type, name, mem, pc, addr1, addr2 in test_cases:
        # Ground truth
        expected_mem, expected_pc, expected_steps = run(mem, pc, max_steps=200)

        # Model prediction
        pred_mem, pred_pc, trace = run_model_multistep(model, mem, pc, max_steps=200, device=device)

        if prog_type == 'fibonacci' and addr2 is not None:
            correct = (pred_mem[addr1] == expected_mem[addr1] and
                      pred_mem[addr2] == expected_mem[addr2])
        elif prog_type == 'random':
            correct = (pred_mem == expected_mem and pred_pc == expected_pc)
        elif addr1 is not None:
            correct = pred_mem[addr1] == expected_mem[addr1]
        else:
            correct = pred_mem == expected_mem

        if correct:
            successes.append(name)
        else:
            # Find first wrong step
            first_wrong_step = None
            for step_idx, (m, p) in enumerate(trace[1:], 1):
                # Run interpreter to step_idx
                m_ref = list(mem)
                p_ref = pc
                for _ in range(step_idx):
                    if p_ref < 0 or p_ref + 2 >= len(m_ref):
                        break
                    m_ref, p_ref, halted = step(m_ref, p_ref)
                    if halted:
                        break
                if m != m_ref or p != p_ref:
                    first_wrong_step = step_idx
                    break

            failures.append({
                'name': name,
                'type': prog_type,
                'mem_init': list(mem),
                'pc_init': pc,
                'expected_mem': list(expected_mem),
                'expected_pc': expected_pc,
                'pred_mem': list(pred_mem),
                'pred_pc': pred_pc,
                'expected_steps': expected_steps,
                'first_wrong_step': first_wrong_step,
                'addr1': addr1,
                'addr2': addr2,
            })

    print(f"  Failures: {len(failures)}/{len(test_cases)}")
    for f in failures[:5]:
        print(f"    FAIL: {f['name']} at step {f['first_wrong_step']}")

    return failures, successes


def compute_residual_divergence(model, failure, device='cpu'):
    """
    Step-by-step L2 divergence between model residuals and expected state.

    For each step t and layer l, compare model's residual stream
    with what a "correct" model would encode.

    Since we don't have the correct model's residuals for round2,
    we compare against the interpreter state at each step.

    Returns:
        step_layer_divergence: (n_steps, n_layers+1) array
        step_data: list of dicts with step metadata
    """
    from subleq import encode

    mem = list(failure['mem_init'])
    pc = failure['pc_init']

    # Run interpreter step by step
    interp_states = [(list(mem), pc)]
    m, p = list(mem), pc
    first_wrong = failure.get('first_wrong_step', 20)
    max_trace = min(first_wrong + 5 if first_wrong else 20, 50)

    for _ in range(max_trace):
        if p < 0 or p + 2 >= len(m):
            break
        m_new, p_new, halted = step(m, p)
        if halted:
            break
        m, p = m_new, p_new
        interp_states.append((list(m), p))

    # Run model step by step and record residuals at each step
    model.eval()
    step_data = []
    m_model = list(failure['mem_init'])
    p_model = failure['pc_init']

    for step_idx in range(min(len(interp_states), max_trace)):
        m_interp, p_interp = interp_states[step_idx]

        # Get model residuals for current state
        inp = encode(m_model, p_model).unsqueeze(0).to(device)
        with torch.no_grad():
            residuals, logits = get_r2_residuals(model, inp, device=device)

        # Compare encoded values with interpreter state
        # The model input is the current state; output predicts next state
        inp_interp = encode(m_interp, p_interp).unsqueeze(0)

        # L2 distance between what model sees and what interpreter says
        inp_model_float = inp.float().cpu()
        inp_interp_float = inp_interp.float()
        input_divergence = (inp_model_float - inp_interp_float).norm().item()

        step_data.append({
            'step': step_idx,
            'pc_model': p_model,
            'pc_interp': p_interp,
            'pc_match': p_model == p_interp,
            'mem_match': m_model == m_interp,
            'input_divergence': input_divergence,
            'residuals': {k: v.cpu().numpy() for k, v in residuals.items()},
        })

        if step_idx < max_trace - 1:
            # Advance model
            if p_model < 0 or p_model + 2 >= MEM_SIZE:
                break
            m_model, p_model = model_step(model, m_model, p_model, device)

    # Compute L2 divergence between model residuals at each step
    # relative to step 0 (initial state, no divergence yet)
    if len(step_data) > 1:
        n_steps = len(step_data)
        n_layers = max(step_data[0]['residuals'].keys()) + 1
        divergence_matrix = np.zeros((n_steps, n_layers))

        baseline_residuals = {k: v for k, v in step_data[0]['residuals'].items()}

        for t, sd in enumerate(step_data):
            for layer in range(n_layers):
                if layer in sd['residuals'] and layer in baseline_residuals:
                    diff = sd['residuals'][layer] - baseline_residuals[layer]
                    # Normalized by number of elements
                    divergence_matrix[t, layer] = np.linalg.norm(diff) / diff.size
    else:
        divergence_matrix = np.zeros((1, 7))

    return divergence_matrix, step_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 4: Failure Case Analysis")
    print(f"Device: {device}")

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(script_dir, 'checkpoints')
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'results')
    os.makedirs(args.output_dir, exist_ok=True)

    # Find checkpoints
    ckpt_paths = []
    seed0_ckpt = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')
    if os.path.exists(seed0_ckpt) and (args.seed is None or args.seed == 0):
        ckpt_paths.append((0, seed0_ckpt))
    for seed in range(1, 5):
        if args.seed is not None and args.seed != seed:
            continue
        cp = os.path.join(args.ckpt_dir, f'seed{seed}_final.pt')
        if os.path.exists(cp):
            ckpt_paths.append((seed, cp))

    all_results = {}
    for seed_id, ckpt_path in ckpt_paths:
        print(f"\n=== Seed {seed_id} ===")
        model, config = load_r2_model(ckpt_path, device)

        # Find failures
        print("Finding failure cases...")
        failures, successes = find_failure_cases(model, device=device)
        print(f"  {len(failures)} failures, {len(successes)} successes")

        # Trace top failures
        failure_traces = []
        for failure in failures[:3]:  # Trace up to 3 failures per seed
            print(f"  Tracing: {failure['name']} (first wrong at step {failure['first_wrong_step']})")
            div_matrix, step_data = compute_residual_divergence(model, failure, device=device)
            failure_traces.append({
                'failure': failure,
                'divergence_matrix': div_matrix,
                'step_data': [{k: v for k, v in sd.items() if k != 'residuals'}
                              for sd in step_data],  # exclude large residual arrays
            })

        all_results[seed_id] = {
            'ckpt': ckpt_path,
            'failures': failures,
            'n_failures': len(failures),
            'n_successes': len(successes),
            'failure_traces': failure_traces,
        }

    # Save results
    out_path = os.path.join(args.output_dir, 'phase4_failures.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")

    # JSON summary
    json_summary = {}
    for seed_id, seed_data in all_results.items():
        json_summary[str(seed_id)] = {
            'n_failures': seed_data['n_failures'],
            'n_successes': seed_data['n_successes'],
            'failure_names': [f['name'] for f in seed_data['failures']],
            'failure_types': [f['type'] for f in seed_data['failures']],
            'first_wrong_steps': [f['first_wrong_step'] for f in seed_data['failures']],
        }

    with open(os.path.join(args.output_dir, 'phase4_summary.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)
    print("JSON summary saved.")

    # Cross-seed analysis
    print("\n=== Cross-seed Failure Analysis ===")
    all_failure_names = {}
    for seed_id, seed_data in all_results.items():
        for f in seed_data['failures']:
            if f['name'] not in all_failure_names:
                all_failure_names[f['name']] = []
            all_failure_names[f['name']].append(seed_id)

    consistent = {name: seeds for name, seeds in all_failure_names.items()
                  if len(seeds) >= len(all_results) // 2}
    print(f"Consistent failures (across ≥{len(all_results)//2} seeds): {len(consistent)}")
    for name, seeds in sorted(consistent.items(), key=lambda x: -len(x[1])):
        print(f"  {name}: failed in seeds {seeds}")


if __name__ == '__main__':
    main()
