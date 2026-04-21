#!/usr/bin/env python3
"""
Phase 8: Circuit Tracing.

Compares the computational circuits learned by trained SUBLEQ transformers
against the known hardcoded circuit structure.

For each model (constrained_ln, base trained):
  A. Attention pattern analysis
     - Fetch head scores  (do heads implement content-addressed reading?)
     - Broadcast head scores (do heads broadcast values to all positions?)
     - Attention entropy  (sharpness of attention)
  B. OV circuit analysis
     - W_OV singular value spectra per head/layer
  C. FFN neuron concept correlation
     - Pearson r between each neuron and SUBLEQ quantities (pc, a_addr, b_addr,
       delta=mem[b]-mem[a], branch_taken)
  D. Circuit completeness
     - Ablation study: does the hypothesised circuit recover ≥90% of accuracy?
  E. Attribution graph
     - Build full frozen-Jacobian graph and summarise path strengths

Output:
  results/phase8_circuit_tracing.json
  figures/phase8_*.png
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)

for p in [_SCRIPT_DIR, _REPO,
          os.path.join(_REPO, 'round2_trained'),
          os.path.join(_REPO, 'round1_constructed')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from subleq.data import generate_batch
from subleq.interpreter import MEM_SIZE, step as r2_step
from subleq.tokenizer import encode, decode, SEQ_LEN
from subleq.programs import generate_random_state

from circuit_tracing.extract_activations import get_all_activations
from circuit_tracing.metrics import (
    fetch_head_score, broadcast_head_score, attention_entropy,
    neuron_concept_correlation, top_neuron_concepts,
    mean_head_activations, circuit_completeness, summarise_head_roles,
)
from circuit_tracing.attribution import (
    compute_ov_circuits, ov_eigenspectrum, build_attribution_graph,
)
from circuit_tracing.visualize import (
    plot_head_role_heatmap, plot_neuron_concept_correlations,
    plot_ov_spectrum, plot_model_comparison, plot_circuit_completeness,
    plot_attention_patterns,
)


# ─── Data generation ──────────────────────────────────────────────────────────

def generate_eval_dataset(n=1000, seed=42):
    """
    Generate N random valid SUBLEQ states for the round2 task (32 cells, 8-bit).

    Returns
    -------
    inputs      : LongTensor (N, SEQ_LEN)
    targets     : LongTensor (N, SEQ_LEN)
    meta        : dict with keys 'pc', 'a_addr', 'b_addr', 'c_addr',
                  'mem_a', 'mem_b', 'delta', 'new_val', 'branch_taken'
                  — each a LongTensor (N,) or BoolTensor (N,)
    """
    random.seed(seed)
    inputs_list, targets_list = [], []
    meta = {k: [] for k in ['pc', 'a_addr', 'b_addr', 'c_addr',
                              'mem_a', 'mem_b', 'delta', 'new_val', 'branch_taken']}
    attempts = 0
    while len(inputs_list) < n and attempts < n * 20:
        attempts += 1
        mem, pc = generate_random_state()
        if pc < 0 or pc + 2 >= MEM_SIZE:
            continue
        a = mem[pc];  b = mem[pc + 1];  c = mem[pc + 2]
        if not (0 <= a < MEM_SIZE and 0 <= b < MEM_SIZE):
            continue
        new_mem, new_pc, halted = r2_step(mem, pc)
        if halted:
            continue

        inp = encode(mem, pc)
        tgt = encode(new_mem, new_pc)
        inputs_list.append(inp)
        targets_list.append(tgt)

        from subleq.interpreter import clamp
        delta = mem[b] - mem[a]
        new_val = clamp(delta)
        meta['pc'].append(pc)
        meta['a_addr'].append(a)
        meta['b_addr'].append(b)
        meta['c_addr'].append(c)
        meta['mem_a'].append(mem[a])
        meta['mem_b'].append(mem[b])
        meta['delta'].append(delta)
        meta['new_val'].append(new_val)
        meta['branch_taken'].append(1 if new_val <= 0 else 0)

    inputs = torch.stack(inputs_list)
    targets = torch.stack(targets_list)
    meta = {k: torch.tensor(v, dtype=torch.long) for k, v in meta.items()}
    meta['branch_taken'] = meta['branch_taken'].bool()
    return inputs, targets, meta


# ─── Model loading ────────────────────────────────────────────────────────────

def load_constrained_ln(ckpt_dir, seed=0, device='cpu'):
    from constrained_model import load_constrained_model
    ckpt = os.path.join(ckpt_dir, f'constrained_ln_seed{seed}', 'best_model.pt')
    model, config = load_constrained_model(ckpt, device=device)
    return model, config


def load_base_trained(ckpt_dir, seed=0, device='cpu'):
    from subleq.model import MiniSUBLEQTransformer
    ckpt_path = os.path.join(ckpt_dir, f'seed{seed}', 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 64),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', 256),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, config


# ─── Per-model analysis ───────────────────────────────────────────────────────

def analyse_model(model, inputs, targets, meta, model_name, device,
                  results_dir, figures_dir, n_completeness_examples=500):
    """Run all circuit tracing analyses on one model."""
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)

    n_layers = len(model.layers)
    T = inputs.shape[1]
    random_baseline = 1.0 / T

    result = {'model': model_name, 'n_layers': n_layers}

    # ── A. Attention pattern analysis ─────────────────────────────────────
    print(f'  [A] Attention patterns...')
    acts = get_all_activations(model, inputs)

    fetch_scores_all = []
    broadcast_scores_all = []
    entropy_all = []

    # For fetch scores, we need the expected target for each operand
    # In round2: token position of mem[X] = X + 1
    # Fetch-a head: attend from pos 0 to pos a_addr+1
    # Fetch-b head: attend from pos 0 to pos b_addr+1
    target_a = (meta['a_addr'] + 1).to(device)
    target_b = (meta['b_addr'] + 1).to(device)
    target_c = (meta['c_addr'] + 1).to(device)

    # For fetch-mem[a] head (L2 in hardcoded): attend from pos 0 to pos mem[a]+1
    # We use a_addr+1 as proxy since the "a" head established a_addr at pos 0
    target_mema = (meta['mem_a'].clamp(0, MEM_SIZE - 1) + 1).to(device)
    target_memb = (meta['mem_b'].clamp(0, MEM_SIZE - 1) + 1).to(device)

    fetch_targets = {
        'a_addr': target_a,
        'b_addr': target_b,
        'c_addr': target_c,
        'mem_a': target_mema,
        'mem_b': target_memb,
    }

    layer_attn_results = {}
    for i in range(n_layers):
        aw = acts[f'attn_weights_{i}'].cpu()   # (N, H, T, T)
        H = aw.shape[1]

        # Fetch scores for each operand type
        fetch_by_target = {}
        for tname, tpos in fetch_targets.items():
            sc, bl = fetch_head_score(aw, tpos.cpu(), query_pos=0)
            fetch_by_target[tname] = sc.tolist()

        # Best fetch score across all target types (per head)
        best_fetch = np.stack(list(fetch_by_target.values())).max(0)   # (H,)

        bc_scores, _ = broadcast_head_score(aw, src_pos=0, exclude_self=True)
        ent = attention_entropy(aw)

        layer_attn_results[i] = {
            'fetch_scores': {k: v for k, v in fetch_by_target.items()},
            'best_fetch': best_fetch.tolist(),
            'broadcast_scores': bc_scores.tolist(),
            'entropy': ent.tolist(),
        }
        fetch_scores_all.append(best_fetch)
        broadcast_scores_all.append(bc_scores)
        entropy_all.append(ent)

    result['attention'] = layer_attn_results

    fetch_arr = np.stack(fetch_scores_all)      # (L, H)
    bcast_arr = np.stack(broadcast_scores_all)  # (L, H)
    roles = summarise_head_roles(fetch_arr, bcast_arr, random_baseline)
    result['head_roles'] = roles
    print(f'    Head roles: {roles}')

    # Plot head roles
    plot_head_role_heatmap(
        fetch_arr, bcast_arr, random_baseline,
        model_name=model_name,
        save_path=os.path.join(figures_dir, f'phase8_head_roles_{model_name}.png'))

    # Plot attention patterns for a single example (first in batch)
    for i in range(n_layers):
        plot_attention_patterns(
            acts[f'attn_weights_{i}'], layer_idx=i, example_idx=0,
            title_prefix=f'{model_name} ',
            save_path=os.path.join(figures_dir,
                                   f'phase8_attn_L{i}_{model_name}.png'))

    # ── B. OV circuit analysis ─────────────────────────────────────────────
    print(f'  [B] OV circuits...')
    ov_all = compute_ov_circuits(model)
    ov_results = {}
    for i in range(n_layers):
        W_OV = ov_all[i].cpu()
        svs = ov_eigenspectrum(W_OV).tolist()   # (H, D)
        ov_results[i] = {
            'singular_values': svs,
            'top_sv': [float(max(h_svs)) for h_svs in svs],
        }
        plot_ov_spectrum(W_OV, layer_idx=i, model_name=model_name,
                         save_path=os.path.join(figures_dir,
                                                f'phase8_ov_L{i}_{model_name}.png'))
    result['ov_circuits'] = ov_results

    # ── C. FFN neuron concept correlation ─────────────────────────────────
    print(f'  [C] FFN neuron concepts...')
    concept_dict_float = {
        'pc':           meta['pc'].float(),
        'a_addr':       meta['a_addr'].float(),
        'b_addr':       meta['b_addr'].float(),
        'mem_a':        meta['mem_a'].float(),
        'mem_b':        meta['mem_b'].float(),
        'delta':        meta['delta'].float(),
        'branch_taken': meta['branch_taken'].float(),
    }

    ffn_corr_results = {}
    for i in range(n_layers):
        ffn_pre = acts[f'ffn_pre_{i}'].cpu()   # (N, T, d_ff)
        layer_corr = {}
        for tpos in [0]:   # focus on PC token; extend to other positions if needed
            top = top_neuron_concepts(ffn_pre, concept_dict_float,
                                      token_pos=tpos, top_k=5)
            layer_corr[f'pos{tpos}'] = {
                name: {
                    'top_neurons': v['top_neurons'],
                    'top_corr': [round(float(c), 4) for c in v['top_corr']],
                }
                for name, v in top.items()
            }

            # Also store full correlation arrays for plotting
            all_corrs = {name: neuron_concept_correlation(ffn_pre, vals, tpos)
                         for name, vals in concept_dict_float.items()}
            plot_neuron_concept_correlations(
                all_corrs, layer_idx=i, token_pos=tpos,
                model_name=model_name,
                save_path=os.path.join(figures_dir,
                                       f'phase8_ffn_L{i}_pos{tpos}_{model_name}.png'))
        ffn_corr_results[i] = layer_corr

    result['ffn_concepts'] = ffn_corr_results

    # ── D. Circuit completeness ────────────────────────────────────────────
    print(f'  [D] Circuit completeness...')

    # Build mean head activations for ablation
    subset_inputs = inputs[:n_completeness_examples]
    subset_targets = targets[:n_completeness_examples]
    dl = DataLoader(TensorDataset(subset_inputs), batch_size=64, shuffle=False)
    mha = mean_head_activations(model, dl, device=device)

    # Changed positions mask: PC token (pos 0) + mem[b] token (pos b+1)
    b_addrs_sub = meta['b_addr'][:n_completeness_examples]
    changed_mask = torch.zeros(n_completeness_examples, T, dtype=torch.bool)
    changed_mask[:, 0] = True                          # PC always changes
    for n_idx, b in enumerate(b_addrs_sub):
        changed_mask[n_idx, b.item() + 1] = True       # mem[b] changes

    # Define circuit hypothesis based on head roles
    circuit_candidates = _build_circuit_hypotheses(roles, n_layers)
    completeness_results = []
    for label, circuit_heads in circuit_candidates.items():
        comp = circuit_completeness(
            model, subset_inputs, subset_targets,
            circuit_heads, mha, changed_mask.to(device), device)
        comp['label'] = label
        completeness_results.append(comp)
        print(f'    {label}: full={comp["full_acc"]:.3f}  circuit={comp["circuit_acc"]:.3f}  '
              f'ratio={comp["ratio"]:.3f}')

    result['completeness'] = completeness_results
    plot_circuit_completeness(
        completeness_results,
        save_path=os.path.join(figures_dir, f'phase8_completeness_{model_name}.png'))

    # ── E. Attribution graph ───────────────────────────────────────────────
    print(f'  [E] Attribution graph...')
    # Run on a small subset to keep memory tractable
    graph_inputs = inputs[:50]
    graph = build_attribution_graph(model, graph_inputs, include_jacobians=True)

    # Summarise: for each layer, the total attribution flowing through each
    # (query_pos, key_pos) pair
    graph_summary = {}
    for i in range(n_layers):
        attn_attr = graph[f'attn_attr_{i}'].detach().cpu().numpy()   # (H, T, T)
        ffn_in = graph[f'ffn_in_attr_{i}'].detach().cpu().numpy()    # (T, d_ff)
        ffn_out = graph[f'ffn_out_attr_{i}'].detach().cpu().numpy()  # (T, d_ff)

        # Top-5 (query, key) pairs by total attribution across heads
        combined = attn_attr.sum(0)   # (T, T)
        flat_top = np.argsort(combined.flatten())[::-1][:10]
        top_edges = [(int(idx // T), int(idx % T), float(combined.flatten()[idx]))
                     for idx in flat_top]

        graph_summary[i] = {
            'top_attn_edges': top_edges,     # [(q, k, weight), ...]
            'mean_ffn_in_norm': float(ffn_in.mean()),
            'mean_ffn_out_norm': float(ffn_out.mean()),
        }

    result['attribution_graph'] = graph_summary

    return result


def _build_circuit_hypotheses(roles, n_layers):
    """
    Build circuit hypotheses based on identified head roles.

    Returns
    -------
    dict  label → {layer_idx: set_of_head_indices}
    """
    circuits = {}

    # Hypothesis 1: all heads (upper bound)
    all_heads = {i: set(range(len(roles[i]))) for i in range(n_layers)}
    circuits['all_heads'] = all_heads

    # Hypothesis 2: fetch + broadcast heads only
    fb_heads = {}
    for i, layer_roles in enumerate(roles):
        keep = {h for h, r in enumerate(layer_roles) if r in ('fetch', 'broadcast', 'mixed')}
        fb_heads[i] = keep if keep else set()
    circuits['fetch_broadcast'] = fb_heads

    # Hypothesis 3: fetch heads only (L1 and L2 analogs)
    fetch_only = {}
    for i, layer_roles in enumerate(roles):
        keep = {h for h, r in enumerate(layer_roles) if r in ('fetch', 'mixed')}
        fetch_only[i] = keep
    circuits['fetch_only'] = fetch_only

    # Hypothesis 4: broadcast heads only (L3 analogs)
    bcast_only = {}
    for i, layer_roles in enumerate(roles):
        keep = {h for h, r in enumerate(layer_roles) if r in ('broadcast', 'mixed')}
        bcast_only[i] = keep
    circuits['broadcast_only'] = bcast_only

    return circuits


# ─── Reference: hardcoded model ───────────────────────────────────────────────

def analyse_hardcoded_reference(figures_dir, n=500, seed=0):
    """
    Run attention metrics on the round1 HandCodedSUBLEQ model to get reference scores.
    Returns a dict with fetch/broadcast scores per layer/head.
    """
    try:
        from round1_constructed.model import HandCodedSUBLEQ, SEQ_LEN as R1_SEQ_LEN
        from round1_constructed.interpreter import (
            MEM_SIZE as R1_MEM, VALUE_OFFSET, VALUE_MIN, VALUE_MAX)
        import random as rnd

        model = HandCodedSUBLEQ()
        model.eval()

        rnd.seed(seed)
        inputs, a_addrs, b_addrs, c_addrs = [], [], [], []
        for _ in range(n * 5):
            if len(inputs) >= n:
                break
            pc = rnd.randint(0, R1_MEM - 4)
            mem = [0] * R1_MEM
            a = rnd.randint(0, R1_MEM - 1)
            b = rnd.randint(0, R1_MEM - 1)
            c = rnd.randint(0, R1_MEM - 3)
            mem[pc] = a; mem[pc+1] = b; mem[pc+2] = c
            for k in range(R1_MEM):
                if k not in {pc, pc+1, pc+2}:
                    mem[k] = rnd.randint(VALUE_MIN, VALUE_MAX)
            tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
            inputs.append(torch.tensor(tokens, dtype=torch.long))
            a_addrs.append(a); b_addrs.append(b); c_addrs.append(c)

        inp_t = torch.stack(inputs)
        acts = get_all_activations(model, inp_t)
        T = R1_SEQ_LEN
        random_baseline = 1.0 / T

        ref_result = {'model': 'hardcoded_r1', 'T': T, 'random_baseline': random_baseline}
        fetch_arr = []
        bcast_arr = []

        for i in range(4):
            aw = acts[f'attn_weights_{i}'].cpu()
            ta = torch.tensor(a_addrs) + 1
            tb = torch.tensor(b_addrs) + 1
            tc = torch.tensor(c_addrs) + 1

            sa, _ = fetch_head_score(aw, ta, query_pos=0)
            sb, _ = fetch_head_score(aw, tb, query_pos=0)
            sc, _ = fetch_head_score(aw, tc, query_pos=0)
            best_f = np.stack([sa, sb, sc]).max(0)

            bc, _ = broadcast_head_score(aw, src_pos=0, exclude_self=True)

            fetch_arr.append(best_f)
            bcast_arr.append(bc)
            ref_result[f'layer_{i}'] = {
                'fetch_a': sa.tolist(), 'fetch_b': sb.tolist(), 'fetch_c': sc.tolist(),
                'broadcast': bc.tolist(),
            }

        plot_head_role_heatmap(
            np.stack(fetch_arr), np.stack(bcast_arr), random_baseline,
            model_name='hardcoded_r1',
            save_path=os.path.join(figures_dir, 'phase8_head_roles_hardcoded_r1.png'))

        return ref_result

    except Exception as e:
        print(f'  [WARN] Could not run hardcoded reference analysis: {e}')
        return {'error': str(e)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', default='checkpoints')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--figures-dir', default=None)
    parser.add_argument('--n-examples', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--models', nargs='+',
                        default=['constrained_ln'],
                        choices=['constrained_ln', 'base_trained'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-hardcoded', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(args.output_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print(f'Device: {args.device}')
    print(f'Generating {args.n_examples} eval examples...')
    inputs, targets, meta = generate_eval_dataset(n=args.n_examples, seed=42)

    all_results = {}

    # ── Hardcoded reference ────────────────────────────────────────────────
    if not args.skip_hardcoded:
        print('\n=== Hardcoded model (reference) ===')
        ref = analyse_hardcoded_reference(figures_dir, n=500)
        all_results['hardcoded_reference'] = ref

    # ── Trained models ────────────────────────────────────────────────────
    loaders = {
        'constrained_ln': load_constrained_ln,
        'base_trained': load_base_trained,
    }

    for model_type in args.models:
        loader = loaders[model_type]
        seed_results = []

        for seed in args.seeds:
            mname = f'{model_type}_seed{seed}'
            print(f'\n=== {mname} ===')
            try:
                model, config = loader(args.ckpt_dir, seed=seed, device=args.device)
                print(f'  Config: {config}')
                print(f'  Params: {model.count_params():,}')

                res = analyse_model(
                    model, inputs, targets, meta,
                    model_name=mname,
                    device=args.device,
                    results_dir=args.output_dir,
                    figures_dir=figures_dir,
                )
                seed_results.append(res)

            except FileNotFoundError as e:
                print(f'  [SKIP] Checkpoint not found: {e}')
            except Exception as e:
                import traceback
                print(f'  [ERROR] {e}')
                traceback.print_exc()

        if seed_results:
            all_results[model_type] = seed_results

    # ── Cross-model comparison figure ─────────────────────────────────────
    if len(all_results) >= 2:
        _make_comparison_figure(all_results, figures_dir)

    # ── Save results ──────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, 'phase8_circuit_tracing.json')
    with open(out_path, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2)
    print(f'\nResults saved to {out_path}')


def _make_comparison_figure(all_results, figures_dir):
    """Produce side-by-side fetch/broadcast comparison for all models."""
    import matplotlib
    matplotlib.use('Agg')

    comparison = {}
    random_baseline = None

    for model_type, results in all_results.items():
        if model_type == 'hardcoded_reference':
            continue
        if not isinstance(results, list) or not results:
            continue
        # Average across seeds
        r = results[0]   # use seed 0
        n_layers = r['n_layers']
        H = len(r['attention'][0]['best_fetch'])
        fetch = np.array([r['attention'][i]['best_fetch'] for i in range(n_layers)])
        bcast = np.array([r['attention'][i]['broadcast_scores'] for i in range(n_layers)])
        comparison[model_type] = {'fetch': fetch, 'broadcast': bcast}
        T_approx = SEQ_LEN
        random_baseline = 1.0 / T_approx

    if comparison and random_baseline:
        from circuit_tracing.visualize import plot_model_comparison
        plot_model_comparison(
            comparison, random_baseline,
            save_path=os.path.join(figures_dir, 'phase8_model_comparison.png'))


def _json_safe(obj):
    """Recursively convert numpy/torch types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, bool):
        return obj
    return obj


if __name__ == '__main__':
    main()
