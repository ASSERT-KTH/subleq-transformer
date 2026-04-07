#!/usr/bin/env python3
"""
Phase 7: Distributional metrics for capacity scaling study.

For every model variant (constrained-LN, scaled-d32/64/128, trained-d256):
  1. Effective rank  per layer  (exp-entropy of singular value spectrum)
  2. Gini sparsity   per layer  (Gini coefficient of |activations| at pos 0)
  3. RSA (pairwise)             (Spearman rho of vectorized RDMs, N=500 inputs)

Hypothesis tests:
  H1: RSA(constrained-LN, scaled-d32) > 0  (permutation test)
  H2: RSA vs d — does similarity to constrained-LN decrease with d?  (Spearman)
  H3: Effective rank increases monotonically with d  (Spearman over d values)
  H4: constrained-LN eff-rank < scaled-d32 eff-rank  (paired t-test per layer)

Note: oracle uses a different input encoding (417 tokens, 16-bit), so RSA between
oracle and round2 models is confounded.  Effective rank IS comparable since it
is computed independently per model on its own inputs.

Saves:
  results/phase7_metrics.json
"""

import os, sys, json
import numpy as np
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))
sys.path.insert(0, script_dir)

import torch
from extract_residuals import (get_r2_residuals_batched, generate_metadata_dataset,
                                load_r2_model, get_r1_residuals)
from constrained_model import load_constrained_model

CKPT_BASE   = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')


# ── Metric helpers ────────────────────────────────────────────────────────────

def effective_rank(H):
    """
    Effective rank of activation matrix H (N, d).
    eff_rank = exp(H_entropy(σ/sum(σ)))  where σ are singular values.
    Range: [1, d].  High = many dimensions used.
    """
    H = H - H.mean(axis=0)  # center
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def gini_sparsity(H):
    """
    Gini coefficient of |activations| per sample, then mean.
    Range: [0, 1].  High = sparse (energy in few dims).
    """
    scores = []
    for h in H:
        a = np.sort(np.abs(h))
        n = len(a)
        if a.sum() < 1e-10:
            continue
        # Gini: 1 - 2*sum(i*a_i)/(n*sum(a_i)) for i=1..n (sorted asc, 1-indexed)
        indices = np.arange(1, n + 1)
        g = 1.0 - 2.0 * (indices * a).sum() / (n * a.sum())
        scores.append(g)
    return float(np.mean(scores)) if scores else float('nan')


def compute_rdm(H, max_n=500):
    """Pairwise cosine distance matrix (upper triangle as vector)."""
    if len(H) > max_n:
        H = H[:max_n]
    H = H.astype(np.float64)
    norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-10
    H_n = H / norms
    sim = H_n @ H_n.T
    np.clip(sim, -1.0, 1.0, out=sim)
    rdm = 1.0 - sim
    idx = np.triu_indices(len(H), k=1)
    return rdm[idx]


def rsa_spearman(rdm_a, rdm_b):
    rho, p = stats.spearmanr(rdm_a, rdm_b)
    return float(rho), float(p)


def permutation_test_rsa(H_a, H_b, n_perm=1000, max_n=500):
    """RSA with permutation test.  Returns (rho_observed, p_value)."""
    if len(H_a) > max_n: H_a = H_a[:max_n]
    if len(H_b) > max_n: H_b = H_b[:max_n]
    N = min(len(H_a), len(H_b))
    H_a, H_b = H_a[:N], H_b[:N]

    rdm_a = compute_rdm(H_a, max_n=N)
    rdm_b = compute_rdm(H_b, max_n=N)
    rho_obs, _ = rsa_spearman(rdm_a, rdm_b)

    null_rhos = []
    for _ in range(n_perm):
        perm = np.random.permutation(N)
        rdm_b_perm = compute_rdm(H_b[perm], max_n=N)
        r, _ = rsa_spearman(rdm_a, rdm_b_perm)
        null_rhos.append(r)
    p_val = float(np.mean(np.array(null_rhos) >= rho_obs))
    return rho_obs, p_val


# ── Load all models ───────────────────────────────────────────────────────────

def load_all_models(device):
    """Return list of (model_key, model, config, is_oracle, n_layers)."""
    entries = []

    # Oracle (round1, different input format — effective rank only)
    try:
        from model import HandCodedSUBLEQ
        oracle = HandCodedSUBLEQ()
        oracle.eval()
        oracle.to(device)
        entries.append(('oracle', oracle, {'d_model': 32, 'n_layers': 4}, True, 4))
        print("  oracle: OK")
    except Exception as e:
        print(f"  oracle: FAILED ({e})")

    # constrained-LN (round2 format, d=32, 4L)
    for s in range(3):
        p = os.path.join(CKPT_BASE, f'constrained_ln_seed{s}', 'best_model.pt')
        if os.path.exists(p):
            m, c = load_constrained_model(p, device)
            entries.append((f'constrained_ln_s{s}', m, c, False, c.get('n_layers', 4)))
    print(f"  constrained_ln: {sum(1 for e in entries if 'constrained_ln' in e[0])} seeds")

    # scaled d=32, 64, 128 (round2 format, 6L)
    for d in [32, 64, 128]:
        for s in range(3):
            p = os.path.join(CKPT_BASE, f'scaled_d{d}_seed{s}', 'best_model.pt')
            if os.path.exists(p):
                m, c = load_r2_model(p, device)
                entries.append((f'scaled_d{d}_s{s}', m, c, False, c.get('n_layers', 6)))
        n = sum(1 for e in entries if f'scaled_d{d}_s' in e[0])
        print(f"  scaled_d{d}: {n} seeds")

    # trained-d256 (round2 format, 6L) — use seeds 0-4
    p0 = os.path.join(repo_root, 'round2_trained', 'checkpoints', 'best_model.pt')
    if os.path.exists(p0):
        m, c = load_r2_model(p0, device)
        entries.append(('trained_d256_s0', m, c, False, c.get('n_layers', 6)))
    for s in range(1, 5):
        p = os.path.join(CKPT_BASE, f'seed{s}_final.pt')
        if os.path.exists(p):
            m, c = load_r2_model(p, device)
            entries.append((f'trained_d256_s{s}', m, c, False, c.get('n_layers', 6)))
    n = sum(1 for e in entries if 'trained_d256' in e[0])
    print(f"  trained_d256: {n} seeds")

    return entries


# ── Generate oracle inputs ────────────────────────────────────────────────────

def generate_oracle_inputs(n=2000, seed=42):
    """Generate round1-format inputs for oracle effective rank computation."""
    import random, sys
    sys.path.insert(0, os.path.join(repo_root, 'round1_constructed'))
    from interpreter import (step as r1_step, MEM_SIZE as R1_MEM, VALUE_OFFSET)
    from programs import make_random_program

    random.seed(seed)
    inputs = []
    attempts = 0
    while len(inputs) < n and attempts < n * 20:
        attempts += 1
        try:
            mem, pc = make_random_program(n_instr=random.randint(1, 8))
        except Exception:
            continue
        if pc < 0 or pc + 2 >= R1_MEM:
            continue
        _, _, halted = r1_step(mem, pc)
        if halted:
            continue
        tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
        inputs.append(torch.tensor(tokens, dtype=torch.long))
    if not inputs:
        return None
    return torch.stack(inputs[:n])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-data',  type=int, default=2000, help='For eff rank & Gini')
    parser.add_argument('--n-rsa',   type=int, default=500,  help='For RSA (N×N RDM)')
    parser.add_argument('--n-perm',  type=int, default=1000, help='Permutations for H1')
    args = parser.parse_args()

    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 7: Distributional metrics | device={device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nGenerating {args.n_data} round2 inputs...")
    r2_inputs, _ = generate_metadata_dataset(n=args.n_data, seed=42)
    # RSA subset
    rsa_inputs = r2_inputs[:args.n_rsa]

    print(f"Generating oracle inputs...")
    oracle_inputs = generate_oracle_inputs(n=args.n_data, seed=42)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    model_entries = load_all_models(device)

    # ── Extract residuals ─────────────────────────────────────────────────────
    print("\nExtracting residuals...")
    model_residuals = {}   # key -> {layer_idx: (N, d) numpy}

    for key, model, config, is_oracle, n_layers in model_entries:
        print(f"  {key}...")
        with torch.no_grad():
            if is_oracle:
                if oracle_inputs is None:
                    print(f"    skipped (no oracle inputs)")
                    continue
                res = get_r1_residuals(model, oracle_inputs, device=device)
                # pos 0 only
                model_residuals[key] = {li: res[li][:, 0, :].numpy() for li in res}
            else:
                res, _ = get_r2_residuals_batched(model, r2_inputs, device=device)
                model_residuals[key] = {li: res[li][:, 0, :].numpy() for li in res}

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing effective rank and Gini sparsity...")
    eff_rank_results = {}   # key -> {layer_idx: float}
    gini_results     = {}   # key -> {layer_idx: float}

    for key, res_by_layer in model_residuals.items():
        eff_rank_results[key] = {}
        gini_results[key]     = {}
        for li, H in sorted(res_by_layer.items()):
            eff_rank_results[key][li] = effective_rank(H)
            gini_results[key][li]     = gini_sparsity(H)
        n_l = max(res_by_layer.keys())
        print(f"  {key}: eff_rank L0={eff_rank_results[key].get(0, 0):.2f} → "
              f"L{n_l}={eff_rank_results[key].get(n_l, 0):.2f}  |  "
              f"Gini L0={gini_results[key].get(0, 0):.3f} → "
              f"L{n_l}={gini_results[key].get(n_l, 0):.3f}")

    # ── RSA (round2 models only) ───────────────────────────────────────────────
    print(f"\nComputing RSA (N={args.n_rsa}, {args.n_perm} permutations)...")
    # RSA residuals (first n_rsa rows)
    rsa_residuals = {}
    for key, res_by_layer in model_residuals.items():
        if key == 'oracle':
            continue
        rsa_residuals[key] = {li: H[:args.n_rsa] for li, H in res_by_layer.items()}

    # Group by model family and find reference (constrained_ln mean across seeds)
    # Compare constrained_ln (each seed) vs scaled models (each seed)
    # For paired comparison, use per-seed RSA at matched layers

    rsa_results = {}  # (key_a, key_b) -> {layer_pair: (rho, p_value)}

    # Build list of (key_a, key_b) pairs of interest
    # Use best-layer (layer with highest probe R² would be ideal, but use last layer here)
    # For now: compare all constrained_ln seeds vs all scaled_d32 seeds at each layer pair

    r2_keys = [k for k in rsa_residuals.keys()]

    # Pre-compute RDMs per key per layer
    print("  Pre-computing RDMs...")
    rdm_cache = {}  # (key, layer) -> rdm_vec
    for key, res_by_layer in rsa_residuals.items():
        for li, H in res_by_layer.items():
            rdm_cache[(key, li)] = compute_rdm(H, max_n=args.n_rsa)

    # Pairwise RSA between constrained_ln and each other model family
    # Use a representative seed (s0) and matched layers
    ref_keys    = [k for k in r2_keys if k.startswith('constrained_ln')]
    target_keys = [k for k in r2_keys if not k.startswith('constrained_ln')]

    for ref_key in ref_keys:
        ref_layers = sorted(rsa_residuals[ref_key].keys())
        for tgt_key in target_keys:
            tgt_layers = sorted(rsa_residuals[tgt_key].keys())
            # Matched layer pairs by relative depth
            n_ref = len(ref_layers)
            n_tgt = len(tgt_layers)
            matched = []
            for i, rl in enumerate(ref_layers):
                tl_idx = round(i * (n_tgt - 1) / max(n_ref - 1, 1))
                tl = tgt_layers[min(tl_idx, n_tgt - 1)]
                matched.append((rl, tl))

            pair_key = f"{ref_key}_vs_{tgt_key}"
            rsa_results[pair_key] = {}
            for rl, tl in matched:
                rdm_a = rdm_cache.get((ref_key, rl))
                rdm_b = rdm_cache.get((tgt_key, tl))
                if rdm_a is None or rdm_b is None:
                    continue
                rho, p = rsa_spearman(rdm_a, rdm_b)
                rsa_results[pair_key][f"ref_L{rl}_tgt_L{tl}"] = {'rho': rho, 'p': p}

    # H1: permutation test for constrained_ln_s0 vs scaled_d32_s0 at best layer
    print(f"\nH1: permutation test (constrained_ln_s0 vs scaled_d32_s0)...")
    h1_results = {}
    if 'constrained_ln_s0' in rsa_residuals and 'scaled_d32_s0' in rsa_residuals:
        ref_layers = sorted(rsa_residuals['constrained_ln_s0'].keys())
        tgt_layers = sorted(rsa_residuals['scaled_d32_s0'].keys())
        for rl, tl in zip(ref_layers,
                          [tgt_layers[round(i*(len(tgt_layers)-1)/max(len(ref_layers)-1,1))]
                           for i in range(len(ref_layers))]):
            H_a = rsa_residuals['constrained_ln_s0'][rl]
            H_b = rsa_residuals['scaled_d32_s0'][tl]
            rho_obs, p_val = permutation_test_rsa(H_a, H_b, n_perm=args.n_perm)
            h1_results[f"ref_L{rl}_tgt_L{tl}"] = {'rho': rho_obs, 'p': p_val}
            print(f"  constrained_ln_s0 L{rl} vs scaled_d32_s0 L{tl}: rho={rho_obs:.4f} p={p_val:.3f}")

    # H2: Does constrained_ln vs scaled_dX RSA decrease with d?
    print("\nH2: RSA vs d (constrained_ln_s0 at final layer vs scaled_dX_s0)...")
    h2_data = {}
    for d in [32, 64, 128, 256]:
        tgt = f'scaled_d{d}_s0' if d < 256 else 'trained_d256_s0'
        if 'constrained_ln_s0' not in rsa_residuals or tgt not in rsa_residuals:
            continue
        ref_layers = sorted(rsa_residuals['constrained_ln_s0'].keys())
        tgt_layers = sorted(rsa_residuals[tgt].keys())
        rl = ref_layers[-1]  # final layer of constrained_ln
        tl = tgt_layers[-1]  # final layer of target
        rdm_a = rdm_cache.get(('constrained_ln_s0', rl))
        rdm_b = rdm_cache.get((tgt, tl))
        if rdm_a is not None and rdm_b is not None:
            rho, p = rsa_spearman(rdm_a, rdm_b)
            h2_data[d] = {'rho': rho, 'p': p}
            print(f"  d={d}: rho={rho:.4f} p={p:.3f}")

    if len(h2_data) >= 3:
        ds = sorted(h2_data.keys())
        rhos = [h2_data[d]['rho'] for d in ds]
        log_ds = np.log2(ds)
        rho_trend, p_trend = stats.spearmanr(log_ds, rhos)
        h2_test = {'spearman_rho_of_rhos_vs_logd': rho_trend, 'p': p_trend,
                   'data': h2_data, 'ds': ds}
        print(f"  Trend: rho={rho_trend:.4f} p={p_trend:.3f}")
    else:
        h2_test = {'data': h2_data}

    # H3: Effective rank increases with d
    print("\nH3: Effective rank vs d...")
    h3_data = {}
    for d in [32, 64, 128, 256]:
        keys = ([f'scaled_d{d}_s{s}' for s in range(3)] if d < 256
                else [f'trained_d256_s{s}' for s in range(5)])
        vals = []
        for k in keys:
            if k in eff_rank_results:
                layers = sorted(eff_rank_results[k].keys())
                # mean over all layers (excluding embed L0)
                mid_layers = layers[1:]
                vals.extend([eff_rank_results[k][l] for l in mid_layers])
        if vals:
            h3_data[d] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                           'n': len(vals)}
            print(f"  d={d}: eff_rank={np.mean(vals):.2f}±{np.std(vals):.2f}")

    if len(h3_data) >= 3:
        ds = sorted(h3_data.keys())
        means = [h3_data[d]['mean'] for d in ds]
        rho_h3, p_h3 = stats.spearmanr(np.log2(ds), means)
        h3_test = {'spearman_rho': rho_h3, 'p_one_sided': float(p_h3/2),
                   'data': h3_data, 'note': 'H_a: rho > 0 (rank increases with d)'}
        print(f"  Trend: rho={rho_h3:.4f} p(one-sided)={p_h3/2:.4f}")
    else:
        h3_test = {'data': h3_data}

    # H4: constrained-LN eff-rank < scaled-d32 eff-rank (same d, different arch)
    print("\nH4: constrained-LN vs scaled-d32 eff-rank (t-test per layer)...")
    h4_results = {}
    cln_keys = [k for k in eff_rank_results if k.startswith('constrained_ln')]
    sd32_keys = [k for k in eff_rank_results if k.startswith('scaled_d32')]
    if cln_keys and sd32_keys:
        # Get common layer count (cln has 5 layers, scaled_d32 has 7)
        cln_layers  = sorted(eff_rank_results[cln_keys[0]].keys())
        sd32_layers = sorted(eff_rank_results[sd32_keys[0]].keys())
        # matched by relative depth
        n_cln, n_sd32 = len(cln_layers), len(sd32_layers)
        for i, cl in enumerate(cln_layers):
            sl_idx = round(i * (n_sd32 - 1) / max(n_cln - 1, 1))
            sl = sd32_layers[min(sl_idx, n_sd32 - 1)]
            cln_vals  = [eff_rank_results[k][cl]  for k in cln_keys  if cl in eff_rank_results[k]]
            sd32_vals = [eff_rank_results[k][sl] for k in sd32_keys if sl in eff_rank_results[k]]
            if len(cln_vals) >= 2 and len(sd32_vals) >= 2:
                t, p = stats.ttest_ind(cln_vals, sd32_vals, alternative='less')
                alpha_bonf = 0.05 / len(cln_layers)
                h4_results[f'cln_L{cl}_vs_sd32_L{sl}'] = {
                    'cln_mean': float(np.mean(cln_vals)),
                    'sd32_mean': float(np.std(sd32_vals)),
                    't': float(t), 'p': float(p),
                    'sig_bonf': bool(p < alpha_bonf),
                    'alpha_bonf': alpha_bonf,
                }
                print(f"  L{cl} vs L{sl}: cln={np.mean(cln_vals):.2f}, "
                      f"sd32={np.mean(sd32_vals):.2f}, p={p:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    def jsonify(x):
        if isinstance(x, dict):
            return {str(k): jsonify(v) for k, v in x.items()}
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [jsonify(v) for v in x]
        return x

    output = {
        'effective_rank': jsonify(eff_rank_results),
        'gini_sparsity':  jsonify(gini_results),
        'rsa':            jsonify(rsa_results),
        'hypothesis_tests': {
            'H1_permutation_constrained_vs_scaled_d32': jsonify(h1_results),
            'H2_rsa_vs_d':                              jsonify(h2_test),
            'H3_effrank_vs_d':                          jsonify(h3_test),
            'H4_constrained_vs_scaled_d32_effrank':     jsonify(h4_results),
        },
    }
    out_path = os.path.join(RESULTS_DIR, 'phase7_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
