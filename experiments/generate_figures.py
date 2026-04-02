#!/usr/bin/env python3
"""
Generate all figures for the SUBLEQ mechanistic interpretability paper.

Figures:
  fig1_oracle_probe_heatmap.png    - Oracle probe heatmap (R² / accuracy × layer)
  fig2_trained_probe_heatmap.png   - Trained model probe heatmap (5-seed mean)
  fig3_probe_comparison.png        - Oracle vs trained side-by-side
  fig4_oracle_patch_heatmap.png    - Oracle activation patching heatmap
  fig5_trained_patch_heatmap.png   - Trained model patching heatmap
  fig6_diagnostic_table.png        - 2×2 (probe presence × patching use) diagnostic
  fig7_localization.png            - Dimensional localization curves
  fig8_dynamics.png                - Training dynamics (accuracy vs fraction)
  fig9_failure_trace.png           - Step-by-step probe trace for failure case
  fig10_constrained_probe.png      - Probe heatmaps: oracle vs constrained-LN vs constrained-noLN vs trained
  fig11_constrained_patch.png     - Patching comparison: oracle vs constrained-LN vs trained
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


TARGETS = ['pc', 'mem_a', 'mem_b', 'delta', 'branch_taken']
TARGET_LABELS = ['PC', 'mem[a]', 'mem[b]', 'Δ', 'branch']

# Color scheme
CMAP_PROBE = 'YlOrRd'
CMAP_PATCH = 'Blues'

FIG_DPI = 150


# ── Helper ────────────────────────────────────────────────────────────────────

def savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Fig 1 & 2: Probe heatmaps ────────────────────────────────────────────────

def make_probe_heatmap(data_matrix, row_labels, col_labels, title,
                        vmin=0.0, vmax=1.0, cmap=CMAP_PROBE,
                        is_classification_row=None):
    """
    data_matrix: (n_targets, n_layers)
    is_classification_row: list of bool, one per row (for label formatting)
    """
    n_rows, n_cols = data_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.9), max(3, n_rows * 0.7)))

    # Clip for display (negative R² → 0 for colormap, but show actual value)
    display = np.clip(data_matrix, vmin, vmax)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Layer')
    ax.set_title(title, fontsize=11)

    # Annotate with values
    for i in range(n_rows):
        for j in range(n_cols):
            val = data_matrix[i, j]
            if val < -10:
                txt = '—'
            elif is_classification_row and is_classification_row[i]:
                txt = f'{val:.2f}'
            else:
                txt = f'{val:.2f}'
            color = 'white' if display[i, j] > (vmin + vmax) * 0.65 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label='Accuracy' if is_classification_row else 'R²')
    return fig


def fig1_oracle_probe(phase1_json_path, output_path):
    with open(phase1_json_path) as f:
        data = json.load(f)

    probe_summary = data.get('probe_summary', {})
    n_layers = max(int(k) for k in probe_summary.keys()) + 1

    matrix = np.full((len(TARGETS), n_layers), np.nan)
    is_cls = [t == 'branch_taken' for t in TARGETS]

    for j in range(n_layers):
        layer_data = probe_summary.get(str(j), {})
        for i, tname in enumerate(TARGETS):
            v = layer_data.get(tname)
            if v is not None:
                matrix[i, j] = v

    col_labels = [f'L{j}' for j in range(n_layers)]
    fig = make_probe_heatmap(matrix, TARGET_LABELS, col_labels,
                              'Oracle Model: Linear Probe Accuracy / R²',
                              is_classification_row=is_cls)
    savefig(fig, output_path)


def fig2_trained_probe(phase2_json_path, output_path):
    with open(phase2_json_path) as f:
        data = json.load(f)

    probe_means = data.get('probe_means', {})
    n_layers = 7  # 0=embed + 6 transformer layers

    matrix = np.full((len(TARGETS), n_layers), np.nan)
    std_matrix = np.full((len(TARGETS), n_layers), np.nan)
    is_cls = [t == 'branch_taken' for t in TARGETS]

    for i, tname in enumerate(TARGETS):
        tdata = probe_means.get(tname, {})
        for j in range(n_layers):
            entry = tdata.get(str(j))
            if entry:
                matrix[i, j] = entry['mean']
                std_matrix[i, j] = entry['std']

    col_labels = [f'L{j}' for j in range(n_layers)]
    fig = make_probe_heatmap(matrix, TARGET_LABELS, col_labels,
                              'Trained Model: Linear Probe (mean ± std, 5 seeds)',
                              is_classification_row=is_cls)
    # Add std annotations
    ax = fig.axes[0]
    for i in range(len(TARGETS)):
        for j in range(n_layers):
            if not np.isnan(std_matrix[i, j]) and std_matrix[i, j] > 0.001:
                ax.text(j, i + 0.28, f'±{std_matrix[i,j]:.2f}',
                        ha='center', va='center', fontsize=5, color='gray')
    savefig(fig, output_path)


def fig3_probe_comparison(phase1_json_path, phase2_json_path, output_path):
    """Side-by-side oracle vs trained heatmaps."""
    with open(phase1_json_path) as f:
        d1 = json.load(f)
    with open(phase2_json_path) as f:
        d2 = json.load(f)

    probe_summary_oracle = d1.get('probe_summary', {})
    probe_means_trained = d2.get('probe_means', {})

    n_oracle = max(int(k) for k in probe_summary_oracle.keys()) + 1  # 5
    n_trained = 7

    mat_oracle = np.full((len(TARGETS), n_oracle), np.nan)
    mat_trained = np.full((len(TARGETS), n_trained), np.nan)
    is_cls = [t == 'branch_taken' for t in TARGETS]

    for j in range(n_oracle):
        ld = probe_summary_oracle.get(str(j), {})
        for i, t in enumerate(TARGETS):
            v = ld.get(t)
            if v is not None:
                mat_oracle[i, j] = v

    for i, t in enumerate(TARGETS):
        tdata = probe_means_trained.get(t, {})
        for j in range(n_trained):
            entry = tdata.get(str(j))
            if entry:
                mat_trained[i, j] = entry['mean']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    vmin, vmax = 0.0, 1.0
    cmap = CMAP_PROBE

    for ax, mat, n_l, title in [
        (axes[0], mat_oracle, n_oracle, 'Oracle'),
        (axes[1], mat_trained, n_trained, 'Trained (mean, 5 seeds)'),
    ]:
        display = np.clip(mat, vmin, vmax)
        im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_l))
        ax.set_xticklabels([f'L{j}' for j in range(n_l)], fontsize=9)
        ax.set_yticks(range(len(TARGETS)))
        ax.set_yticklabels(TARGET_LABELS, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Layer')
        for i in range(len(TARGETS)):
            for j in range(n_l):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = f'{v:.2f}'
                color = 'white' if display[i, j] > 0.65 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle('Linear Probe Comparison: Oracle vs Trained', fontsize=12)
    savefig(fig, output_path)


# ── Fig 4 & 5: Patching heatmaps ─────────────────────────────────────────────

def make_patch_heatmap(effects_mean, pair_types, title, output_path, n_layers, n_positions,
                        pos_highlight=None):
    """
    effects_mean: {ptype: (n_layers+1, seq_len) array}
    """
    n_pt = len(pair_types)
    fig, axes = plt.subplots(1, n_pt, figsize=(5 * n_pt, 4))
    if n_pt == 1:
        axes = [axes]

    pt_labels = {'mem_a': 'mem[a] contrast',
                 'mem_b': 'mem[b] contrast',
                 'branch': 'branch contrast'}

    for ax, ptype in zip(axes, pair_types):
        effects = effects_mean.get(ptype)
        if effects is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
            ax.set_title(pt_labels.get(ptype, ptype))
            continue

        effects_t = np.array(effects)
        # Show only first few positions for clarity
        max_pos = min(33, effects_t.shape[1])
        disp = effects_t[:, :max_pos]

        im = ax.imshow(disp, cmap=CMAP_PATCH, vmin=0, vmax=disp.max(),
                       aspect='auto', interpolation='nearest')
        ax.set_xticks(range(0, max_pos, 4))
        ax.set_xticklabels(range(0, max_pos, 4), fontsize=7)
        ax.set_yticks(range(effects_t.shape[0]))
        ax.set_yticklabels([f'L{l}' for l in range(effects_t.shape[0])], fontsize=8)
        ax.set_xlabel('Position')
        ax.set_title(pt_labels.get(ptype, ptype), fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Effect')

    fig.suptitle(title, fontsize=11)
    savefig(fig, output_path)


def fig4_oracle_patch(phase5_pkl_path, output_path):
    if not os.path.exists(phase5_pkl_path):
        print(f"  Skipping fig4: {phase5_pkl_path} not found")
        return
    with open(phase5_pkl_path, 'rb') as f:
        data = pickle.load(f)
    oracle_effects = data.get('oracle_effects', {})
    pair_types = data.get('pair_types', list(oracle_effects.keys()))
    make_patch_heatmap(oracle_effects, pair_types,
                        'Oracle: Activation Patching Effect',
                        output_path, n_layers=5, n_positions=417)


def fig5_trained_patch(phase3_pkl_path, output_path):
    if not os.path.exists(phase3_pkl_path):
        print(f"  Skipping fig5: {phase3_pkl_path} not found")
        return
    with open(phase3_pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = data.get('results', {})
    pair_types = data.get('pair_types', ['mem_a', 'mem_b', 'branch'])

    # Average across seeds
    mean_effects = {}
    for ptype in pair_types:
        seed_effects = []
        for seed_data in results.values():
            eff = seed_data.get('effects', {}).get(ptype)
            if eff is not None:
                seed_effects.append(np.array(eff))
        if seed_effects:
            mean_effects[ptype] = np.mean(seed_effects, axis=0)

    make_patch_heatmap(mean_effects, pair_types,
                        'Trained Model: Activation Patching Effect (mean, 5 seeds)',
                        output_path, n_layers=7, n_positions=33)


# ── Fig 6: 2×2 Diagnostic Table ──────────────────────────────────────────────

def fig6_diagnostic(phase2_json_path, phase3_json_path, output_path):
    """
    2×2 (high/low probe accuracy) × (high/low patching effect) for each quantity.
    """
    with open(phase2_json_path) as f:
        probe_data = json.load(f)
    with open(phase3_json_path) as f:
        patch_data = json.load(f)

    probe_means = probe_data.get('probe_means', {})

    # Best probe metric per target (max across layers)
    best_probe = {}
    for tname in TARGETS:
        best = -np.inf
        for layer_data in probe_means.get(tname, {}).values():
            v = layer_data.get('mean', -np.inf)
            if v > best:
                best = v
        best_probe[tname] = best

    # Max patching effect per target
    # pair_type → target mapping:
    # mem_a contrast → mem_a is causally important
    # mem_b contrast → mem_b is causally important
    # branch contrast → branch_taken is causally important
    # For pc and delta, average over all pair types
    type_to_target = {'mem_a': 'mem_a', 'mem_b': 'mem_b', 'branch': 'branch_taken'}

    best_patch = {t: 0.0 for t in TARGETS}
    # Average over seeds (keys 0-4 in patch_data)
    seed_count = 0
    for seed_key, seed_data in patch_data.items():
        seed_count += 1
        for ptype, target in type_to_target.items():
            pdata = seed_data.get(ptype, {})
            layer_maxes = [v.get('max', 0) for v in pdata.values()]
            if layer_maxes:
                max_eff = max(layer_maxes)
                best_patch[target] = max(best_patch[target], max_eff)
    # For pc and delta, use max of all pair types
    for tname in ['pc', 'delta']:
        for ptype in ['mem_a', 'mem_b', 'branch']:
            for seed_data in patch_data.values():
                pdata = seed_data.get(ptype, {})
                layer_maxes = [v.get('max', 0) for v in pdata.values()]
                if layer_maxes:
                    best_patch[tname] = max(best_patch[tname], max(layer_maxes))

    # Classification thresholds
    probe_threshold = 0.9
    patch_threshold = 0.1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(probe_threshold, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(patch_threshold, color='gray', linestyle='--', linewidth=0.8)

    colors = {'pc': '#e41a1c', 'mem_a': '#377eb8', 'mem_b': '#4daf4a',
              'delta': '#ff7f00', 'branch_taken': '#984ea3'}

    for tname, label in zip(TARGETS, TARGET_LABELS):
        px = best_patch.get(tname, 0)
        py = max(best_probe.get(tname, 0), 0)
        ax.scatter(px, py, s=200, color=colors.get(tname, 'black'),
                   zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(label, (px, py), textcoords='offset points',
                    xytext=(6, 4), fontsize=10)

    # Quadrant labels
    ax.text(patch_threshold / 2, (probe_threshold + 1.1) / 2,
            'Probed\nnot causal', ha='center', va='center',
            fontsize=8, color='gray', alpha=0.7)
    ax.text((patch_threshold + 1.1) / 2, (probe_threshold + 1.1) / 2,
            'Probed\n& causal', ha='center', va='center',
            fontsize=8, color='steelblue', alpha=0.7)
    ax.text(patch_threshold / 2, probe_threshold / 2,
            'Not probed\nnot causal', ha='center', va='center',
            fontsize=8, color='gray', alpha=0.7)
    ax.text((patch_threshold + 1.1) / 2, probe_threshold / 2,
            'Causal\nnot probed', ha='center', va='center',
            fontsize=8, color='orange', alpha=0.7)

    ax.set_xlabel('Max patching effect', fontsize=11)
    ax.set_ylabel('Best probe score (R² or accuracy)', fontsize=11)
    ax.set_title('2×2 Diagnostic: Probed vs Causal', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    savefig(fig, output_path)


# ── Fig 7: Dimensional Localization ──────────────────────────────────────────

def fig7_localization(phase6_loc_path, output_path):
    if not os.path.exists(phase6_loc_path):
        print(f"  Skipping fig7: {phase6_loc_path} not found")
        return
    with open(phase6_loc_path) as f:
        data = json.load(f)

    oracle_loc = data.get('oracle', {})
    trained_loc = data.get('trained_seed0', {})

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4), sharey=False)

    for ax, tname, tlabel in zip(axes, TARGETS, TARGET_LABELS):
        # Oracle: use best (max-metric) layer
        oracle_curves = oracle_loc.get(tname, {})
        best_oracle = None
        best_oracle_score = -np.inf
        for layer_key, kv_list in oracle_curves.items():
            if kv_list:
                score = max(m for _, m in kv_list)
                if score > best_oracle_score:
                    best_oracle_score = score
                    best_oracle = kv_list

        # Trained: use best layer
        trained_curves = trained_loc.get(tname, {})
        best_trained = None
        best_trained_score = -np.inf
        for layer_key, kv_list in trained_curves.items():
            if kv_list:
                score = max(m for _, m in kv_list)
                if score > best_trained_score:
                    best_trained_score = score
                    best_trained = kv_list

        if best_oracle:
            ks, ms = zip(*best_oracle)
            ax.plot(ks, ms, 'o-', label='Oracle', color='#e41a1c', linewidth=1.5)
        if best_trained:
            ks, ms = zip(*best_trained)
            ax.plot(ks, ms, 's--', label='Trained', color='#377eb8', linewidth=1.5)

        ax.set_xscale('log')
        ax.set_xlabel('Top-k dims', fontsize=9)
        ax.set_title(tlabel, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if ax == axes[0]:
            ax.set_ylabel('Probe metric', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle('Dimensional Localization: Oracle vs Trained', fontsize=12)
    savefig(fig, output_path)


# ── Fig 8: Training Dynamics ──────────────────────────────────────────────────

def fig8_dynamics(phase6_dyn_path, output_path):
    if not os.path.exists(phase6_dyn_path):
        print(f"  Skipping fig8: {phase6_dyn_path} not found")
        return
    with open(phase6_dyn_path) as f:
        data = json.load(f)

    fracs = sorted(int(k) for k in data.keys())

    # For each target, collect mean ± std across seeds at each frac at best layer
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4), sharey=False)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    for ax, tname, tlabel, color in zip(axes, TARGETS, TARGET_LABELS, colors):
        frac_means = []
        frac_stds = []
        valid_fracs = []

        for frac in fracs:
            seed_dict = data.get(str(frac), {})
            best_per_seed = []
            for seed, tdict in seed_dict.items():
                layer_vals = tdict.get(tname, {})
                if layer_vals:
                    best = max(float(v) for v in layer_vals.values())
                    best_per_seed.append(best)
            if best_per_seed:
                frac_means.append(np.mean(best_per_seed))
                frac_stds.append(np.std(best_per_seed))
                valid_fracs.append(frac)

        if valid_fracs:
            frac_means = np.array(frac_means)
            frac_stds = np.array(frac_stds)
            ax.plot(valid_fracs, frac_means, 'o-', color=color, linewidth=2)
            ax.fill_between(valid_fracs,
                             frac_means - frac_stds,
                             frac_means + frac_stds,
                             alpha=0.25, color=color)

        ax.set_xlabel('Training fraction (%)', fontsize=9)
        ax.set_title(tlabel, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Best probe metric', fontsize=9)

    fig.suptitle('Training Dynamics: Probe Accuracy vs Training Fraction', fontsize=12)
    savefig(fig, output_path)


# ── Fig 9: Failure Trace ──────────────────────────────────────────────────────

def fig9_failure_trace(phase6_trace_path, output_path):
    if not os.path.exists(phase6_trace_path):
        print(f"  Skipping fig9: {phase6_trace_path} not found")
        return
    with open(phase6_trace_path) as f:
        data = json.load(f)

    if not data:
        print("  Skipping fig9: no trace data")
        return

    # Use first program
    prog_name = list(data.keys())[0]
    trace = data[prog_name]

    # Plot: for each step, show model correct/incorrect, and decoded PC vs GT PC
    steps = [s['step'] for s in trace]
    correct = [s.get('correct', False) for s in trace]
    gt_pcs = [s.get('true_pc', s.get('gt', {}).get('pc', 0)) for s in trace]
    pred_pcs = [s.get('pred_pc', 0) for s in trace]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    ax1 = axes[0]
    ax1.plot(steps, gt_pcs, 'go-', label='Ground truth PC', linewidth=1.5)
    ax1.plot(steps, pred_pcs, 'r^--', label='Predicted PC', linewidth=1.5)
    ax1.set_ylabel('PC value')
    ax1.set_title(f'Failure Trace: {prog_name}', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = ['green' if c else 'red' for c in correct]
    ax2.bar(steps, [1] * len(steps), color=colors, alpha=0.7)
    ax2.set_ylabel('Step correct')
    ax2.set_xlabel('Step')
    ax2.set_yticks([])
    ax2.set_xticks(steps)
    # Add first-wrong annotation
    for i, c in enumerate(correct):
        if not c:
            ax2.axvline(steps[i], color='red', linewidth=2, linestyle='--', alpha=0.5)
            ax2.text(steps[i], 0.5, f'First wrong\nat step {steps[i]}',
                     ha='center', fontsize=8, color='red')
            break

    ax2.grid(True, alpha=0.3, axis='x')
    savefig(fig, output_path)


# ── Fig 10: Constrained model probe comparison ───────────────────────────────

def fig10_constrained_probe(constrained_summary_path, phase1_path, phase2_path, output_path):
    """
    4-panel heatmap: Oracle | Constrained-LN | Constrained-noLN | Trained.
    All at pos0, best metric per layer.
    Skips gracefully if constrained_summary_path doesn't exist.
    """
    if not os.path.exists(constrained_summary_path):
        print(f"  Skipping Fig 10: {constrained_summary_path} not found")
        return

    with open(constrained_summary_path) as f:
        cdata = json.load(f)
    with open(phase1_path) as f:
        oracle_data = json.load(f)
    with open(phase2_path) as f:
        trained_data = json.load(f)

    # Oracle matrix (embed + 4 blocks)
    probe_summary_oracle = oracle_data.get('probe_summary', {})
    n_oracle = max(int(k) for k in probe_summary_oracle.keys()) + 1
    mat_oracle = np.full((len(TARGETS), n_oracle), np.nan)
    for j in range(n_oracle):
        ld = probe_summary_oracle.get(str(j), {})
        for i, t in enumerate(TARGETS):
            v = ld.get(t)
            if v is not None:
                mat_oracle[i, j] = v

    panels = [('Oracle (constructed)', mat_oracle, n_oracle)]

    for variant_key, title in [('ln', 'Constrained-LN'),
                                ('no_ln', 'Constrained-noLN')]:
        vdata = cdata.get(variant_key, {})
        probe_means = vdata.get('probe_means', {})
        if not probe_means:
            panels.append((title, np.full((len(TARGETS), 5), np.nan), 5))
            continue
        n_l = max(int(k) for t in probe_means.values() for k in t.keys()) + 1
        mat = np.full((len(TARGETS), n_l), np.nan)
        for i, tname in enumerate(TARGETS):
            tdata = probe_means.get(tname, {})
            for j in range(n_l):
                entry = tdata.get(str(j))
                if entry:
                    mat[i, j] = entry['mean']
        panels.append((title, mat, n_l))

    # Trained model panel (d=256, 6 layers)
    trained_probe_means = trained_data.get('probe_means', {})
    n_trained = 7  # embed + 6 blocks
    mat_trained = np.full((len(TARGETS), n_trained), np.nan)
    for i, tname in enumerate(TARGETS):
        tdata = trained_probe_means.get(tname, {})
        for j in range(n_trained):
            entry = tdata.get(str(j))
            if entry:
                mat_trained[i, j] = entry['mean']
    panels.append(('Trained (d=256)', mat_trained, n_trained))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0
    for ax, (title, mat, n_l) in zip(axes, panels):
        display = np.clip(mat, vmin, vmax)
        im = ax.imshow(display, cmap=CMAP_PROBE, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_l))
        ax.set_xticklabels([f'L{j}' for j in range(n_l)], fontsize=8)
        ax.set_yticks(range(len(TARGETS)))
        ax.set_yticklabels(TARGET_LABELS, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Layer')
        for i in range(len(TARGETS)):
            for j in range(n_l):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = 'white' if display[i, j] > 0.65 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle('Linear Probe Heatmaps: Oracle vs Constrained vs Trained', fontsize=12)
    savefig(fig, output_path)


# ── Fig 11: Constrained-LN patching heatmap ──────────────────────────────────

def fig11_constrained_patch(constrained_patch_pkl, phase3_pkl, phase5_pkl, output_path):
    """
    3-panel activation patching comparison: Oracle | Constrained-LN | Trained.
    Shows mean effect heatmap (layer × pair_type) for each model.
    """
    if not os.path.exists(constrained_patch_pkl):
        print(f"  Skipping Fig 11: {constrained_patch_pkl} not found")
        return

    with open(constrained_patch_pkl, 'rb') as f:
        c_data = pickle.load(f)
    with open(phase5_pkl, 'rb') as f:
        oracle_raw = pickle.load(f)
    with open(phase3_pkl, 'rb') as f:
        trained_raw = pickle.load(f)

    pair_types = c_data.get('pair_types', ['mem_a', 'mem_b', 'branch'])
    pair_labels = ['mem_a', 'mem_b', 'branch']

    def max_over_pos(effects_dict, pair_types):
        """Return (n_layers+1, n_pair_types) array of max effect over positions."""
        rows = []
        for ptype in pair_types:
            arr = effects_dict.get(ptype)
            if arr is None:
                rows.append(np.zeros(5))
                continue
            rows.append(np.array(arr).max(axis=1))  # max over positions per layer
        return np.array(rows).T  # (n_layers+1, n_pair_types)

    # Constrained-LN: mean over seeds
    c_results = c_data.get('results', {})
    if c_results:
        stacked = {}
        for ptype in pair_types:
            arrs = [c_results[s]['effects'][ptype] for s in c_results
                    if ptype in c_results[s]['effects']]
            stacked[ptype] = np.stack(arrs).mean(axis=0)
        c_mat = max_over_pos(stacked, pair_types)
    else:
        c_mat = np.zeros((5, len(pair_types)))

    # Oracle
    oracle_effects = oracle_raw.get('oracle_effects', {})
    # oracle_effects: {ptype: (n_layers+1, seq_len)} numpy
    o_mat = max_over_pos(oracle_effects, pair_types)

    # Trained: mean over seeds
    t_results = trained_raw.get('results', {})
    if t_results:
        stacked_t = {}
        for ptype in pair_types:
            arrs = [t_results[s]['effects'][ptype] for s in t_results
                    if ptype in t_results[s].get('effects', {})]
            stacked_t[ptype] = np.stack(arrs).mean(axis=0)
        t_mat = max_over_pos(stacked_t, pair_types)
    else:
        t_mat = np.zeros((7, len(pair_types)))

    panels = [
        ('Oracle (constructed, L0-4)', o_mat),
        ('Constrained-LN (L0-4)', c_mat),
        ('Trained (d=256, L0-6)', t_mat),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = 0.0, 1.0
    for ax, (title, mat) in zip(axes, panels):
        n_l = mat.shape[0]
        display = np.clip(mat, vmin, vmax)
        im = ax.imshow(display, cmap=CMAP_PATCH, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(pair_types)))
        ax.set_xticklabels(pair_labels, fontsize=9)
        ax.set_yticks(range(n_l))
        ax.set_yticklabels([f'L{l}' for l in range(n_l)], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Pair type')
        ax.set_ylabel('Patch layer')
        for i in range(n_l):
            for j in range(len(pair_types)):
                v = mat[i, j]
                color = 'white' if display[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle('Activation Patching: Oracle vs Constrained-LN vs Trained\n'
                 '(max effect over positions per layer)', fontsize=11)
    savefig(fig, output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                        default=os.path.join(script_dir, 'results'))
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(script_dir, 'figures'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    R = args.results_dir
    O = args.output_dir

    print("Generating figures...")

    print("Fig 1: Oracle probe heatmap")
    fig1_oracle_probe(
        os.path.join(R, 'phase1_oracle.json'),
        os.path.join(O, 'fig1_oracle_probe_heatmap.png'),
    )

    print("Fig 2: Trained model probe heatmap")
    fig2_trained_probe(
        os.path.join(R, 'phase2_summary.json'),
        os.path.join(O, 'fig2_trained_probe_heatmap.png'),
    )

    print("Fig 3: Oracle vs trained comparison")
    fig3_probe_comparison(
        os.path.join(R, 'phase1_oracle.json'),
        os.path.join(R, 'phase2_summary.json'),
        os.path.join(O, 'fig3_probe_comparison.png'),
    )

    print("Fig 4: Oracle patching heatmap")
    fig4_oracle_patch(
        os.path.join(R, 'phase5_oracle_patch.pkl'),
        os.path.join(O, 'fig4_oracle_patch_heatmap.png'),
    )

    print("Fig 5: Trained model patching heatmap")
    fig5_trained_patch(
        os.path.join(R, 'phase3_patching.pkl'),
        os.path.join(O, 'fig5_trained_patch_heatmap.png'),
    )

    print("Fig 6: 2×2 diagnostic table")
    fig6_diagnostic(
        os.path.join(R, 'phase2_summary.json'),
        os.path.join(R, 'phase3_summary.json'),
        os.path.join(O, 'fig6_diagnostic_table.png'),
    )

    print("Fig 7: Dimensional localization")
    fig7_localization(
        os.path.join(R, 'phase6_localization.json'),
        os.path.join(O, 'fig7_localization.png'),
    )

    print("Fig 8: Training dynamics")
    fig8_dynamics(
        os.path.join(R, 'phase6_dynamics.json'),
        os.path.join(O, 'fig8_dynamics.png'),
    )

    print("Fig 9: Failure trace")
    fig9_failure_trace(
        os.path.join(R, 'phase6_failure_trace.json'),
        os.path.join(O, 'fig9_failure_trace.png'),
    )

    print("Fig 10: Constrained model probe comparison (4-panel)")
    fig10_constrained_probe(
        os.path.join(R, 'phase2_constrained_summary.json'),
        os.path.join(R, 'phase1_oracle.json'),
        os.path.join(R, 'phase2_summary.json'),
        os.path.join(O, 'fig10_constrained_probe.png'),
    )

    print("Fig 11: Constrained-LN patching comparison")
    fig11_constrained_patch(
        os.path.join(R, 'phase3_constrained_ln.pkl'),
        os.path.join(R, 'phase3_patching.pkl'),
        os.path.join(R, 'phase5_oracle_patch.pkl'),
        os.path.join(O, 'fig11_constrained_patch.png'),
    )

    print(f"\nAll figures saved to {O}/")


if __name__ == '__main__':
    main()
