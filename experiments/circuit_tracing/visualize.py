"""
Visualisation helpers for circuit tracing results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# ─── Attention pattern plots ──────────────────────────────────────────────────

def plot_attention_patterns(attn_weights, layer_idx, example_idx=0,
                            title_prefix='', save_path=None):
    """
    Plot all heads' attention matrices for one example at one layer.

    attn_weights : Tensor (B, H, T, T)  or  (H, T, T)
    """
    if attn_weights.dim() == 4:
        attn = attn_weights[example_idx].detach().cpu().numpy()   # (H, T, T)
    else:
        attn = attn_weights.detach().cpu().numpy()

    H, T, _ = attn.shape
    ncols = min(H, 4)
    nrows = (H + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for h in range(H):
        ax = axes[h]
        im = ax.imshow(attn[h], vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {h}', fontsize=9)
        ax.set_xlabel('Key pos')
        ax.set_ylabel('Query pos')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[H:]:
        ax.set_visible(False)

    fig.suptitle(f'{title_prefix}Layer {layer_idx} attention patterns', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_head_role_heatmap(fetch_scores, broadcast_scores, random_baseline,
                           model_name='', save_path=None):
    """
    Two-panel heatmap: fetch scores (left) and broadcast scores (right).

    fetch_scores     : ndarray (L, H)
    broadcast_scores : ndarray (L, H)
    """
    L, H = fetch_scores.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(6, H * 0.8 + 2) * 2, L * 0.8 + 2))

    vmax_f = max(fetch_scores.max(), random_baseline * 10)
    vmax_b = max(broadcast_scores.max(), random_baseline * 10)

    for ax, data, title, vmax in zip(
        axes,
        [fetch_scores, broadcast_scores],
        ['Fetch score (attn weight @ target)', 'Broadcast score (attn weight @ pos 0)'],
        [vmax_f, vmax_b],
    ):
        im = ax.imshow(data, vmin=0, vmax=vmax, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(H))
        ax.set_xticklabels([f'H{h}' for h in range(H)])
        ax.set_yticks(range(L))
        ax.set_yticklabels([f'L{l}' for l in range(L)])
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Mark random baseline as a reference line in colourbar
        ax.axhline(-0.5, color='white', linewidth=0)  # invisible spacer

    fig.suptitle(f'Head circuit roles — {model_name}\n'
                 f'(random baseline ≈ {random_baseline:.3f})', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


# ─── FFN neuron plots ─────────────────────────────────────────────────────────

def plot_neuron_concept_correlations(corr_dict, layer_idx, token_pos=0,
                                     top_k=10, model_name='', save_path=None):
    """
    Bar chart: top neurons correlated with each SUBLEQ concept.

    corr_dict : dict  concept_name → ndarray (d_ff,)
    """
    n_concepts = len(corr_dict)
    fig, axes = plt.subplots(1, n_concepts,
                              figsize=(4 * n_concepts, 3), squeeze=False)
    axes = axes[0]

    for ax, (name, corr) in zip(axes, corr_dict.items()):
        top_idx = np.argsort(np.abs(corr))[::-1][:top_k]
        vals = corr[top_idx]
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in vals]
        ax.barh(range(top_k), vals, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f'n{i}' for i in top_idx], fontsize=8)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Pearson r')
        ax.set_title(f'{name}\n(L{layer_idx}, pos {token_pos})', fontsize=9)

    fig.suptitle(f'FFN neuron–concept correlations — {model_name}', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_ov_spectrum(ov_circuits, layer_idx, model_name='', save_path=None):
    """
    Plot singular value spectra of W_OV for each head at a layer.

    ov_circuits : Tensor (H, d_model, d_model)
    """
    from .attribution import ov_eigenspectrum
    svs = ov_eigenspectrum(ov_circuits).detach().cpu().numpy()   # (H, d_model)
    H = svs.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, H), 3))
    for h in range(H):
        ax.plot(svs[h], label=f'H{h}', marker='o', markersize=3)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    ax.set_title(f'W_OV singular value spectra — {model_name}, Layer {layer_idx}', fontsize=10)
    ax.legend(ncol=4, fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


# ─── Summary comparison plot ─────────────────────────────────────────────────

def plot_model_comparison(results_dict, random_baseline, save_path=None):
    """
    Compare fetch/broadcast scores across multiple models side by side.

    results_dict : dict  model_name → {'fetch': (L,H), 'broadcast': (L,H)}
    """
    n_models = len(results_dict)
    names = list(results_dict.keys())
    L, H = next(iter(results_dict.values()))['fetch'].shape

    fig, axes = plt.subplots(2, n_models,
                              figsize=(4 * n_models, 3 * 2),
                              squeeze=False)

    for col, name in enumerate(names):
        data = results_dict[name]
        for row, (key, title) in enumerate([
            ('fetch', 'Fetch score'),
            ('broadcast', 'Broadcast score'),
        ]):
            ax = axes[row][col]
            scores = data[key]
            vmax = max(scores.max(), random_baseline * 10)
            im = ax.imshow(scores, vmin=0, vmax=vmax, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(H))
            ax.set_xticklabels([f'H{h}' for h in range(H)], fontsize=7)
            ax.set_yticks(range(L))
            ax.set_yticklabels([f'L{l}' for l in range(L)], fontsize=7)
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(name, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Circuit role comparison (random baseline ≈ {random_baseline:.3f})',
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_circuit_completeness(completeness_results, save_path=None):
    """
    Bar chart comparing full model vs. circuit-only accuracy across circuit hypotheses.

    completeness_results : list of dicts
        each has 'label', 'full_acc', 'circuit_acc', 'ratio'
    """
    labels = [r['label'] for r in completeness_results]
    full_accs = [r['full_acc'] for r in completeness_results]
    circ_accs = [r['circuit_acc'] for r in completeness_results]
    ratios = [r['ratio'] for r in completeness_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.5), 4))

    ax1.bar(x - width / 2, full_accs, width, label='Full model', color='#1f77b4')
    ax1.bar(x + width / 2, circ_accs, width, label='Circuit only', color='#ff7f0e')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax1.set_ylabel('Token accuracy')
    ax1.set_title('Full vs. circuit accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    ax2.bar(x, ratios, color='#2ca02c')
    ax2.axhline(0.9, color='r', linestyle='--', linewidth=1, label='90% threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('circuit_acc / full_acc')
    ax2.set_title('Circuit completeness ratio')
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig
