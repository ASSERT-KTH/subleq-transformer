"""
Circuit tracing summary diagram.
Produces figures/fig_circuit_summary.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

# Hardcoded model head roles (manually derived from the scores in the JSON)
# L0: H3=broadcast(1.0). L1: H0=fetch_a(1.0), H1=fetch_b(1.0), H3=broadcast(0.23)
# L2: H0=broadcast(1.0), H1=broadcast(1.0). L3: all inactive.
hardcoded_roles = [
    ['other', 'other', 'other', 'broadcast', 'other', 'other', 'other', 'other'],
    ['fetch',  'fetch',  'other', 'broadcast', 'other', 'other', 'other', 'other'],
    ['broadcast', 'broadcast', 'other', 'other', 'other', 'other', 'other', 'other'],
    ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
]

trained_roles = {
    'Seed 0': [
        ['other', 'other', 'broadcast', 'broadcast', 'other', 'other', 'broadcast', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'broadcast', 'fetch', 'other', 'broadcast', 'fetch', 'other', 'other'],
    ],
    'Seed 1': [
        ['broadcast', 'other', 'other', 'broadcast', 'other', 'broadcast', 'broadcast', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'other', 'fetch', 'other', 'other', 'other', 'fetch', 'other'],
    ],
    'Seed 2': [
        ['broadcast', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],
        ['other', 'other', 'other', 'other', 'other', 'other', 'fetch', 'other'],
        ['other', 'other', 'other', 'other', 'fetch', 'other', 'fetch', 'fetch'],
    ],
}

completeness = {
    'Seed 0': {'all_heads': 1.0, 'fetch+broadcast': 0.211, 'fetch only': 0.193, 'broadcast only': 0.113},
    'Seed 1': {'all_heads': 1.0, 'fetch+broadcast': 0.181, 'fetch only': 0.174, 'broadcast only': 0.344},
    'Seed 2': {'all_heads': 1.0, 'fetch+broadcast': 0.338, 'fetch only': 0.338, 'broadcast only': 0.344},
}

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    'fetch':     '#e15759',   # red
    'broadcast': '#4e79a7',   # blue
    'mixed':     '#f28e2b',   # orange
    'other':     '#d9d9d9',   # light grey
}

def role_grid(ax, roles, title, n_layers=4, n_heads=8):
    """Draw a 4×8 grid of coloured squares for head roles."""
    ax.set_xlim(-0.5, n_heads - 0.5)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for l, layer_roles in enumerate(roles):
        for h, role in enumerate(layer_roles):
            color = COLORS[role]
            rect = mpatches.FancyBboxPatch(
                (h - 0.42, l - 0.42), 0.84, 0.84,
                boxstyle='round,pad=0.05',
                facecolor=color, edgecolor='white', linewidth=1.2
            )
            ax.add_patch(rect)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{h}' for h in range(n_heads)], fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{l}' for l in range(n_layers)], fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(
    3, 4,
    figure=fig,
    left=0.06, right=0.97,
    top=0.91, bottom=0.08,
    hspace=0.55, wspace=0.35,
)

# Row 0: head-role grids (hardcoded + 3 trained seeds)
ax_hc = fig.add_subplot(gs[0, 0])
role_grid(ax_hc, hardcoded_roles, 'Hardcoded (reference)')

for col, (seed_name, roles) in enumerate(trained_roles.items(), start=1):
    ax = fig.add_subplot(gs[0, col])
    role_grid(ax, roles, f'Trained — {seed_name}')

# Legend for role colours
legend_patches = [mpatches.Patch(color=COLORS[r], label=r.capitalize()) for r in ('fetch', 'broadcast', 'other')]
fig.legend(handles=legend_patches, loc='upper center', ncol=3,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.975))

# ── Row 1–2: circuit completeness bars ───────────────────────────────────────
bar_labels  = ['All heads\n(baseline)', 'Fetch +\nbroadcast', 'Fetch\nonly', 'Broadcast\nonly']
bar_x       = np.arange(len(bar_labels))
bar_colors  = ['#59a14f', '#f28e2b', '#e15759', '#4e79a7']

seed_names = list(completeness.keys())
for col, seed_name in enumerate(seed_names):
    ax = fig.add_subplot(gs[1:, col + 1])   # span rows 1–2, columns 1–3
    vals = list(completeness[seed_name].values())
    bars = ax.bar(bar_x, vals, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.6)
    ax.axhline(1.0, color='#59a14f', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, fontsize=7.5)
    ax.set_ylabel('Accuracy (fraction of full model)' if col == 0 else '', fontsize=8)
    ax.set_title(f'Circuit completeness — {seed_name}', fontsize=9, fontweight='bold', pad=4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                f'{val:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ── Explanation text in bottom-left cell ─────────────────────────────────────
ax_txt = fig.add_subplot(gs[1:, 0])
ax_txt.axis('off')
explanation = (
    "What the circuit tracing shows:\n\n"
    "TOP ROW — Head roles\n"
    "Each square is one attention head.\n"
    "Red = fetch head (reads a specific\n"
    "  memory address).\n"
    "Blue = broadcast head (sends the\n"
    "  current PC to all positions).\n"
    "Grey = 'other' (no clear single role).\n\n"
    "Hardcoded model has clean, sparse\n"
    "circuits: fetch in L1, broadcast in\n"
    "L0–L2.  Trained models spread the\n"
    "same roles across L0 and L3.\n\n"
    "BOTTOM — Circuit completeness\n"
    "If we keep only fetch/broadcast heads\n"
    "and zero out everything else, accuracy\n"
    "drops to ≈20–34%.  The 'grey' heads\n"
    "in L1–L2 are doing real work that the\n"
    "simple fetch/broadcast story misses."
)
ax_txt.text(0.04, 0.97, explanation, transform=ax_txt.transAxes,
            fontsize=8.2, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#cccccc'))

fig.suptitle('Phase 8 — Circuit Tracing: Trained vs Hardcoded SUBLEQ Transformer',
             fontsize=11, fontweight='bold', y=0.998)

out = '/mimer/NOBACKUP/groups/naiss2025-5-243/andre/subleq-transformer/figures/fig_circuit_summary.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
out2 = out.replace('.pdf', '.png')
fig.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
