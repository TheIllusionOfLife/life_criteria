"""Figure: Boxplots + jitter for all conditions across regimes.

Multi-panel figure showing survival AUC distributions for all
conditions under each perturbation regime (normal, famine, boom-bust,
seasonal), for both Candidate A and B.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT

_EXP_DIR = PROJECT_ROOT / "experiments"

_CANDIDATE_A_COLORS = {
    "baseline": "#000000",
    "criterion8_on": "#0072B2",
    "criterion8_ablated": "#D55E00",
    "sham": "#CC79A7",
}
_CANDIDATE_B_COLORS = {
    "baseline": "#000000",
    "candidateB_on": "#0072B2",
    "candidateB_ablated": "#D55E00",
    "sham": "#CC79A7",
}

_LABEL_MAP = {
    "baseline": "Baseline",
    "criterion8_on": "+Memory",
    "criterion8_ablated": "+Memory (ablated)",
    "candidateB_on": "+Kin-Sensing",
    "candidateB_ablated": "+Kin-Sensing (ablated)",
    "sham": "Sham",
}


def _load_aucs(path, conditions):
    """Load per-seed AUC values from experiment JSONs."""
    data = {}
    for cond in conditions:
        fpath = _EXP_DIR / f"{path}_{cond}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            results = json.load(f)
        data[cond] = [sum(s["alive_count"] for s in r.get("samples", [])) for r in results]
    return data


def _plot_panel(ax, data, colors, title):
    """Plot boxplot + jitter for one regime."""
    if not data:
        ax.set_visible(False)
        return
    conditions = [c for c in data if data[c]]
    if not conditions:
        ax.set_visible(False)
        return
    vals = [data[c] for c in conditions]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5, showfliers=False)
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(colors.get(cond, "#999999"))
        patch.set_alpha(0.4)
    for patch in bp["medians"]:
        patch.set_color("black")
        patch.set_linewidth(1.5)
    # Jitter
    rng = np.random.default_rng(42)
    for i, (cond, v) in enumerate(zip(conditions, vals)):
        jitter = rng.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(
            np.full(len(v), i + 1) + jitter,
            v,
            c=colors.get(cond, "#999999"),
            s=8, alpha=0.6, edgecolors="none", zorder=3,
        )
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels([_LABEL_MAP.get(c, c) for c in conditions], rotation=30, ha="right")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Survival AUC")


def generate_boxplot_conditions():
    """Generate multi-panel boxplot figure."""
    datasets = [
        ("criterion8", ["baseline", "criterion8_on", "criterion8_ablated", "sham"],
         _CANDIDATE_A_COLORS, "Candidate A"),
        ("stress_famine", ["baseline", "criterion8_on", "criterion8_ablated", "sham"],
         _CANDIDATE_A_COLORS, "Candidate A — Famine"),
        ("stress_boom_bust", ["baseline", "criterion8_on", "criterion8_ablated", "sham"],
         _CANDIDATE_A_COLORS, "Candidate A — Boom-Bust"),
        ("candidateB_famine", ["baseline", "candidateB_on", "candidateB_ablated", "sham"],
         _CANDIDATE_B_COLORS, "Candidate B — Famine"),
        ("candidateB_boom_bust", ["baseline", "candidateB_on", "candidateB_ablated", "sham"],
         _CANDIDATE_B_COLORS, "Candidate B — Boom-Bust"),
    ]

    # Only plot panels with data
    available = [(prefix, conds, colors, title)
                 for prefix, conds, colors, title in datasets
                 if any((_EXP_DIR / f"{prefix}_{c}.json").exists() for c in conds)]

    if not available:
        print("  SKIP: no experiment data found for boxplot figure")
        return

    n = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (prefix, conds, colors, title) in enumerate(available):
        data = _load_aucs(prefix, conds)
        _plot_panel(axes[i], data, colors, title)

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "fig_boxplot_conditions.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
