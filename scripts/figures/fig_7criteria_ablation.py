"""Figure: 7-criteria ablation bar chart.

Bar chart showing Cohen's d effect sizes for all 7 single-criterion
ablations from the final_graph_statistics.json data.  All 7 are
significant (d = 0.78 to 14.7), demonstrating the framework's
diagnostic power before testing 8th-criterion candidates.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import COLORS, FIG_DIR, LABELS, PROJECT_ROOT

_EXP_DIR = PROJECT_ROOT / "experiments"
_STATS_PATH = _EXP_DIR / "final_graph_statistics.json"

_ABLATION_ORDER = [
    "no_reproduction",
    "no_response",
    "no_metabolism",
    "no_homeostasis",
    "no_growth",
    "no_boundary",
    "no_evolution",
]


def generate_7criteria_ablation():
    """Generate 7-criteria ablation effect size bar chart."""
    if not _STATS_PATH.exists():
        print(f"  SKIP: {_STATS_PATH} not found")
        return

    with open(_STATS_PATH) as f:
        stats = json.load(f)

    comparisons_raw = stats.get("comparisons", [])
    if not comparisons_raw:
        print("  SKIP: no comparisons in statistics JSON")
        return

    # Build lookup: condition name → stats dict
    if isinstance(comparisons_raw, list):
        comparisons = {c["condition"]: c for c in comparisons_raw if "condition" in c}
    else:
        comparisons = comparisons_raw

    conditions = []
    d_values = []
    colors_list = []
    significances = []

    for cond in _ABLATION_ORDER:
        comp = comparisons.get(cond, {})
        d = comp.get("cohens_d") or comp.get("cohen_d")
        if d is None:
            continue
        conditions.append(cond)
        d_values.append(abs(d))  # absolute value for visualization
        colors_list.append(COLORS.get(cond, "#999999"))
        p_adj = comp.get("p_corrected") or comp.get("p_adjusted", 1.0)
        if p_adj < 0.001:
            significances.append("***")
        elif p_adj < 0.01:
            significances.append("**")
        elif p_adj < 0.05:
            significances.append("*")
        else:
            significances.append("ns")

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(conditions))
    bars = ax.bar(x, d_values, color=colors_list, alpha=0.7, edgecolor="black", lw=0.5)

    # Significance stars
    for i, (bar, sig) in enumerate(zip(bars, significances)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            sig,
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS.get(c, c).replace("No ", "−") for c in conditions],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_ylabel("|Cohen's d| vs Normal")
    ax.set_title("7-Criteria Ablation Effect Sizes", fontsize=10)

    # Reference lines
    ax.axhline(0.8, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(len(conditions) - 0.5, 0.85, "large", fontsize=6, color="gray", ha="right")

    fig.tight_layout()
    out = FIG_DIR / "fig_7criteria_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
