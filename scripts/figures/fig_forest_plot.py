"""Figure: Effect size CI forest plot with TOST equivalence bounds.

Shows paired Cohen's d with 95% CI for each condition×regime comparison,
with SESOI bounds (±0.5) shown as vertical dashed lines.  CIs falling
entirely within the bounds support a bounded null claim.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT

_EXP_DIR = PROJECT_ROOT / "experiments"


def _load_comparisons(analysis_path, comparison_key="pairwise_vs_baseline"):
    """Load paired comparison results from an analysis JSON."""
    if not analysis_path.exists():
        return {}
    with open(analysis_path) as f:
        data = json.load(f)
    # Handle nested regime structure
    if "famine" in data or "boom_bust" in data:
        results = {}
        for regime, rdata in data.items():
            if not isinstance(rdata, dict) or "error" in rdata:
                continue
            comparisons = rdata.get(comparison_key, {})
            for cond, stats in comparisons.items():
                if "paired_cohens_d" in stats:
                    label = f"{cond}\n({regime})"
                    results[label] = stats
        return results
    # Flat structure
    comparisons = data.get(comparison_key, {})
    return {
        cond: stats
        for cond, stats in comparisons.items()
        if "paired_cohens_d" in stats
    }


def generate_forest_plot():
    """Generate forest plot of effect sizes with TOST bounds."""
    analyses = [
        (_EXP_DIR / "criterion8_analysis.json", "Normal"),
        (_EXP_DIR / "stress_analysis.json", "Stress (A)"),
        (_EXP_DIR / "candidateB_stress_analysis.json", "Stress (B)"),
        (_EXP_DIR / "candidateB_preshock_analysis.json", "Pre-shock (B)"),
        (_EXP_DIR / "seasonal_analysis.json", "Seasonal"),
    ]

    all_rows = []  # (label, d, ci_lo, ci_hi)
    for path, group_label in analyses:
        comparisons = _load_comparisons(path)
        if not comparisons:
            continue
        for label, stats in comparisons.items():
            d = stats["paired_cohens_d"]
            ci_lo = stats.get("paired_cohens_d_ci_lo", d)
            ci_hi = stats.get("paired_cohens_d_ci_hi", d)
            all_rows.append((f"{group_label}: {label}", d, ci_lo, ci_hi))

    if not all_rows:
        print("  SKIP: no paired comparison data found for forest plot")
        return

    n = len(all_rows)
    fig, ax = plt.subplots(figsize=(5, max(2, 0.35 * n + 0.5)))

    # SESOI bounds
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="green", zorder=0)
    ax.axvline(0.5, color="green", ls="--", lw=0.8, alpha=0.5, label="SESOI (±0.5)")
    ax.axvline(-0.5, color="green", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", ls="-", lw=0.5, alpha=0.3)

    for i, (label, d, ci_lo, ci_hi) in enumerate(reversed(all_rows)):
        y = i
        within_sesoi = -0.5 <= ci_lo and ci_hi <= 0.5
        color = "#009E73" if within_sesoi else "#D55E00"
        ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=1.5, solid_capstyle="round")
        ax.plot(d, y, "o", color=color, markersize=5, zorder=5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([row[0] for row in reversed(all_rows)], fontsize=7)
    ax.set_xlabel("Paired Cohen's d (95% CI)")
    ax.set_title("Effect Size Forest Plot with TOST Bounds", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / "fig_forest_plot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
