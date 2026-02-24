"""Figure 20: Trait evolution / selection differential."""

import numpy as np
from figures._shared import *


def generate_trait_evolution() -> None:
    """Figure 20: Trait evolution / selection differential.

    Panel A: Mean energy trajectory over snapshot steps (evolved vs no-evolution).
    Panel B: Box plots of per-seed high-gen vs low-gen energy at step 10000.
    """
    analysis_path = PROJECT_ROOT / "experiments" / "trait_evolution_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path) as f:
        import json

        data = json.load(f)

    trajectory = data.get("trajectory", {})
    sel_diff = data.get("selection_differential", {})

    if not trajectory or not sel_diff:
        print("  SKIP: missing trajectory or selection_differential data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # --- Panel A: Trajectory ---
    ax = axes[0]
    cond_styles = {
        "normal": ("Evolved", "#000000", "-"),
        "no_evo": ("No Evolution", "#CC79A7", "--"),
    }
    for cond, (label, color, ls) in cond_styles.items():
        traj = trajectory.get(cond, {})
        steps = traj.get("steps", [])
        means = traj.get("energy_mean", [])
        sems = traj.get("energy_sem", [])
        if not steps:
            continue
        means_arr = np.array(means)
        sems_arr = np.array(sems)
        ax.plot(steps, means_arr, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax.fill_between(steps, means_arr - sems_arr, means_arr + sems_arr, color=color, alpha=0.15)

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Organism Energy")
    ax.set_title("(A) Energy Trajectory", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: Box plot of high vs low gen energy at step 10000 ---
    ax = axes[1]

    box_data = []
    box_labels = []
    box_colors = []

    for cond, (label, color, _ls) in cond_styles.items():
        sd = sel_diff.get(cond, {})
        high = sd.get("per_seed_high_gen_energy", [])
        low = sd.get("per_seed_low_gen_energy", [])
        if high:
            box_data.append(high)
            box_labels.append(f"{label}\nHigh-gen")
            box_colors.append(color)
        if low:
            box_data.append(low)
            box_labels.append(f"{label}\nLow-gen")
            box_colors.append(color)

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for median_line in bp["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(1.5)

    # Annotate Cliff's delta and p-value for evolved condition
    sd_normal = sel_diff.get("normal", {})
    delta = sd_normal.get("cliff_delta", None)
    p_val = sd_normal.get("p_value", None)
    n_seeds = sd_normal.get("n_seeds_used", 0)
    if delta is not None and p_val is not None:
        ax.text(
            0.98,
            0.98,
            f"Evolved: \u03b4={delta:.2f}, p={p_val:.3g}\n(n={n_seeds} seeds)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
        )

    ax.set_ylabel("Mean Energy (per seed)")
    ax.set_title("(B) Selection Differential (step 10k)", fontsize=9)
    ax.tick_params(axis="x", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_trait_evolution.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_trait_evolution.pdf'}")
