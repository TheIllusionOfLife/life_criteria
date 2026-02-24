"""Figure 5: Homeostasis trajectory."""

from collections import defaultdict

import numpy as np
from figures._shared import *


def generate_homeostasis() -> None:
    """Figure 5: Homeostasis trajectory — internal state regulation over time.

    Panel A: mean internal_state_mean[0] over time for Normal vs No Homeostasis.
    Panel B: mean internal_state_std[0] over time — population-level variance.
    """
    exp_dir = PROJECT_ROOT / "experiments"
    conditions = {
        "normal": ("Normal", "#000000", "-"),
        "no_homeostasis": ("No Homeostasis", "#009E73", "--"),
    }

    # Load JSON data and extract internal_state trajectories
    cond_data: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for cond, (_label, _color, _ls) in conditions.items():
        path = exp_dir / f"final_graph_{cond}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            return
        results = load_json(path)
        step_vals: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in results:
            for s in r["samples"]:
                is_mean = s.get("internal_state_mean", [0, 0, 0, 0])
                is_std = s.get("internal_state_std", [0, 0, 0, 0])
                step_vals[s["step"]].append((is_mean[0], is_std[0]))
        cond_data[cond] = step_vals

    # Check that data has non-zero internal_state values
    sample_vals = list(cond_data["normal"].values())
    if not sample_vals or all(v[0] == 0.0 for v in sample_vals[0]):
        print("  SKIP: internal_state_mean data is all zeros (regenerate JSONs)")
        return

    def _plot_panel(ax, val_index: int, conditions, cond_data):
        """Plot one panel of the homeostasis figure (mean±SEM for each condition)."""
        for cond, (label, color, ls) in conditions.items():
            steps = sorted(cond_data[cond].keys())
            means = []
            sems = []
            for step in steps:
                vals = [v[val_index] for v in cond_data[cond][step]]
                arr = np.array(vals)
                means.append(arr.mean())
                sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) >= 2 else 0.0)
            means_arr = np.array(means)
            sems_arr = np.array(sems)
            lw = 2.0 if cond == "normal" else 1.2
            ax.plot(steps, means_arr, color=color, linewidth=lw, linestyle=ls, label=label)
            ax.fill_between(
                steps,
                means_arr - sems_arr,
                means_arr + sems_arr,
                color=color,
                alpha=0.15,
            )
        ax.set_xlabel("Simulation Step")
        ax.set_xlim(0, 2000)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Panel A: internal_state_mean[0] over time
    _plot_panel(axes[0], 0, conditions, cond_data)
    axes[0].set_ylabel("Internal State Mean [0]")
    axes[0].set_title("(A) Homeostatic Regulation", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower left", fontsize=7)

    # Panel B: internal_state_std[0] over time
    _plot_panel(axes[1], 1, conditions, cond_data)
    axes[1].set_ylabel("Internal State Std [0]")
    axes[1].set_title("(B) Population Variance", fontsize=9)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_homeostasis.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_homeostasis.pdf'}")
