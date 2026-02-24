"""Figure 19: Ecology stressor conditions — alive-count time series."""

from collections import defaultdict

import numpy as np
from figures._shared import *


def generate_ecology_stress() -> None:
    """Figure 19: Ecology stressor conditions — alive-count time series."""
    exp_dir = PROJECT_ROOT / "experiments"

    condition_specs = {
        "normal": ("Normal", "#000000", "-"),
        "resource_shift": ("Resource Shift", "#E69F00", "--"),
        "cyclic_stress": ("Cyclic Stress", "#D55E00", "-."),
        "cyclic_stress_no_evolution": ("Cyclic Stress (No Evo)", "#CC79A7", ":"),
    }

    # Load data for all conditions
    cond_data: dict[str, dict[int, list[float]]] = {}
    for cond in condition_specs:
        path = exp_dir / f"ecology_stress_{cond}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        results = load_json(path)
        step_vals: dict[int, list[float]] = defaultdict(list)
        for r in results:
            for s in r.get("samples", []):
                step_vals[int(s["step"])].append(float(s["alive_count"]))
        cond_data[cond] = step_vals

    if not cond_data:
        print("  SKIP: no ecology stress data found")
        return

    fig, ax = plt.subplots(figsize=(7, 3.2))

    for cond, (label, color, ls) in condition_specs.items():
        if cond not in cond_data:
            continue
        steps = sorted(cond_data[cond].keys())
        means = [np.mean(cond_data[cond][s]) for s in steps]
        sems = [
            np.std(cond_data[cond][s], ddof=1) / np.sqrt(len(cond_data[cond][s]))
            if len(cond_data[cond][s]) >= 2
            else 0.0
            for s in steps
        ]
        means_arr = np.array(means)
        sems_arr = np.array(sems)
        lw = 2.0 if cond == "normal" else 1.2
        ax.plot(steps, means_arr, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill_between(steps, means_arr - sems_arr, means_arr + sems_arr, color=color, alpha=0.12)

    # Mark resource-shift event at step 1000
    ax.axvline(
        x=1000, color="#888888", linestyle=":", linewidth=1.0, label="Resource shift (step 1000)"
    )

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Alive Count ($n$=30)")
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ecology_stress.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_ecology_stress.pdf'}")
