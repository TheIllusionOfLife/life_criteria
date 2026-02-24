"""Figures 8 and 13: Cyclic environment and cyclic period sweep."""

from collections import defaultdict

import numpy as np
from figures._shared import *


def generate_cyclic() -> None:
    """Figure 8: Cyclic environment — population dynamics under periodic stress."""
    exp_dir = PROJECT_ROOT / "experiments"

    conditions = {
        "cyclic_evo_on": ("Evolution On", "#000000", "-"),
        "cyclic_evo_off": ("Evolution Off", "#CC79A7", "--"),
    }

    cond_data: dict[str, dict[int, list[float]]] = {}
    for cond in conditions:
        path = exp_dir / f"cyclic_{cond}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            return
        results = load_json(path)
        step_vals: dict[int, list[float]] = defaultdict(list)
        for r in results:
            for s in r["samples"]:
                step_vals[s["step"]].append(s["alive_count"])
        cond_data[cond] = step_vals

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for cond, (label, color, ls) in conditions.items():
        steps = sorted(cond_data[cond].keys())
        means = [np.mean(cond_data[cond][s]) for s in steps]
        sems = [
            np.std(cond_data[cond][s], ddof=1) / np.sqrt(len(cond_data[cond][s])) for s in steps
        ]
        means, sems = np.array(means), np.array(sems)
        lw = 2.0 if "evo_on" in cond else 1.2
        ax.plot(steps, means, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15)

    # Mark cycle boundaries
    cycle_period = 2000
    for i in range(1, 6):
        ax.axvline(x=i * cycle_period, color="#888888", linestyle=":", linewidth=0.5)

    # Shade low-rate phases
    max_step = max(max(cond_data[c].keys()) for c in conditions)
    phase = 0
    for start in range(0, int(max_step) + 1, cycle_period):
        if phase % 2 == 1:
            ax.axvspan(start, min(start + cycle_period, max_step), color="#FF0000", alpha=0.03)
        phase += 1

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Alive Count ($n$=30)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cyclic.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_cyclic.pdf'}")


def generate_cyclic_sweep() -> None:
    """Update cyclic figure with period sweep data."""
    exp_dir = PROJECT_ROOT / "experiments"
    periods = [500, 1000, 2000, 5000]

    # Collect final alive counts per period x condition
    period_data: dict[int, dict[str, list[float]]] = {}
    for period in periods:
        period_data[period] = {}
        for evo_label in ["evo_on", "evo_off"]:
            path = exp_dir / f"cyclic_sweep_sweep_p{period}_{evo_label}.json"
            if not path.exists():
                print(f"  SKIP: {path} not found")
                return
            results = load_json(path)
            period_data[period][evo_label] = [
                r["final_alive_count"] for r in results if "samples" in r
            ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    x = np.arange(len(periods))
    width = 0.35

    def safe_mean(arr: list[float]) -> float:
        return float(np.mean(arr)) if len(arr) >= 1 else np.nan

    def safe_sem(arr: list[float]) -> float:
        if len(arr) < 2:
            return 0.0
        return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    evo_on_means = [safe_mean(period_data[p]["evo_on"]) for p in periods]
    evo_on_sems = [safe_sem(period_data[p]["evo_on"]) for p in periods]
    evo_off_means = [safe_mean(period_data[p]["evo_off"]) for p in periods]
    evo_off_sems = [safe_sem(period_data[p]["evo_off"]) for p in periods]

    ax.bar(
        x - width / 2,
        evo_on_means,
        width,
        yerr=evo_on_sems,
        label="Evolution On",
        color="#000000",
        alpha=0.6,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        evo_off_means,
        width,
        yerr=evo_off_sems,
        label="Evolution Off",
        color="#CC79A7",
        alpha=0.6,
        capsize=3,
    )

    ax.set_xlabel("Cycle Period (steps)")
    ax.set_ylabel("Final Alive Count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in periods])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cyclic_sweep.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_cyclic_sweep.pdf'}")
