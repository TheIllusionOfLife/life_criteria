"""Figure 2: Population dynamics time-series."""

from collections import defaultdict

import numpy as np
from figures._shared import *


def generate_timeseries(data: list[dict]) -> None:
    """Figure 2: Population dynamics time-series with confidence bands."""
    # Group by (condition, step) → list of alive_count values
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in data:
        key = (row["condition"], int(row["step"]))
        groups[key].append(row["alive_count"])

    fig, ax = plt.subplots(figsize=(7, 3.2))

    for condition in CONDITION_ORDER:
        steps = sorted({s for (c, s) in groups if c == condition})
        means = []
        sems = []
        for step in steps:
            vals = groups[(condition, step)]
            arr = np.array(vals)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) >= 2 else 0.0)

        means = np.array(means)
        sems = np.array(sems)
        color = COLORS[condition]
        lw = 2.0 if condition == "normal" else 1.2
        ls = "-" if condition == "normal" else "--"
        ax.plot(
            steps,
            means,
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=LABELS[condition],
            zorder=10 if condition == "normal" else 5,
        )
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Alive Count ($n$=30)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, edgecolor="0.8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig_timeseries.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_timeseries.pdf'}")
