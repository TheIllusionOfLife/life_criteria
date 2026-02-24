"""Figure 14: Orthogonal outcome metrics (spatial cohesion and median lifespan)."""

import numpy as np
from figures._shared import *


def plot_violin_strip(
    ax,
    data: dict[str, list[float]],
    title: str | None,
    ylabel: str,
    baseline_fmt: str = ".1f",
    seed: int = 0,
) -> None:
    """Plot violin distributions with overlaid strip plot and baseline."""
    positions = list(range(len(CONDITION_ORDER)))
    data_list = [np.array(data[c]) for c in CONDITION_ORDER]

    parts = ax.violinplot(
        data_list,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        color = COLORS[CONDITION_ORDER[i]]
        body.set_facecolor(color)
        body.set_alpha(0.3)
        body.set_edgecolor(color)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(seed)
    for i, condition in enumerate(CONDITION_ORDER):
        vals = np.array(data[condition])
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=8,
            alpha=0.6,
            color=COLORS[condition],
            edgecolors="none",
            zorder=5,
        )
        ax.scatter(
            i,
            np.median(vals),
            s=30,
            color=COLORS[condition],
            edgecolors="white",
            linewidths=0.8,
            zorder=10,
            marker="D",
        )

    baseline_mean = float(np.mean(data.get("normal", [0])))
    ax.axhline(
        y=baseline_mean,
        color="#000000",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label=f"Normal mean ({baseline_mean:{baseline_fmt}})",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in CONDITION_ORDER], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_orthogonal() -> None:
    """Figure 14: orthogonal outcome metrics.

    Plots spatial cohesion and median lifespan per condition.
    """
    exp_dir = PROJECT_ROOT / "experiments"

    # Panel A: spatial cohesion from final_graph_{condition}.json (last sample)
    # Panel B: median lifespan from lifespans list
    spatial_data: dict[str, list[float]] = {}
    lifespan_data: dict[str, list[float]] = {}

    for condition in CONDITION_ORDER:
        path = exp_dir / f"final_graph_{condition}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            return
        results = load_json(path)
        cohesions = []
        medians = []
        for r in results:
            if r.get("samples"):
                last = r["samples"][-1]
                cohesions.append(last.get("spatial_cohesion_mean", 0.0))
            ls = r.get("lifespans", [])
            if ls:
                medians.append(float(sorted(ls)[len(ls) // 2]))
            else:
                medians.append(0.0)
        spatial_data[condition] = cohesions
        lifespan_data[condition] = medians

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    plot_violin_strip(
        axes[0],
        spatial_data,
        title="(A) Spatial Cohesion",
        ylabel="Spatial Cohesion",
        baseline_fmt=".1f",
        seed=0,
    )

    plot_violin_strip(
        axes[1],
        lifespan_data,
        title="(B) Median Lifespan",
        ylabel="Median Lifespan (steps)",
        baseline_fmt=".0f",
        seed=1,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_orthogonal.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_orthogonal.pdf'}")
