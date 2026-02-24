"""Figure 3: Proxy control comparison — 3 metabolism modes."""

from collections import defaultdict

import numpy as np
from figures._shared import *

PROXY_COLORS = {
    "counter": "#56B4E9",  # sky blue
    "toy": "#E69F00",  # orange
    "graph": "#009E73",  # bluish green
}

PROXY_LABELS = {
    "counter": "Counter (minimal)",
    "toy": "Toy (single-step + waste)",
    "graph": "Graph (multi-step network)",
}


def generate_proxy() -> None:
    """Figure 3: Proxy control comparison — 3 metabolism modes."""
    exp_dir = PROJECT_ROOT / "experiments"
    modes = ["counter", "toy", "graph"]

    # Collect time-series data per mode, skipping missing files
    available_modes = []
    mode_data: dict[str, dict[int, list[float]]] = {}
    for mode in modes:
        path = exp_dir / f"proxy_{mode}.json"
        if not path.exists():
            print(f"  SKIP mode '{mode}': {path} not found")
            continue
        available_modes.append(mode)
        results = load_json(path)
        step_vals: dict[int, list[float]] = defaultdict(list)
        for r in results:
            for s in r["samples"]:
                step_vals[s["step"]].append(s["alive_count"])
        mode_data[mode] = step_vals

    if len(available_modes) < 2:
        print("  SKIP figure: need at least 2 modes for comparison")
        return
    modes = available_modes

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))

    # Panel 1: Alive count time-series
    ax = axes[0]
    for mode in modes:
        steps = sorted(mode_data[mode].keys())
        means = [np.mean(mode_data[mode][s]) for s in steps]
        sems = [
            np.std(mode_data[mode][s], ddof=1) / np.sqrt(len(mode_data[mode][s])) for s in steps
        ]
        means, sems = np.array(means), np.array(sems)
        ax.plot(steps, means, color=PROXY_COLORS[mode], label=PROXY_LABELS[mode])
        ax.fill_between(steps, means - sems, means + sems, color=PROXY_COLORS[mode], alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Alive Count")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Final alive count boxplot
    ax = axes[1]
    final_alive = {
        m: np.array([r["final_alive_count"] for r in load_json(exp_dir / f"proxy_{m}.json")])
        for m in modes
    }
    bp = ax.boxplot(
        [final_alive[m] for m in modes],
        tick_labels=[m.capitalize() for m in modes],
        patch_artist=True,
        widths=0.6,
    )
    for patch, mode in zip(bp["boxes"], modes, strict=True):
        patch.set_facecolor(PROXY_COLORS[mode])
        patch.set_alpha(0.4)
    ax.set_ylabel("Final Alive")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Genome diversity boxplot (reuse loaded data from panel 2)
    ax = axes[2]
    final_div = {
        m: np.array(
            [
                r["samples"][-1].get("genome_diversity", 0)
                for r in load_json(exp_dir / f"proxy_{m}.json")
            ]
        )
        for m in modes
    }
    bp = ax.boxplot(
        [final_div[m] for m in modes],
        tick_labels=[m.capitalize() for m in modes],
        patch_artist=True,
        widths=0.6,
    )
    for patch, mode in zip(bp["boxes"], modes, strict=True):
        patch.set_facecolor(PROXY_COLORS[mode])
        patch.set_alpha(0.4)
    ax.set_ylabel("Genome Diversity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shared legend
    handles = [mlines.Line2D([], [], color=PROXY_COLORS[m], label=PROXY_LABELS[m]) for m in modes]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_proxy.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_proxy.pdf'}")
