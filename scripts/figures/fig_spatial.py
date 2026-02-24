"""Figure 11: Spatial cohesion under boundary ablation."""

import numpy as np
from figures._shared import *


def generate_spatial() -> None:
    """Figure 11: Spatial cohesion under boundary ablation."""
    exp_dir = PROJECT_ROOT / "experiments"

    conditions = {
        "normal": ("Normal", "#000000"),
        "no_boundary": ("No Boundary", "#56B4E9"),
    }

    cond_data: dict[str, list[float]] = {}
    for cond, (_label, _color) in conditions.items():
        path = exp_dir / f"spatial_{cond}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            return
        results = load_json(path)
        # Collect final spatial_cohesion_mean from each seed
        finals = []
        for r in results:
            if r.get("samples"):
                last = r["samples"][-1]
                finals.append(last.get("spatial_cohesion_mean", 0.0))
        cond_data[cond] = finals

    if not all(cond_data.values()):
        print("  SKIP: insufficient spatial data")
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    cond_list = list(conditions.keys())
    data_list = [np.array(cond_data[c]) for c in cond_list]
    labels = [conditions[c][0] for c in cond_list]
    colors = [conditions[c][1] for c in cond_list]

    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Overlay individual points
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data_list, colors, strict=True)):
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(
            i + 1 + jitter,
            vals,
            s=12,
            alpha=0.5,
            color=color,
            edgecolors="none",
            zorder=5,
        )

    ax.set_ylabel("Spatial Cohesion (mean pairwise dist.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_spatial.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_spatial.pdf'}")
