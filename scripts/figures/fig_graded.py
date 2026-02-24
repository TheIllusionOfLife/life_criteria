"""Figure 7: Graded ablation dose-response curve."""

import numpy as np
from figures._shared import *


def generate_graded() -> None:
    """Figure 7: Graded ablation dose-response curve."""
    exp_dir = PROJECT_ROOT / "experiments"
    levels = [1.0, 0.75, 0.5, 0.25, 0.0]

    level_data: dict[float, np.ndarray] = {}
    for level in levels:
        path = exp_dir / f"graded_graded_{level:.2f}.json"
        if not path.exists():
            print(f"  SKIP: {path} not found")
            return
        results = load_json(path)
        level_data[level] = np.array([r["final_alive_count"] for r in results if "samples" in r])

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    medians = [float(np.median(level_data[lv])) for lv in levels]
    q25s = [float(np.percentile(level_data[lv], 25)) for lv in levels]
    q75s = [float(np.percentile(level_data[lv], 75)) for lv in levels]

    ax.plot(levels, medians, "o-", color="#0072B2", linewidth=1.5, markersize=5)
    ax.fill_between(levels, q25s, q75s, color="#0072B2", alpha=0.2)

    ax.set_xlabel("Metabolism Efficiency Multiplier")
    ax.set_ylabel("Final Alive Count ($N_T$)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_graded.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_graded.pdf'}")
