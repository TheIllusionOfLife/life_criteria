"""Figure 17: Mid-run vs step-0 ablation comparison."""

import numpy as np
from figures._shared import *


def generate_midrun_ablation() -> None:
    """Figure 17: Mid-run vs step-0 ablation comparison — grouped bar chart."""
    analysis_path = PROJECT_ROOT / "experiments" / "midrun_ablation_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path) as f:
        import json

        data = json.load(f)

    criteria_list = data.get("criteria", [])
    if not criteria_list:
        print("  SKIP: no criteria data in midrun_ablation_analysis.json")
        return

    criterion_names = [c["criterion"] for c in criteria_list]
    step0_means = [c["step0_mean"] for c in criteria_list]
    midrun_means = [c["midrun_mean"] for c in criteria_list]
    normal_mean = criteria_list[0]["normal_mean"]

    x = np.arange(len(criterion_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.2))

    ax.bar(x - width / 2, step0_means, width, label="Step-0 ablation", color="#0072B2", alpha=0.7)
    ax.bar(x + width / 2, midrun_means, width, label="Mid-run ablation", color="#D55E00", alpha=0.7)

    # Normal mean baseline
    ax.axhline(
        y=normal_mean,
        color="#000000",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label=f"Normal mean ({normal_mean:.1f})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in criterion_names], rotation=30, ha="right"
    )
    ax.set_ylabel("Final Alive Count ($N_T$)")
    ax.set_title("Mid-run vs. Step-0 Ablation", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_midrun_ablation.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_midrun_ablation.pdf'}")
