"""Figure 18: Implementation invariance — effect sizes for boundary/homeostasis."""

from figures._shared import *


def generate_invariance() -> None:
    """Figure 18: Implementation invariance — effect sizes for boundary/homeostasis."""
    analysis_path = PROJECT_ROOT / "experiments" / "invariance_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    for ax, criterion in zip(axes, ["boundary", "homeostasis"], strict=True):
        effect_default = data[criterion]["effect_default"]
        effect_alt = data[criterion]["effect_alt"]
        direction_consistent = data[criterion]["direction_consistent"]

        ax.bar(
            ["Default\nmode", "Alt.\nmode"],
            [effect_default, effect_alt],
            color=["#0072B2", "#D55E00"],
            alpha=0.7,
            width=0.5,
        )
        ax.axhline(y=0, color="#888888", linewidth=0.8, linestyle="--")

        if direction_consistent:
            ax.text(
                0.5,
                0.95,
                "[+] Direction consistent",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color="#009E73",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", edgecolor="#009E73", alpha=0.9
                ),
            )

        ax.set_title(f"({criterion.capitalize()})", fontsize=9)
        ax.set_ylabel("Ablation effect (alive count drop)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared baseline annotation
    baseline_default = data["baseline"]["default_modes"]
    baseline_alt = data["baseline"]["alt_modes"]
    fig.text(
        0.5,
        -0.02,
        f"Baseline: default modes={baseline_default:.1f}, alt. modes={baseline_alt:.1f}",
        ha="center",
        fontsize=7,
        color="#666666",
    )

    fig.suptitle("Implementation Invariance: Boundary & Homeostasis", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_invariance.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_invariance.pdf'}")
