"""Figure: Candidate space taxonomy diagram.

Conceptual diagram showing where the two tested candidates sit in
a taxonomy of possible 8th-criterion candidates, alongside untested
alternatives.  Pure literature/conceptual figure — no experiment data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from figures._shared import FIG_DIR


def generate_candidate_space():
    """Generate candidate space taxonomy figure."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Category boxes
    categories = [
        (0.5, 2.5, "Within-Lifetime\nAdaptation", "#E69F00"),
        (2.0, 2.5, "Collective\nOrganization", "#56B4E9"),
        (3.5, 2.5, "Anticipatory\nRegulation", "#009E73"),
    ]

    for x, y, label, color in categories:
        rect = mpatches.FancyBboxPatch(
            (x - 0.6, y - 0.35), 1.2, 0.7,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.3, edgecolor=color, lw=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=7, fontweight="bold")

    # Candidate boxes (tested)
    tested = [
        (0.5, 1.3, "A: EMA Memory\n(null)", "#E69F00"),
        (2.0, 1.3, "B: Kin-Sensing\n(null)", "#56B4E9"),
    ]
    for x, y, label, color in tested:
        rect = mpatches.FancyBboxPatch(
            (x - 0.55, y - 0.3), 1.1, 0.6,
            boxstyle="round,pad=0.08",
            facecolor="white", edgecolor=color, lw=1.5, linestyle="-",
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5)
        # Arrow from category to candidate
        ax.annotate(
            "", xy=(x, y + 0.3), xytext=(x, 2.15),
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
        )

    # Untested candidates (dashed)
    untested = [
        (0.5, 0.3, "Synaptic\nPlasticity"),
        (2.0, 0.3, "Niche\nConstruction"),
        (3.5, 1.3, "Predictive\nCoding"),
        (3.5, 0.3, "Immune\nRecognition"),
    ]
    for x, y, label in untested:
        rect = mpatches.FancyBboxPatch(
            (x - 0.5, y - 0.25), 1.0, 0.5,
            boxstyle="round,pad=0.08",
            facecolor="#f0f0f0", edgecolor="#999999", lw=1, linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=6, color="#666666")

    # Arrows from categories to untested
    for x_cat, x_unt, y_unt in [(0.5, 0.5, 0.55), (2.0, 2.0, 0.55),
                                  (3.5, 3.5, 1.6), (3.5, 3.5, 0.55)]:
        ax.annotate(
            "", xy=(x_unt, y_unt), xytext=(x_cat, min(y_unt + 0.55, 2.15)),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8, linestyle="--"),
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="#333", lw=1, label="Tested (this work)"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#999", lw=0.8, linestyle="--",
                       label="Untested (future)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.9)

    ax.set_title("Candidate Space for an 8th Criterion", fontsize=10, pad=10)

    fig.tight_layout()
    out = FIG_DIR / "fig_candidate_space.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
