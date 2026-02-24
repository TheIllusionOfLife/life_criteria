"""Figure 1: Architecture diagram showing two-layer hierarchy."""

from figures._shared import *


def _draw_environment(ax: plt.Axes) -> None:
    # Environment box
    env_rect = mpatches.FancyBboxPatch(
        (0.3, 0.3),
        9.4,
        5.9,
        boxstyle="round,pad=0.1",
        facecolor="#F5F5F5",
        edgecolor="#333333",
        linewidth=1.5,
    )
    ax.add_patch(env_rect)
    ax.text(
        5.0,
        5.85,
        "Environment (Toroidal 2D, 100$\\times$100)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    # Resource field
    ax.text(
        1.2,
        5.35,
        "Resource Field",
        ha="left",
        va="center",
        fontsize=7,
        fontstyle="italic",
        color="#666666",
    )


def _draw_organism_box(ax: plt.Axes, ox: float, oy: float) -> None:
    org_rect = mpatches.FancyBboxPatch(
        (ox, oy),
        4.8,
        4.2,
        boxstyle="round,pad=0.08",
        facecolor="#FFFFFF",
        edgecolor="#0072B2",
        linewidth=1.2,
    )
    ax.add_patch(org_rect)
    ax.text(
        ox + 2.4,
        oy + 3.85,
        "Organism (10-50 per environment)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#0072B2",
    )


def _draw_organism_components(ax: plt.Axes, ox: float, oy: float) -> None:
    # Internal components (wider boxes for the single organism)
    components = [
        ("Genome\n(7 segments, 256 floats)", ox + 0.2, oy + 2.7, 2.1, 0.85, "#E69F00"),
        ("NN Controller\n(8>16>4, 212 wt)", ox + 2.5, oy + 2.7, 2.1, 0.85, "#009E73"),
        (
            "Graph Metabolism\n(2-4 nodes, directed)",
            ox + 0.2,
            oy + 1.3,
            2.1,
            0.85,
            "#D55E00",
        ),
        (
            "Boundary Maintenance\n(10-50 swarm agents)",
            ox + 2.5,
            oy + 1.3,
            2.1,
            0.85,
            "#56B4E9",
        ),
    ]

    for label, cx, cy, w, h, color in components:
        comp_rect = mpatches.FancyBboxPatch(
            (cx, cy),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="none",
            alpha=0.2,
        )
        ax.add_patch(comp_rect)
        comp_border = mpatches.FancyBboxPatch(
            (cx, cy),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
        )
        ax.add_patch(comp_border)
        ax.text(
            cx + w / 2,
            cy + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="#333333",
        )


def _draw_organism_arrows(ax: plt.Axes, ox: float, oy: float) -> None:
    # Arrows: Genome -> NN, Genome -> Metabolism
    ax.annotate(
        "",
        xy=(ox + 2.5, oy + 3.12),
        xytext=(ox + 2.3, oy + 3.12),
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(ox + 1.25, oy + 2.7),
        xytext=(ox + 1.25, oy + 2.15),
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    # NN -> Boundary (response to stimuli)
    ax.annotate(
        "",
        xy=(ox + 3.55, oy + 2.7),
        xytext=(ox + 3.55, oy + 2.15),
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    # Metabolism <-> Boundary (energy <-> integrity)
    ax.annotate(
        "",
        xy=(ox + 2.5, oy + 1.72),
        xytext=(ox + 2.3, oy + 1.72),
        arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8),
    )


def _draw_internal_state_label(ax: plt.Axes, ox: float, oy: float) -> None:
    # Internal state label
    ax.text(
        ox + 2.4,
        oy + 0.45,
        "Internal State Vector (homeostatic regulation)",
        ha="center",
        va="center",
        fontsize=6,
        fontstyle="italic",
        color="#666666",
    )


def _draw_criteria_sidebar(ax: plt.Axes, sidebar_x: float) -> None:
    # Criteria mapping sidebar (right side, clearly separated)
    sidebar_rect = mpatches.FancyBboxPatch(
        (sidebar_x, 0.8),
        3.2,
        4.2,
        boxstyle="round,pad=0.08",
        facecolor="#FFFFFF",
        edgecolor="#999999",
        linewidth=0.8,
        linestyle="--",
    )
    ax.add_patch(sidebar_rect)
    ax.text(
        sidebar_x + 1.6,
        4.65,
        "7 Criteria",
        fontsize=8,
        fontweight="bold",
        ha="center",
        color="#333333",
    )

    criteria_items = [
        ("(1) Cellular Org.", "#56B4E9"),
        ("(2) Metabolism", "#D55E00"),
        ("(3) Homeostasis", "#009E73"),
        ("(4) Growth/Dev.", "#CC79A7"),
        ("(5) Reproduction", "#0072B2"),
        ("(6) Response", "#009E73"),
        ("(7) Evolution", "#E69F00"),
    ]
    for j, (label, color) in enumerate(criteria_items):
        yy = 4.25 - j * 0.47
        ax.plot(sidebar_x + 0.3, yy, "s", color=color, markersize=5)
        ax.text(sidebar_x + 0.55, yy, label, fontsize=6.5, color="#333333", va="center")


def generate_architecture() -> None:
    """Figure 1: Architecture diagram showing two-layer hierarchy."""
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_environment(ax)

    # Single organism shown in detail (left side)
    ox, oy = 0.8, 0.8
    _draw_organism_box(ax, ox, oy)
    _draw_organism_components(ax, ox, oy)
    _draw_organism_arrows(ax, ox, oy)
    _draw_internal_state_label(ax, ox, oy)

    # Criteria mapping sidebar (right side, clearly separated)
    _draw_criteria_sidebar(ax, 6.2)

    fig.savefig(FIG_DIR / "fig_architecture.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_architecture.pdf'}")
