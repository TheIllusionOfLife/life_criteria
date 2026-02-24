"""Figure 10: Criterion coupling graph."""

import numpy as np
from figures._shared import *


def _draw_coupling_nodes(
    ax: plt.Axes, positions: dict[str, tuple[float, float]], node_colors: list[str]
) -> None:
    """Draw circular nodes for each criterion."""
    for i, (name, (x, y)) in enumerate(positions.items()):
        circle = plt.Circle(
            (x, y),
            0.18,
            facecolor=node_colors[i],
            alpha=0.3,
            edgecolor=node_colors[i],
            linewidth=1.2,
        )
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=5.5, fontweight="bold")


def _draw_coupling_edges(
    ax: plt.Axes,
    pairs: list[dict],
    positions: dict[str, tuple[float, float]],
    short_names: dict[str, str],
) -> None:
    """Draw directed edges representing significant coupling correlations."""
    for pair in pairs:
        var_a = short_names.get(pair["var_a"], pair["var_a"])
        var_b = short_names.get(pair["var_b"], pair["var_b"])
        r, lag = get_coupling_best(pair)
        if r is None or lag is None:
            continue

        if var_a not in positions or var_b not in positions:
            continue

        x1, y1 = positions[var_a]
        x2, y2 = positions[var_b]

        # Shorten arrow to not overlap nodes
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.01:
            continue
        shrink = 0.22 / dist
        sx1, sy1 = x1 + dx * shrink, y1 + dy * shrink
        sx2, sy2 = x2 - dx * shrink, y2 - dy * shrink

        lw = max(0.5, min(3.0, abs(r) * 4))
        color = "#0072B2" if r > 0 else "#D55E00"
        ax.annotate(
            "",
            xy=(sx2, sy2),
            xytext=(sx1, sy1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        )

        # Label at midpoint
        mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        ax.text(
            mx,
            my + 0.08,
            f"r={r:.2f}\nlag={lag}",
            ha="center",
            va="center",
            fontsize=5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8),
        )


def _draw_intervention_effects(ax: plt.Axes, analysis: dict) -> None:
    """Draw text box summarizing key intervention effects."""
    interventions = analysis.get("intervention_effects", {})
    if not interventions:
        return

    matrix = interventions.get("matrix", [])
    effects_summary = []
    for row in matrix:
        criterion = row["ablated_criterion"]
        metric_effects = []
        max_effect = 0.0
        for key in INTERVENTION_METRICS:
            val = row.get(key, 0.0)
            if abs(val) > 20:
                short_key = key.replace("_mean", "").replace("_0", "")
                sign = "+" if val > 0 else "-"
                metric_effects.append(f"{short_key} {sign}{abs(val):.0f}%")
                max_effect = max(max_effect, abs(val))
        if metric_effects:
            effects_summary.append((max_effect, f"  {criterion}: {', '.join(metric_effects)}"))

    if effects_summary:
        # Show top entries sorted by largest absolute intervention effect.
        effects_summary.sort(key=lambda row: row[0], reverse=True)
        box_text = "Intervention effects:\n" + "\n".join(
            summary for _, summary in effects_summary[:4]
        )
        ax.text(
            -1.45,
            -1.45,
            box_text,
            fontsize=5,
            va="bottom",
            ha="left",
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#F5F5F5",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )


def _draw_design_edges(
    ax: plt.Axes,
    positions: dict[str, tuple[float, float]],
    pairs: list[dict],
    short_names: dict[str, str],
) -> None:
    """Draw dashed edges for designed interactions not captured by data."""
    design_edges = [
        ("Growth/Dev.", "Reproduction", "gate"),
        ("Metabolism", "Cellular Org.", "energy"),
        ("Homeostasis", "Cellular Org.", "repair"),
    ]
    for src, dst, _label in design_edges:
        if src not in positions or dst not in positions:
            continue
        # Check if already drawn from data
        already = any(
            (short_names.get(p["var_a"]) == src and short_names.get(p["var_b"]) == dst)
            or (short_names.get(p["var_a"]) == dst and short_names.get(p["var_b"]) == src)
            for p in pairs
        )
        if already:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.01:
            continue
        shrink = 0.22 / dist
        sx1, sy1 = x1 + dx * shrink, y1 + dy * shrink
        sx2, sy2 = x2 - dx * shrink, y2 - dy * shrink
        ax.annotate(
            "",
            xy=(sx2, sy2),
            xytext=(sx1, sy1),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8, linestyle="--"),
        )


def generate_coupling() -> None:
    """Figure 10: Criterion coupling graph — directed edges with correlation coefficients."""
    analysis_path = PROJECT_ROOT / "experiments" / "coupling_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    pairs = analysis.get("pairs", [])
    if not pairs:
        print("  SKIP: no coupling pairs found")
        return

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    short_names = COUPLING_METRIC_MAPPING.copy()
    # Ensure all metric keys from coupling data have a mapping so no pairs are silently dropped
    for pair in pairs:
        for key in ("var_a", "var_b"):
            metric = pair[key]
            if metric not in short_names:
                # Generate a readable fallback from the metric key
                short_names[metric] = metric.replace("_", " ").title()

    n = len(COUPLING_CRITERIA)
    angles = [2 * np.pi * i / n - np.pi / 2 for i in range(n)]
    positions = {
        name: (np.cos(a), np.sin(a)) for name, a in zip(COUPLING_CRITERIA, angles, strict=True)
    }

    # Draw nodes
    _draw_coupling_nodes(ax, positions, COUPLING_NODE_COLORS)

    # Draw edges from coupling analysis
    _draw_coupling_edges(ax, pairs, positions, short_names)

    # Intervention effects text box (from coupling_analysis.json)
    _draw_intervention_effects(ax, analysis)

    # Design-based edges (from Table 2)
    _draw_design_edges(ax, positions, pairs, short_names)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_coupling.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_coupling.pdf'}")
