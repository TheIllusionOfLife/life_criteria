"""Figure: Candidate B (kin-sensing) stress-test — 2×2 panels per regime.

Famine panels
-------------
  A  Population trajectories (median + IQR) — all 4 conditions, vertical line at shift
  B  Post-shock AUC boxplot — 4 conditions with significance annotations

Boom-bust panels
----------------
  C  Population trajectories — all 4 conditions, shaded bust phases
  D  Kin-fraction evolution trajectory (mean across seeds)

Data sources
------------
  experiments/candidateB_famine_*.json
  experiments/candidateB_boom_bust_*.json
  experiments/candidateB_stress_analysis.json  (for annotations)
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Okabe-Ito colors for Candidate B conditions
# ---------------------------------------------------------------------------

_COLORS = {
    "baseline": "#000000",
    "candidateB_on": "#0072B2",
    "candidateB_ablated": "#D55E00",
    "sham": "#CC79A7",
}

_LABELS = {
    "baseline": "Baseline",
    "candidateB_on": "+Kin-Sensing",
    "candidateB_ablated": "+Kin-Sensing (ablated)",
    "sham": "Sham",
}

_COND_ORDER = ["baseline", "sham", "candidateB_ablated", "candidateB_on"]

_EXP_DIR = PROJECT_ROOT / "experiments"

_FAMINE_SHIFT_STEP = 3_000
_CYCLE_PERIOD = 2_500

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_stress_conditions(regime: str) -> dict[str, list[dict]]:
    """Load all 4 condition JSONs for a given regime."""
    result = {}
    for cond in _COND_ORDER:
        path = _EXP_DIR / f"candidateB_{regime}_{cond}.json"
        if path.exists():
            with open(path) as f:
                result[cond] = json.load(f)
        else:
            result[cond] = []
    return result


def _trajectories(
    results: list[dict], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, median, q25, q75) across seeds for `field`."""
    step_vals: dict[int, list[float]] = defaultdict(list)
    for r in results:
        for s in r.get("samples", []):
            v = s.get(field)
            if v is not None:
                step_vals[int(s["step"])].append(float(v))

    if not step_vals:
        return np.array([]), np.array([]), np.array([]), np.array([])

    steps = np.array(sorted(step_vals.keys()))
    medians = np.array([np.median(step_vals[st]) for st in steps])
    q25s = np.array([np.percentile(step_vals[st], 25) for st in steps])
    q75s = np.array([np.percentile(step_vals[st], 75) for st in steps])
    return steps, medians, q25s, q75s


def _post_shock_auc(result: dict, shock_step: int = _FAMINE_SHIFT_STEP) -> float:
    return float(
        sum(s["alive_count"] for s in result.get("samples", []) if s["step"] >= shock_step)
    )


# ---------------------------------------------------------------------------
# Famine panels
# ---------------------------------------------------------------------------


def _panel_famine_trajectories(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel A: Population trajectories for famine regime."""
    for cond in _COND_ORDER:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "alive_count")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.12)
        ax.plot(steps, med, color=color, linewidth=1.5, label=_LABELS[cond])

    ax.axvline(
        _FAMINE_SHIFT_STEP,
        color="#888888",
        linewidth=1.0,
        linestyle="--",
        label=f"Famine onset ({_FAMINE_SHIFT_STEP:,d})",
    )
    ax.set_ylabel("Alive count (median)")
    ax.set_xlabel("Simulation step")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_famine_boxplot(ax, conditions: dict[str, list[dict]], analysis: dict | None) -> None:
    """Panel B: Post-shock AUC boxplot with significance annotations."""
    data = []
    labels = []
    colors = []
    plotted_conds: list[str] = []
    for cond in _COND_ORDER:
        results = conditions.get(cond, [])
        if not results:
            continue
        aucs = [_post_shock_auc(r) for r in results]
        data.append(aucs)
        labels.append(_LABELS[cond])
        colors.append(_COLORS[cond])
        plotted_conds.append(cond)

    if not data:
        return

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Strip plot overlay
    for xi, (vals, color) in enumerate(zip(data, colors, strict=True)):
        rng = np.random.default_rng(seed=42 + xi)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + 1 + jitter, vals, color=color, s=10, alpha=0.6, zorder=3)

    # Significance annotation from analysis
    if analysis and "baseline" in plotted_conds and "candidateB_on" in plotted_conds:
        famine_data = analysis.get("famine", {})
        comp = famine_data.get("pairwise_vs_baseline", {})
        cb_on = comp.get("candidateB_on", {})
        p_adj = cb_on.get("vs_baseline_mwu_p_adj")
        d = cb_on.get("vs_baseline_cohen_d")
        if p_adj is not None and not np.isnan(p_adj):
            if p_adj < 0.001:
                sig = "***"
            elif p_adj < 0.01:
                sig = "**"
            elif p_adj < 0.05:
                sig = "*"
            else:
                sig = "ns"
            all_vals = [v for vals in data if vals for v in vals]
            y_max = max(all_vals) if all_vals else 0
            y_ann = y_max * 1.05
            baseline_xi = plotted_conds.index("baseline") + 1
            cb_xi = plotted_conds.index("candidateB_on") + 1
            ax.annotate(
                "",
                xy=(cb_xi, y_ann),
                xytext=(baseline_xi, y_ann),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.0),
            )
            d_str = f"d={d:.2f}" if d is not None else ""
            ax.text(
                (cb_xi + baseline_xi) / 2,
                y_ann * 1.01,
                f"{sig} ({d_str})" if d_str else sig,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Post-shock AUC (steps 3k-10k)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=7)


# ---------------------------------------------------------------------------
# Boom-bust panels
# ---------------------------------------------------------------------------


def _panel_boom_bust_trajectories(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel C: Population trajectories with bust-phase shading."""
    n_cycles = 10_000 // (_CYCLE_PERIOD * 2)
    for i in range(n_cycles):
        bust_start = _CYCLE_PERIOD * (2 * i + 1)
        bust_end = _CYCLE_PERIOD * (2 * i + 2)
        ax.axvspan(bust_start, bust_end, color="#FFE0E0", alpha=0.4, zorder=0)

    for cond in _COND_ORDER:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "alive_count")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.12)
        ax.plot(steps, med, color=color, linewidth=1.5, label=_LABELS[cond])

    ax.set_ylabel("Alive count (median)")
    ax.set_xlabel("Simulation step")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_kin_fraction_trajectory(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel D: Kin-fraction trajectory (mean across seeds) — shows signal evolution."""
    for cond in _COND_ORDER:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "kin_fraction_mean")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.12)
        ax.plot(steps, med, color=color, linewidth=1.5, label=_LABELS[cond])

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Kin fraction (median)")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_candidateB_stress() -> None:
    """Generate Candidate B stress-test figure: 2×2 panels."""
    famine = _load_stress_conditions("famine")
    boom_bust = _load_stress_conditions("boom_bust")

    has_famine = any(len(v) > 0 for v in famine.values())
    has_boom_bust = any(len(v) > 0 for v in boom_bust.values())

    if not has_famine and not has_boom_bust:
        print("  SKIP: no Candidate B stress-test data found")
        return

    # Load analysis annotations
    analysis_path = _EXP_DIR / "candidateB_stress_analysis.json"
    analysis: dict | None = None
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)

    if has_famine and has_boom_bust:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
        (ax_a, ax_b), (ax_c, ax_d) = axes

        _panel_famine_trajectories(ax_a, famine)
        ax_a.set_title("A  Famine: population dynamics", loc="left", fontsize=9)

        _panel_famine_boxplot(ax_b, famine, analysis)
        ax_b.set_title("B  Famine: post-shock survival", loc="left", fontsize=9)

        _panel_boom_bust_trajectories(ax_c, boom_bust)
        ax_c.set_title("C  Boom-bust: population dynamics", loc="left", fontsize=9)

        _panel_kin_fraction_trajectory(ax_d, boom_bust)
        ax_d.set_title("D  Boom-bust: kin-fraction evolution", loc="left", fontsize=9)

        n_str = _n_seeds_str(famine, boom_bust)
        fig.suptitle(f"Candidate B: Kin-Sensing Under Harsh Perturbations ({n_str})", fontsize=10)

    elif has_famine:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.2))

        _panel_famine_trajectories(ax_a, famine)
        ax_a.set_title("A  Famine: population dynamics", loc="left", fontsize=9)

        _panel_famine_boxplot(ax_b, famine, analysis)
        ax_b.set_title("B  Famine: post-shock survival", loc="left", fontsize=9)

        n_str = _n_seeds_str(famine)
        fig.suptitle(f"Candidate B: Famine Regime ({n_str})", fontsize=10)

    else:
        fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(7.5, 3.2))

        _panel_boom_bust_trajectories(ax_c, boom_bust)
        ax_c.set_title("C  Boom-bust: population dynamics", loc="left", fontsize=9)

        _panel_kin_fraction_trajectory(ax_d, boom_bust)
        ax_d.set_title("D  Boom-bust: kin-fraction evolution", loc="left", fontsize=9)

        n_str = _n_seeds_str(boom_bust=boom_bust)
        fig.suptitle(f"Candidate B: Boom-Bust Regime ({n_str})", fontsize=10)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_candidateB_stress.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _n_seeds_str(
    famine: dict[str, list[dict]] | None = None,
    boom_bust: dict[str, list[dict]] | None = None,
) -> str:
    """Build a concise seed count string for figure title."""
    parts = []
    if famine:
        n = max((len(v) for v in famine.values()), default=0)
        if n > 0:
            parts.append(f"famine n={n}")
    if boom_bust:
        n = max((len(v) for v in boom_bust.values()), default=0)
        if n > 0:
            parts.append(f"boom-bust n={n}")
    return ", ".join(parts) if parts else "no data"
