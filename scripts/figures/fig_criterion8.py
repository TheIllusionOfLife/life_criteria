"""Figure: 8th Criterion (Learning/Memory) — 4-panel results.

Panels
------
  A  Survival AUC — violin + strip plot comparing 4 conditions
  B  alive_count trajectories — baseline vs criterion8_on (median + IQR ribbon)
  C  Memory trace trajectory — criterion8_on EMA vs sham (experience-dependence)
  D  Ablation effect — alive_count: criterion8_on vs criterion8_ablated,
     vertical dashed line at ablation step 5 000

Data sources
------------
  experiments/criterion8_baseline.json
  experiments/criterion8_criterion8_on.json
  experiments/criterion8_criterion8_ablated.json
  experiments/criterion8_sham.json
  experiments/criterion8_analysis.json  (for annotation)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT, load_json

# ---------------------------------------------------------------------------
# Okabe-Ito colors for the 4 conditions
# ---------------------------------------------------------------------------

_COLORS = {
    "baseline": "#000000",          # black
    "criterion8_on": "#0072B2",     # blue
    "criterion8_ablated": "#D55E00", # vermillion
    "sham": "#CC79A7",              # reddish purple
}

_LABELS = {
    "baseline": "Baseline",
    "criterion8_on": "+Memory",
    "criterion8_ablated": "+Memory (ablated)",
    "sham": "Sham",
}

_COND_ORDER = ["baseline", "sham", "criterion8_ablated", "criterion8_on"]

_EXP_DIR = PROJECT_ROOT / "experiments"

_ABLATION_STEP = 5_000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_conditions(exp_dir: Path) -> dict[str, list[dict]]:
    """Load all four condition JSON files; return empty list for missing."""
    files = {
        "baseline": exp_dir / "criterion8_baseline.json",
        "criterion8_on": exp_dir / "criterion8_criterion8_on.json",
        "criterion8_ablated": exp_dir / "criterion8_criterion8_ablated.json",
        "sham": exp_dir / "criterion8_sham.json",
    }
    result = {}
    for cond, path in files.items():
        result[cond] = load_json(path) if path.exists() else []
    return result


def _survival_auc(result: dict) -> float:
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _trajectories(
    results: list[dict], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, median, q25, q75) across seeds for `field`."""
    from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _panel_violin(ax, conditions: dict[str, list[dict]], analysis: dict | None) -> None:
    """Panel A: Survival AUC violin + strip plot, 4 conditions."""
    data_by_cond = {
        cond: [_survival_auc(r) for r in conditions.get(cond, [])]
        for cond in _COND_ORDER
    }

    xs = list(range(len(_COND_ORDER)))
    for xi, cond in enumerate(_COND_ORDER):
        vals = data_by_cond[cond]
        if not vals:
            continue
        color = _COLORS[cond]
        # Violin
        parts = ax.violinplot([vals], positions=[xi], widths=0.6,
                              showmedians=True, showextrema=False)
        for part in parts["bodies"]:
            part.set_facecolor(color)
            part.set_alpha(0.35)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(2.0)
        # Strip (jittered)
        rng = np.random.default_rng(seed=42 + xi)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=12, alpha=0.6, zorder=3)

    # Significance annotations from analysis JSON
    if analysis:
        comp = analysis.get("pairwise_vs_baseline", {})
        c8_on_data = comp.get("criterion8_on", {})
        p_adj = c8_on_data.get("vs_baseline_mwu_p_adj")
        d = c8_on_data.get("vs_baseline_cohen_d")
        if p_adj is not None and not np.isnan(p_adj):
            sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
            y_max = max((_survival_auc(r) for results in conditions.values() for r in results), default=0)
            y_ann = y_max * 1.05
            baseline_xi = _COND_ORDER.index("baseline")
            c8_xi = _COND_ORDER.index("criterion8_on")
            ax.annotate(
                "",
                xy=(c8_xi, y_ann), xytext=(baseline_xi, y_ann),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.0),
            )
            ax.text(
                (c8_xi + baseline_xi) / 2, y_ann * 1.01,
                f"{sig} (d={d:.2f})" if d is not None else sig,
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([_LABELS[c] for c in _COND_ORDER], fontsize=8)
    ax.set_ylabel("Survival AUC (Σ alive count)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_alive_trajectory(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel B: alive_count trajectories — baseline vs +memory."""
    for cond in ["baseline", "criterion8_on"]:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "alive_count")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.15)
        ax.plot(steps, med, color=color, linewidth=1.8, label=_LABELS[cond])

    ax.set_ylabel("Alive count (median)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_memory_trace(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel C: Experience-dependence — EMA memory trace (criterion8_on) vs sham.

    This is the must-have figure for the paper: same genotype, identical
    perturbation regime, differing only in memory coherence.  The EMA should
    converge to a stable value; the sham should stay near 0.5 with high variance.
    """
    for cond in ["criterion8_on", "sham"]:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "memory_mean")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.15)
        ax.plot(steps, med, color=color, linewidth=1.8, label=_LABELS[cond])

    # Reference line at 0.5 (uniform random expectation for sham)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":",
               label="Sham expected (0.5)")

    ax.set_ylabel("Memory trace — mean IS[0] EMA")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_ablation(ax, conditions: dict[str, list[dict]]) -> None:
    """Panel D: Ablation effect — criterion8_on vs criterion8_ablated alive_count."""
    for cond in ["criterion8_on", "criterion8_ablated"]:
        results = conditions.get(cond, [])
        if not results:
            continue
        steps, med, q25, q75 = _trajectories(results, "alive_count")
        if len(steps) == 0:
            continue
        color = _COLORS[cond]
        ax.fill_between(steps, q25, q75, color=color, alpha=0.15)
        ax.plot(steps, med, color=color, linewidth=1.8, label=_LABELS[cond])

    ax.axvline(_ABLATION_STEP, color="#888888", linewidth=1.0, linestyle="--",
               label=f"Ablation step ({_ABLATION_STEP:,d})")

    ax.set_ylabel("Alive count (median)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_criterion8() -> None:
    """Figure: 8th Criterion (memory) — 4-panel results."""
    conditions = _load_conditions(_EXP_DIR)

    if not conditions.get("baseline"):
        print("  SKIP: criterion8_baseline.json not found")
        return

    # Load analysis annotations if available
    analysis_path = _EXP_DIR / "criterion8_analysis.json"
    analysis: dict | None = None
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)

    n_seeds = len(conditions["baseline"])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- Panel A: AUC violin ---
    _panel_violin(ax_a, conditions, analysis)
    ax_a.set_title("A  Survival AUC by condition", loc="left", fontsize=9)
    ax_a.text(
        0.98, 0.97, f"$n$={n_seeds} seeds",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=7,
    )

    # --- Panel B: alive_count trajectories ---
    _panel_alive_trajectory(ax_b, conditions)
    ax_b.set_title("B  Population dynamics", loc="left", fontsize=9)

    # --- Panel C: memory trace (experience-dependence) ---
    _panel_memory_trace(ax_c, conditions)
    ax_c.set_xlabel("Simulation step")
    ax_c.set_title("C  Experience-dependence (memory trace)", loc="left", fontsize=9)

    # --- Panel D: ablation ---
    _panel_ablation(ax_d, conditions)
    ax_d.set_xlabel("Simulation step")
    ax_d.set_title("D  Ablation at step 5 000", loc="left", fontsize=9)

    # Shared x labels for top row
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Simulation step" if ax is ax_b else "Condition")

    fig.suptitle("8th Criterion: Learning/Memory (10 000 steps, 4 conditions)", fontsize=10)
    fig.tight_layout()

    out_path = FIG_DIR / "fig_criterion8.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
