"""Figure: Life-likeness gap — 4-panel long-horizon diagnostic with memory comparison.

Panels
------
  A  alive_count trajectories: normal vs normal+memory (median + IQR ribbon)
  B  genome_diversity trajectories + late-window regression slope annotation
  C  Lineage survival curve (founder-cohort Kaplan-Meier style), normal vs memory
  D  Adaptation lag — normal vs shift vs shift+memory, vertical line at shift step

Data sources
------------
  experiments/lifelikeness_t1_normal_graph.json
  experiments/lifelikeness_t1_shift_graph.json
  experiments/lifelikeness_t1_normal_graph_memory.json
  experiments/lifelikeness_t1_shift_graph_memory.json
  experiments/lifelikeness_analysis.json  (for slope/lag annotations)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_lifelikeness import _build_founder_descendants
from figures._shared import FIG_DIR, PROJECT_ROOT, load_json
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Colors (Okabe-Ito colorblind-safe)
# ---------------------------------------------------------------------------

_COLORS = {
    "normal": "#000000",  # black
    "normal_memory": "#0072B2",  # blue
    "shift": "#E69F00",  # orange
    "shift_memory": "#009E73",  # bluish green
}

_LABELS = {
    "normal": "7-criteria",
    "normal_memory": "7-criteria + memory",
    "shift": "Shift (7-criteria)",
    "shift_memory": "Shift + memory",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_all(exp_dir: Path) -> dict[str, list[dict]]:
    """Load all 4 Tier 1 condition files."""
    names = {
        "normal": "lifelikeness_t1_normal_graph.json",
        "normal_memory": "lifelikeness_t1_normal_graph_memory.json",
        "shift": "lifelikeness_t1_shift_graph.json",
        "shift_memory": "lifelikeness_t1_shift_graph_memory.json",
    }
    loaded: dict[str, list[dict]] = {}
    for key, fname in names.items():
        path = exp_dir / fname
        loaded[key] = load_json(path) if path.exists() else []
    return loaded


def _trajectories(
    results: list[dict], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, median, q25, q75) arrays across seeds for `field`."""
    step_vals: dict[int, list[float]] = defaultdict(list)
    for r in results:
        for s in r.get("samples", []):
            step_vals[int(s["step"])].append(float(s.get(field, 0.0)))

    steps = np.array(sorted(step_vals.keys()))
    n = len(results)
    mat = np.full((n, len(steps)), np.nan)
    for ri, r in enumerate(results):
        sv = {int(s["step"]): float(s.get(field, 0.0)) for s in r.get("samples", [])}
        for si, st in enumerate(steps):
            if st in sv:
                mat[ri, si] = sv[st]

    median = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)
    return steps, median, q25, q75


def _per_seed_trajectories(results: list[dict], field: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (steps_common, list of per-seed value arrays)."""
    step_sets = [{int(s["step"]) for s in r.get("samples", [])} for r in results]
    non_empty = [ss for ss in step_sets if ss]
    common_steps = sorted(set.intersection(*non_empty) if non_empty else set())
    steps = np.array(common_steps)
    per_seed = []
    for r in results:
        sv = {int(s["step"]): float(s.get(field, 0.0)) for s in r.get("samples", [])}
        per_seed.append(np.array([sv.get(st, np.nan) for st in common_steps]))
    return steps, per_seed


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _plot_condition_ribbon(
    ax, results: list[dict], field: str, color: str, label: str, ls: str = "-"
) -> None:
    """Plot median + IQR ribbon for one condition."""
    if not results:
        return
    steps, median, q25, q75 = _trajectories(results, field)
    if len(steps) == 0:
        return
    ax.fill_between(steps, q25, q75, color=color, alpha=0.12)
    ax.plot(steps, median, color=color, linewidth=1.6, linestyle=ls, label=label)


def _annotate_slope(ax, results: list[dict], color: str, label_prefix: str) -> None:
    """Overlay median per-seed late-window regression slope."""
    slopes = []
    for r in results:
        xs = [float(s["step"]) for s in r.get("samples", []) if 5_000 <= s["step"] <= 10_000]
        ys = [
            float(s.get("genome_diversity", 0.0))
            for s in r.get("samples", [])
            if 5_000 <= s["step"] <= 10_000
        ]
        if len(xs) >= 2:
            slopes.append(linregress(xs, ys).slope * 1_000)

    if not slopes:
        return

    median_slope = float(np.median(slopes))

    x0, x1 = 5_000.0, 10_000.0
    div_at_x0 = [
        float(s.get("genome_diversity", 0.0))
        for r in results
        for s in r.get("samples", [])
        if int(s["step"]) == int(x0)
    ]
    if not div_at_x0:
        return
    y0 = float(np.median(div_at_x0))
    y1 = y0 + median_slope * (x1 - x0) / 1_000

    ax.plot(
        [x0, x1],
        [y0, y1],
        color=color,
        linewidth=1.8,
        linestyle="--",
        label=f"{label_prefix} slope {median_slope:+.3f}/1k",
        zorder=5,
    )


def _panel_lineage_survival_comparison(ax, data: dict[str, list[dict]], keys: list[str]) -> None:
    """Panel C: Kaplan-Meier style lineage survival, comparing conditions."""
    checkpoints = list(range(0, 11_000, 1_000))
    window = 1_000

    for key in keys:
        results = data.get(key, [])
        if not results:
            continue

        per_seed_survival: list[list[float | None]] = []
        for r in results:
            events: list[dict] = r.get("lineage_events", [])
            founder_descendants = _build_founder_descendants(events)

            if not founder_descendants:
                per_seed_survival.append([None] * len(checkpoints))
                continue

            seed_survival = []
            for t in checkpoints:
                if t == 0:
                    seed_survival.append(1.0)
                    continue
                active = {int(e["parent_stable_id"]) for e in events if t - window < e["step"] <= t}
                n_surv = sum(1 for desc in founder_descendants.values() if desc & active)
                seed_survival.append(n_surv / len(founder_descendants))
            per_seed_survival.append(seed_survival)

        if not per_seed_survival or all(v[0] is None for v in per_seed_survival):
            continue

        mean_surv = []
        sem_surv = []
        for ci in range(len(checkpoints)):
            vals = [v[ci] for v in per_seed_survival if v[ci] is not None]
            if vals:
                mean_surv.append(float(np.mean(vals)))
                sem_surv.append(
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
            else:
                mean_surv.append(np.nan)
                sem_surv.append(np.nan)

        xs = np.array(checkpoints, dtype=float)
        ys = np.array(mean_surv)
        errs = np.array(sem_surv)
        color = _COLORS[key]

        ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.12)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8,
            drawstyle="steps-post",
            label=f"{_LABELS[key]} (mean ± SEM)",
        )

    ax.set_ylabel("Founder lineage survival")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_lifelikeness_gap() -> None:
    """Figure: Life-likeness gap — 4-panel diagnostic (Tier 1, 10k steps)."""
    exp_dir = PROJECT_ROOT / "experiments"
    data = _load_all(exp_dir)

    if not data["normal"]:
        print("  SKIP: lifelikeness_t1_normal_graph.json not found")
        return

    # Load analysis annotations if available
    analysis_path = exp_dir / "lifelikeness_analysis.json"
    lag_annotation = ""
    memory_annotation = ""
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        lag_med = (analysis.get("metrics", {}).get("adaptation_lag", {}) or {}).get(
            "median_lag_steps"
        )
        if lag_med is not None:
            lag_annotation = f"Median lag: {lag_med:.0f} steps"
        candidate = analysis.get("decision", {}).get("criterion8_candidate", "")
        if candidate:
            lag_annotation += f"\nCandidate: {candidate}"

        # Memory comparison annotation
        mem_comp = analysis.get("memory_comparisons", {})
        normal_comp = mem_comp.get("normal_graph_memory_vs_normal_graph", {})
        if normal_comp:
            auc = normal_comp.get("survival_auc", {})
            p_adj = auc.get("mwu_p_adj")
            d = auc.get("cohen_d")
            if p_adj is not None and d is not None:
                sig = "*" if p_adj < 0.05 else "n.s."
                memory_annotation = f"Memory effect: d={d:.2f}, p_adj={p_adj:.3f} ({sig})"

    has_memory = bool(data["normal_memory"])
    n_seeds = len(data["normal"])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- Panel A: alive_count — normal vs normal+memory ---
    _plot_condition_ribbon(
        ax_a, data["normal"], "alive_count", _COLORS["normal"], _LABELS["normal"]
    )
    if has_memory:
        _plot_condition_ribbon(
            ax_a,
            data["normal_memory"],
            "alive_count",
            _COLORS["normal_memory"],
            _LABELS["normal_memory"],
        )
    ax_a.set_ylabel("Alive count")
    ax_a.set_ylim(bottom=0)
    ax_a.set_title("A  Population dynamics", loc="left", fontsize=9)
    seed_label = f"$n$={n_seeds} seeds"
    if memory_annotation:
        seed_label += f"\n{memory_annotation}"
    ax_a.text(
        0.98,
        0.97,
        seed_label,
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    if has_memory:
        ax_a.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # --- Panel B: genome_diversity — normal vs normal+memory ---
    _plot_condition_ribbon(
        ax_b, data["normal"], "genome_diversity", _COLORS["normal"], _LABELS["normal"]
    )
    if has_memory:
        _plot_condition_ribbon(
            ax_b,
            data["normal_memory"],
            "genome_diversity",
            _COLORS["normal_memory"],
            _LABELS["normal_memory"],
        )
    ax_b.axvspan(5_000, 10_000, color="#F0E442", alpha=0.12, label="Late window")
    _annotate_slope(ax_b, data["normal"], "#D55E00", "7-crit")
    if has_memory:
        _annotate_slope(ax_b, data["normal_memory"], "#009E73", "+mem")
    ax_b.set_ylabel("Genome diversity")
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("B  Diversity trajectory", loc="left", fontsize=9)
    ax_b.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # --- Panel C: lineage survival — normal vs normal+memory ---
    survival_keys = ["normal"]
    if has_memory:
        survival_keys.append("normal_memory")
    _panel_lineage_survival_comparison(ax_c, data, survival_keys)
    ax_c.set_xlabel("Simulation step")
    ax_c.set_title("C  Founder lineage survival", loc="left", fontsize=9)
    if ax_c.get_legend_handles_labels()[1]:
        ax_c.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # --- Panel D: adaptation lag — normal vs shift vs shift+memory ---
    shift_step = 5_000
    _plot_condition_ribbon(
        ax_d, data["normal"], "alive_count", _COLORS["normal"], _LABELS["normal"]
    )
    _plot_condition_ribbon(
        ax_d, data["shift"], "alive_count", _COLORS["shift"], _LABELS["shift"], ls="--"
    )
    if data["shift_memory"]:
        _plot_condition_ribbon(
            ax_d,
            data["shift_memory"],
            "alive_count",
            _COLORS["shift_memory"],
            _LABELS["shift_memory"],
            ls="-.",
        )

    ax_d.axvline(
        shift_step,
        color="#888888",
        linewidth=1.0,
        linestyle=":",
        label=f"Resource shift (step {shift_step:,d})",
    )

    if lag_annotation:
        ax_d.text(
            0.98,
            0.97,
            lag_annotation,
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
        )

    ax_d.set_ylabel("Alive count (median)")
    ax_d.set_ylim(bottom=0)
    ax_d.set_xlim(0)
    ax_d.set_xlabel("Simulation step")
    ax_d.set_title("D  Adaptation lag (shift at step 5k)", loc="left", fontsize=9)
    ax_d.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # Shared x labels for top row
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Simulation step")

    title = "Life-likeness gap: 7-criteria system (10 000 steps)"
    if has_memory:
        title = "Life-likeness gap: 7-criteria vs 7+memory (10 000 steps)"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    out_path = FIG_DIR / "fig_lifelikeness_gap.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
