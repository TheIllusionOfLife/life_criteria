"""Figure: Life-likeness gap — 4-panel long-horizon diagnostic.

Panels
------
  A  alive_count trajectories (all seeds + median + IQR ribbon)
  B  genome_diversity trajectories + late-window regression slope annotation
  C  Lineage survival curve (founder-cohort Kaplan-Meier style)
  D  Adaptation lag — normal vs. shift alive_count, vertical line at shift step

Data sources
------------
  experiments/lifelikeness_t1_normal_graph.json
  experiments/lifelikeness_t1_shift_graph.json
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
# Data helpers
# ---------------------------------------------------------------------------


def _load_tier1(exp_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load normal and shift Tier 1 JSON files; return (normal, shift)."""
    normal_path = exp_dir / "lifelikeness_t1_normal_graph.json"
    shift_path = exp_dir / "lifelikeness_t1_shift_graph.json"

    normal: list[dict] = load_json(normal_path) if normal_path.exists() else []
    shift: list[dict] = load_json(shift_path) if shift_path.exists() else []
    return normal, shift


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


def _per_seed_trajectories(
    results: list[dict], field: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (steps_common, list of per-seed value arrays)."""
    step_sets = [
        {int(s["step"]) for s in r.get("samples", [])} for r in results
    ]
    common_steps = sorted(set.intersection(*step_sets) if step_sets else set())
    steps = np.array(common_steps)
    per_seed = []
    for r in results:
        sv = {int(s["step"]): float(s.get(field, 0.0)) for s in r.get("samples", [])}
        per_seed.append(np.array([sv.get(st, np.nan) for st in common_steps]))
    return steps, per_seed


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _panel_trajectory(
    ax,
    results: list[dict],
    field: str,
    ylabel: str,
    color_main: str = "#000000",
    color_seed: str = "#888888",
) -> None:
    """Plot individual-seed thin lines + median + IQR ribbon."""
    if not results:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_ylabel(ylabel)
        return

    steps, per_seed = _per_seed_trajectories(results, field)
    if len(steps) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_ylabel(ylabel)
        return

    mat = np.stack(per_seed, axis=0)  # (n_seeds, n_steps)
    median = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)

    # Thin seed lines
    for vals in per_seed:
        ax.plot(steps, vals, color=color_seed, linewidth=0.4, alpha=0.25, rasterized=True)

    # Median + ribbon
    ax.fill_between(steps, q25, q75, color=color_main, alpha=0.15)
    ax.plot(steps, median, color=color_main, linewidth=1.8, label="Median")

    ax.set_ylabel(ylabel)
    ax.set_xlim(0, steps[-1] if len(steps) else 1)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_slope(ax, results: list[dict], steps: np.ndarray) -> None:
    """Overlay late-window regression slope on Panel B (genome_diversity)."""
    late_steps_all: list[float] = []
    late_divs_all: list[float] = []
    for r in results:
        for s in r.get("samples", []):
            if 5_000 <= s["step"] <= 10_000:
                late_steps_all.append(float(s["step"]))
                late_divs_all.append(float(s.get("genome_diversity", 0.0)))

    if len(late_steps_all) < 4:
        return

    reg = linregress(late_steps_all, late_divs_all)
    x0, x1 = 5_000.0, 10_000.0
    y0 = reg.slope * x0 + reg.intercept
    y1 = reg.slope * x1 + reg.intercept

    ax.plot([x0, x1], [y0, y1], color="#D55E00", linewidth=1.8,
            linestyle="--", label=f"Slope {reg.slope * 1000:+.3f}/1k steps", zorder=5)
    ax.axvspan(5_000, 10_000, color="#F0E442", alpha=0.12, label="Late window")


def _panel_lineage_survival(ax, results: list[dict]) -> None:
    """Panel C: Kaplan-Meier style lineage survival curve."""
    checkpoints = list(range(0, 11_000, 1_000))
    window = 1_000

    # Compute per-seed survival at each checkpoint
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
            active = {
                int(e["parent_stable_id"])
                for e in events
                if t - window < e["step"] <= t
            }
            n_surv = sum(1 for desc in founder_descendants.values() if desc & active)
            seed_survival.append(n_surv / len(founder_descendants))
        per_seed_survival.append(seed_survival)

    if not per_seed_survival or all(v[0] is None for v in per_seed_survival):
        ax.text(0.5, 0.5, "No lineage data", transform=ax.transAxes, ha="center")
        ax.set_ylabel("Founder lineage survival")
        return

    # Cross-seed mean ± SEM
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

    ax.fill_between(xs, ys - errs, ys + errs, color="#0072B2", alpha=0.15)
    ax.plot(xs, ys, color="#0072B2", linewidth=1.8, drawstyle="steps-post", label="Mean ± SEM")
    ax.set_ylabel("Founder lineage survival")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_adaptation_lag(
    ax, normal_results: list[dict], shift_results: list[dict], lag_annotation: str = ""
) -> None:
    """Panel D: Normal vs. shift alive_count with vertical shift line."""
    shift_step = 5_000
    colors = {"normal": "#000000", "shift": "#E69F00"}

    for label, results, color, ls in [
        ("Normal", normal_results, colors["normal"], "-"),
        ("Shift at 5k", shift_results, colors["shift"], "--"),
    ]:
        if not results:
            continue
        steps, median, q25, q75 = _trajectories(results, "alive_count")
        ax.fill_between(steps, q25, q75, color=color, alpha=0.12)
        ax.plot(steps, median, color=color, linewidth=1.6, linestyle=ls, label=label)

    ax.axvline(shift_step, color="#888888", linewidth=1.0, linestyle=":",
               label=f"Resource shift (step {shift_step:,d})")

    if lag_annotation:
        ax.text(
            0.98, 0.97, lag_annotation,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
        )

    ax.set_ylabel("Alive count (median)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure generator
# ---------------------------------------------------------------------------


def generate_lifelikeness_gap() -> None:
    """Figure: Life-likeness gap — 4-panel diagnostic (Tier 1, 10k steps)."""
    exp_dir = PROJECT_ROOT / "experiments"
    normal_results, shift_results = _load_tier1(exp_dir)

    if not normal_results:
        print("  SKIP: lifelikeness_t1_normal_graph.json not found")
        return

    # Load analysis annotations if available
    analysis_path = exp_dir / "lifelikeness_analysis.json"
    lag_annotation = ""
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        lag_med = (
            analysis.get("metrics", {})
            .get("adaptation_lag", {}) or {}
        ).get("median_lag_steps")
        if lag_med is not None:
            lag_annotation = f"Median lag: {lag_med:.0f} steps"
        candidate = analysis.get("decision", {}).get("criterion8_candidate", "")
        if candidate:
            lag_annotation += f"\nCandidate: {candidate}"

    n_seeds = len(normal_results)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- Panel A: alive_count ---
    _panel_trajectory(ax_a, normal_results, "alive_count", "Alive count")
    ax_a.set_title("A  Population dynamics", loc="left", fontsize=9)
    ax_a.text(
        0.98, 0.97, f"$n$={n_seeds} seeds",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=7,
    )

    # --- Panel B: genome_diversity ---
    steps, per_seed = _per_seed_trajectories(normal_results, "genome_diversity")
    _panel_trajectory(ax_b, normal_results, "genome_diversity", "Genome diversity")
    if len(steps):
        _annotate_slope(ax_b, normal_results, steps)
        ax_b.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_b.set_title("B  Diversity trajectory", loc="left", fontsize=9)

    # --- Panel C: lineage survival ---
    _panel_lineage_survival(ax_c, normal_results)
    ax_c.set_xlabel("Simulation step")
    ax_c.set_title("C  Founder lineage survival", loc="left", fontsize=9)
    if ax_c.get_legend_handles_labels()[1]:
        ax_c.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # --- Panel D: adaptation lag ---
    _panel_adaptation_lag(ax_d, normal_results, shift_results, lag_annotation)
    ax_d.set_xlabel("Simulation step")
    ax_d.set_title("D  Adaptation lag (shift at step 5k)", loc="left", fontsize=9)
    ax_d.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # Shared x labels for top row
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Simulation step")

    fig.suptitle("Life-likeness gap: 7-criteria system (10 000 steps)", fontsize=10)
    fig.tight_layout()

    out_path = FIG_DIR / "fig_lifelikeness_gap.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
