"""Life-likeness gap analysis — compute metrics from long-horizon Tier 1 data.

Reads
-----
    experiments/lifelikeness_t1_normal_graph.json
    experiments/lifelikeness_t1_shift_graph.json

Writes
------
    experiments/lifelikeness_analysis.json  — all 6 metrics + criterion-8 decision
    paper/figures/fig_lifelikeness_gap.pdf  — 4-panel diagnostic figure

Decision rules applied here are the SAME pre-registered rules recorded in
experiment_lifelikeness.py's module docstring before any experiments were run.
First-match-wins, applied in order:

    extinction_fraction_at_10k > 0.50  →  adaptive_robustness
    median_diversity_slope_late <= 0   →  generative_capacity
    novelty_halflife_steps < 5000      →  niche_construction
    (none triggered)                   →  open_ended_already

Usage
-----
    uv run python scripts/analyze_lifelikeness.py
    uv run python scripts/analyze_lifelikeness.py --no-figure
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"

_NORMAL_PATH = _EXP_DIR / "lifelikeness_t1_normal_graph.json"
_SHIFT_PATH = _EXP_DIR / "lifelikeness_t1_shift_graph.json"
_ANALYSIS_OUT = _EXP_DIR / "lifelikeness_analysis.json"


# ---------------------------------------------------------------------------
# Metric 1: Extinction timing
# ---------------------------------------------------------------------------


def compute_extinction(results: list[dict]) -> dict:
    """Return extinction statistics across seeds.

    A seed is "extinct" at step t if alive_count drops below 2 at or before t.
    """
    extinct_steps: list[int | None] = []
    for r in results:
        ext_step: int | None = None
        for s in r["samples"]:
            if s["alive_count"] < 2:
                ext_step = int(s["step"])
                break
        extinct_steps.append(ext_step)

    n = len(extinct_steps)
    frac_2_5k = sum(1 for s in extinct_steps if s is not None and s <= 2_500) / n
    frac_5k = sum(1 for s in extinct_steps if s is not None and s <= 5_000) / n
    frac_10k = sum(1 for s in extinct_steps if s is not None and s <= 10_000) / n

    non_none = [s for s in extinct_steps if s is not None]
    median_ext: float | None = float(np.median(non_none)) if non_none else None

    return {
        "per_seed": extinct_steps,
        "n_seeds": n,
        "fraction_extinct_by_2_5k": frac_2_5k,
        "fraction_extinct_by_5k": frac_5k,
        "fraction_extinct_by_10k": frac_10k,
        "median_extinction_step": median_ext,
        "n_extinct_by_10k": len(non_none),
    }


# ---------------------------------------------------------------------------
# Metric 2: Complexity growth slope
# ---------------------------------------------------------------------------


def compute_diversity_slope(results: list[dict]) -> dict:
    """Linear regression on genome_diversity over the late window [5k, 10k].

    Slope units: diversity change per 1 000 steps.
    """
    slopes: list[float] = []
    for r in results:
        xs: list[float] = []
        ys: list[float] = []
        for s in r["samples"]:
            if 5_000 <= s["step"] <= 10_000:
                xs.append(float(s["step"]))
                ys.append(float(s.get("genome_diversity", 0.0)))
        if len(xs) >= 2:
            reg = linregress(xs, ys)
            slopes.append(float(reg.slope) * 1_000)  # per 1 000 steps

    return {
        "per_seed": slopes,
        "n_seeds_with_data": len(slopes),
        "median_slope_per_1k_steps": float(np.median(slopes)) if slopes else None,
        "mean_slope_per_1k_steps": float(np.mean(slopes)) if slopes else None,
    }


# ---------------------------------------------------------------------------
# Metric 3: Novelty rate decay (birth-rate half-life)
# ---------------------------------------------------------------------------


def compute_novelty_halflife(results: list[dict]) -> dict:
    """Fit exponential decay to birth rate per 1 000-step window.

    Uses lineage_events (each event = one birth) for accurate window counts,
    since birth_count in StepMetrics is a per-single-step snapshot.
    """
    window_size = 1_000
    # Count births per window per seed
    per_seed_windows: list[dict[int, int]] = []
    for r in results:
        events = r.get("lineage_events", [])
        wb: dict[int, int] = defaultdict(int)
        for e in events:
            w = (e["step"] // window_size) * window_size
            wb[w] += 1
        per_seed_windows.append(dict(wb))

    all_windows = sorted({w for wb in per_seed_windows for w in wb})
    if not all_windows:
        return {
            "halflife_steps": None,
            "decay_constant_k": None,
            "window_starts": [],
            "mean_births_per_window": [],
        }

    mean_births = [
        float(np.mean([wb.get(w, 0) for wb in per_seed_windows])) for w in all_windows
    ]

    # Attempt exponential decay fit: y = A * exp(-k * t)
    t_arr = np.array(all_windows, dtype=float)
    y_arr = np.array(mean_births, dtype=float)

    halflife: float | None = None
    k_fit: float | None = None
    try:
        popt, _ = curve_fit(
            lambda t, A, k: A * np.exp(-k * t),
            t_arr,
            y_arr,
            p0=[float(y_arr[0]) if y_arr[0] > 0 else 1.0, 1e-5],
            maxfev=10_000,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
        )
        _, k_fit = float(popt[0]), float(popt[1])
        halflife = float(np.log(2) / k_fit) if k_fit > 0 else None
    except Exception:
        pass

    return {
        "halflife_steps": halflife,
        "decay_constant_k": k_fit,
        "window_starts": all_windows,
        "mean_births_per_window": mean_births,
    }


# ---------------------------------------------------------------------------
# Metric 4: Lineage survival curve
# ---------------------------------------------------------------------------


def compute_lineage_survival(results: list[dict]) -> dict:
    """Kaplan-Meier-style survival of founder lineages.

    Founders: all organisms (stable_ids) born before step 1 000.
    A founder lineage "survives" at checkpoint t if any descendant of that
    founder reproduced in the window (t - 1 000, t].
    """
    checkpoints = [2_500, 5_000, 10_000]
    window = 1_000

    per_seed: list[dict] = []
    for r in results:
        events: list[dict] = r.get("lineage_events", [])

        # Identify founders: organisms present as parent or child in early events
        founders: set[int] = set()
        for e in events:
            if e["step"] < 1_000:
                founders.add(int(e["parent_stable_id"]))
                founders.add(int(e["child_stable_id"]))

        if not founders:
            per_seed.append({str(t): None for t in checkpoints})
            continue

        # Build parent→children adjacency for descendant expansion (BFS)
        parent_to_children: dict[int, list[int]] = defaultdict(list)
        for e in events:
            parent_to_children[int(e["parent_stable_id"])].append(
                int(e["child_stable_id"])
            )

        # For each founder expand all descendants
        founder_descendants: dict[int, set[int]] = {}
        for f in founders:
            desc: set[int] = {f}
            queue = [f]
            while queue:
                node = queue.pop()
                for child in parent_to_children.get(node, []):
                    if child not in desc:
                        desc.add(child)
                        queue.append(child)
            founder_descendants[f] = desc

        # At each checkpoint, check if any descendant reproduced recently
        survival: dict[str, float | None] = {}
        for t in checkpoints:
            active_parents = {
                int(e["parent_stable_id"])
                for e in events
                if t - window < e["step"] <= t
            }
            n_surviving = sum(
                1 for f, desc in founder_descendants.items() if desc & active_parents
            )
            survival[str(t)] = n_surviving / len(founder_descendants)

        per_seed.append(survival)

    # Cross-seed mean
    aggregated: dict[str, float | None] = {}
    for t in checkpoints:
        vals = [s[str(t)] for s in per_seed if s.get(str(t)) is not None]
        aggregated[str(t)] = float(np.mean(vals)) if vals else None

    return {
        "per_seed": per_seed,
        "mean_survival_by_checkpoint": aggregated,
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# Metric 5: Bedau activity class
# ---------------------------------------------------------------------------


def compute_bedau_class(results: list[dict]) -> dict:
    """Assign a Bedau activity class per seed, then report the modal class.

    Uses lineage_events for accurate window-level birth counts.
    Classes:
        1 — birth rate approximately constant (late ≈ early)
        2 — birth rate declines to ≈ 0 by step 20k
        3 — birth rate intermittent (non-zero but declining)
        4 — birth rate sustained or growing
    """
    from collections import Counter

    per_seed_class: list[int] = []
    for r in results:
        events: list[dict] = r.get("lineage_events", [])
        early = sum(1 for e in events if 0 <= e["step"] < 2_500)
        late = sum(1 for e in events if 7_500 <= e["step"] <= 10_000)

        if early == 0 and late == 0:
            cls = 2  # no activity at all → extinct pattern
        elif late == 0:
            cls = 2  # declined to zero
        elif late >= early * 0.9:
            cls = 4  # sustained or growing (within 10% of early)
        elif abs(late - early) <= 0.1 * max(early, 1):
            cls = 1  # approximately constant
        else:
            cls = 3  # declining but non-zero → intermittent

        per_seed_class.append(cls)

    counts = Counter(per_seed_class)
    modal_class = counts.most_common(1)[0][0] if counts else None

    return {
        "per_seed": per_seed_class,
        "class_counts": dict(counts),
        "modal_class": modal_class,
    }


# ---------------------------------------------------------------------------
# Metric 6: Adaptation lag
# ---------------------------------------------------------------------------


def compute_adaptation_lag(
    normal_results: list[dict], shift_results: list[dict]
) -> dict:
    """Measure how long the population takes to recover after resource shift at step 5k.

    Pre-shift baseline: mean alive_count over steps 4k–4.75k in the shift condition
    (identical to normal before the shift).
    Recovery target: 80% of pre-shift baseline.
    Lag: first step after 5k where alive_count reaches the target.
    """
    shift_step = 5_000
    pre_start = 4_000
    pre_end = 4_750
    recovery_frac = 0.8

    lags: list[int | None] = []
    for shift_r in shift_results:
        pre_samples = [
            s for s in shift_r["samples"] if pre_start <= s["step"] <= pre_end
        ]
        if not pre_samples:
            lags.append(None)
            continue

        baseline = float(np.mean([s["alive_count"] for s in pre_samples]))
        target = recovery_frac * baseline

        post_samples = sorted(
            [s for s in shift_r["samples"] if s["step"] > shift_step],
            key=lambda s: s["step"],
        )
        lag: int | None = None
        for s in post_samples:
            if s["alive_count"] >= target:
                lag = int(s["step"]) - shift_step
                break

        lags.append(lag)

    non_none = [lag for lag in lags if lag is not None]
    return {
        "per_seed": lags,
        "median_lag_steps": float(np.median(non_none)) if non_none else None,
        "fraction_recovered": len(non_none) / len(lags) if lags else None,
        "shift_step": shift_step,
        "recovery_target_fraction": recovery_frac,
    }


# ---------------------------------------------------------------------------
# Decision rule (mirrors pre-registered rules in experiment_lifelikeness.py)
# ---------------------------------------------------------------------------


def apply_decision_rule(
    extinction_fraction_at_10k: float,
    median_diversity_slope_late: float | None,
    novelty_halflife: float | None,
) -> tuple[str, str]:
    """Return (criterion8_candidate, triggered_rule) per the pre-registered rules."""
    if extinction_fraction_at_10k > 0.50:
        return (
            "adaptive_robustness",
            f"Rule 1: extinction_fraction_at_10k={extinction_fraction_at_10k:.3f} > 0.50",
        )
    if median_diversity_slope_late is not None and median_diversity_slope_late <= 0:
        return (
            "generative_capacity",
            f"Rule 2: median_diversity_slope_late={median_diversity_slope_late:.4f} <= 0",
        )
    if novelty_halflife is not None and novelty_halflife < 5_000:
        return (
            "niche_construction",
            f"Rule 3: novelty_halflife_steps={novelty_halflife:.1f} < 5000",
        )
    return "open_ended_already", "Rule 4: none of the above triggered"


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    normal_path: Path = _NORMAL_PATH,
    shift_path: Path = _SHIFT_PATH,
    out_path: Path = _ANALYSIS_OUT,
) -> dict:
    """Compute all metrics, apply the decision rule, write JSON, return result dict."""
    if not normal_path.exists():
        print(f"ERROR: {normal_path} not found — run experiment_lifelikeness.py --tier 1 first.")
        sys.exit(1)

    with open(normal_path) as f:
        normal_results: list[dict] = json.load(f)

    shift_results: list[dict] = []
    if shift_path.exists():
        with open(shift_path) as f:
            shift_results = json.load(f)
    else:
        print(f"WARNING: {shift_path} not found — adaptation_lag metric will be empty.")

    print(f"Loaded normal_graph: {len(normal_results)} seeds")
    if shift_results:
        print(f"Loaded shift_graph:  {len(shift_results)} seeds")

    print("Computing metrics...")

    ext = compute_extinction(normal_results)
    div = compute_diversity_slope(normal_results)
    nov = compute_novelty_halflife(normal_results)
    lin = compute_lineage_survival(normal_results)
    bed = compute_bedau_class(normal_results)
    lag = compute_adaptation_lag(normal_results, shift_results) if shift_results else None

    candidate, rule = apply_decision_rule(
        extinction_fraction_at_10k=ext["fraction_extinct_by_10k"],
        median_diversity_slope_late=div["median_slope_per_1k_steps"],
        novelty_halflife=nov["halflife_steps"],
    )

    print(f"\nDecision: {candidate}")
    print(f"  Triggered: {rule}")
    print(f"  extinction_fraction_at_10k = {ext['fraction_extinct_by_10k']:.3f}")
    print(f"  median_diversity_slope     = {div['median_slope_per_1k_steps']}")
    print(f"  novelty_halflife_steps     = {nov['halflife_steps']}")
    print(f"  bedau_modal_class          = {bed['modal_class']}")
    if lag:
        print(f"  adaptation_lag_median      = {lag['median_lag_steps']}")

    analysis = {
        "data_sources": {
            "normal_graph": str(normal_path),
            "shift_graph": str(shift_path) if shift_results else None,
        },
        "n_seeds_normal": len(normal_results),
        "n_seeds_shift": len(shift_results) if shift_results else 0,
        "metrics": {
            "extinction": ext,
            "diversity_slope": div,
            "novelty_halflife": nov,
            "lineage_survival": lin,
            "bedau_class": bed,
            "adaptation_lag": lag,
        },
        "decision": {
            "criterion8_candidate": candidate,
            "triggered_rule": rule,
            "summary": {
                "extinction_fraction_at_10k": ext["fraction_extinct_by_10k"],
                "median_diversity_slope_per_1k": div["median_slope_per_1k_steps"],
                "novelty_halflife_steps": nov["halflife_steps"],
                "bedau_modal_class": bed["modal_class"],
                "adaptation_lag_median_steps": lag["median_lag_steps"] if lag else None,
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")

    return analysis


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute life-likeness metrics from Tier 1 experiment data."
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation (useful when paper/figures/ is not writable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis()

    if not args.no_figure:
        # Import here so the script stays usable even without the figures package
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from figures import generate_lifelikeness_gap

            generate_lifelikeness_gap()
        except ImportError as exc:
            print(f"WARNING: Could not import figures package ({exc}); skipping figure.")


if __name__ == "__main__":
    main()
