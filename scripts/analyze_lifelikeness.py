"""Life-likeness gap analysis — compute metrics from long-horizon Tier 1 data.

Reads
-----
    experiments/lifelikeness_t1_normal_graph.json
    experiments/lifelikeness_t1_shift_graph.json
    experiments/lifelikeness_t1_normal_graph_memory.json
    experiments/lifelikeness_t1_shift_graph_memory.json

Writes
------
    experiments/lifelikeness_analysis.json  — all 6 metrics + criterion-8 decision
                                              + pairwise memory comparisons
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress, mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"

_CONDITION_PATHS: dict[str, Path] = {
    "normal_graph": _EXP_DIR / "lifelikeness_t1_normal_graph.json",
    "shift_graph": _EXP_DIR / "lifelikeness_t1_shift_graph.json",
    "normal_graph_memory": _EXP_DIR / "lifelikeness_t1_normal_graph_memory.json",
    "shift_graph_memory": _EXP_DIR / "lifelikeness_t1_shift_graph_memory.json",
}

_ANALYSIS_OUT = _EXP_DIR / "lifelikeness_analysis.json"


# ---------------------------------------------------------------------------
# Metric 1: Extinction timing
# ---------------------------------------------------------------------------


def compute_extinction(results: list[dict]) -> dict:
    """Return extinction statistics across seeds.

    A seed is "extinct" at step t if alive_count drops below 2 at or before t.
    """
    if not results:
        return {
            "per_seed": [],
            "n_seeds": 0,
            "fraction_extinct_by_2_5k": None,
            "fraction_extinct_by_5k": None,
            "fraction_extinct_by_10k": None,
            "median_extinction_step": None,
            "n_extinct_by_10k": 0,
        }

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

    # Build a complete window grid (including zero-birth windows) so the
    # exponential fit is not biased toward longer half-lives by omitting gaps.
    max_step = max(
        (s["step"] for r in results for s in r.get("samples", [])),
        default=0,
    )
    if max_step == 0:
        return {
            "halflife_steps": None,
            "decay_constant_k": None,
            "window_starts": [],
            "mean_births_per_window": [],
        }

    all_windows = list(range(0, max_step + 1, window_size))
    mean_births = [float(np.mean([wb.get(w, 0) for wb in per_seed_windows])) for w in all_windows]

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
    except Exception as exc:
        print(f"WARNING: curve_fit for novelty halflife failed: {exc}")

    return {
        "halflife_steps": halflife,
        "decay_constant_k": k_fit,
        "window_starts": all_windows,
        "mean_births_per_window": mean_births,
    }


# ---------------------------------------------------------------------------
# Shared BFS helper (also imported by figures/fig_lifelikeness_gap.py)
# ---------------------------------------------------------------------------


def _build_founder_descendants(events: list[dict]) -> dict[int, set[int]]:
    """BFS expansion of all lineages whose founders appeared before step 1 000.

    Returns a mapping from each founder stable_id to the set of all its
    descendants (including itself).  Returns {} when no founders are found.

    NOTE: Founders are derived from lineage_events (reproduction events only).
    Organisms alive before step 1 000 that never reproduced are excluded from
    the denominator; the survival metric therefore reflects "fraction of
    reproducing early lineages that persisted", not all early organisms.
    """
    founders: set[int] = set()
    for e in events:
        if e["step"] < 1_000:
            founders.add(int(e["parent_stable_id"]))
            founders.add(int(e["child_stable_id"]))
    if not founders:
        return {}

    parent_to_children: dict[int, list[int]] = defaultdict(list)
    for e in events:
        parent_to_children[int(e["parent_stable_id"])].append(int(e["child_stable_id"]))

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
    return founder_descendants


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
        founder_descendants = _build_founder_descendants(events)

        if not founder_descendants:
            per_seed.append({str(t): None for t in checkpoints})
            continue

        survival: dict[str, float | None] = {}
        for t in checkpoints:
            active_parents = {
                int(e["parent_stable_id"]) for e in events if t - window < e["step"] <= t
            }
            n_surviving = sum(1 for desc in founder_descendants.values() if desc & active_parents)
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
        1 — birth rate approximately constant (late within ±10% of early)
        2 — birth rate declines to ≈ 0 by step 10k
        3 — birth rate intermittent (non-zero but declining >10%)
        4 — birth rate genuinely growing (late > 110% of early)
    """
    per_seed_class: list[int] = []
    for r in results:
        events: list[dict] = r.get("lineage_events", [])
        early = sum(1 for e in events if 0 <= e["step"] < 2_500)
        late = sum(1 for e in events if 7_500 <= e["step"] <= 10_000)

        if early == 0 and late == 0:
            cls = 2  # no activity at all → extinct pattern
        elif late == 0:
            cls = 2  # declined to zero
        elif late > early * 1.1:
            cls = 4  # genuinely growing (>10% above early)
        elif abs(late - early) <= 0.1 * max(early, 1):
            cls = 1  # approximately constant (±10%)
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


def compute_adaptation_lag(shift_results: list[dict]) -> dict:
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
        pre_samples = [s for s in shift_r["samples"] if pre_start <= s["step"] <= pre_end]
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
# Survival AUC helper (matches criterion8 analysis pattern)
# ---------------------------------------------------------------------------


def _survival_auc(result: dict) -> float:
    """Sum of alive_count across all sample steps."""
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


# ---------------------------------------------------------------------------
# Statistical helpers (reused from analyze_criterion8.py pattern)
# ---------------------------------------------------------------------------


def _cohen_d(a: list[float], b: list[float]) -> float | None:
    """Pooled-variance Cohen's d (a vs b; positive = a > b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_sd = float(np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)))
    if pooled_sd == 0.0:
        return None
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_sd


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction.  Returns adjusted p-values in the same order."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        k = rank + 1
        adj = min(running_min, p * k)
        running_min = adj
        adjusted[orig_idx] = adj
    return adjusted


def _mwu_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided).  Returns (U_statistic, p_value)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


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
    if median_diversity_slope_late is None:
        print(
            "WARNING: median_diversity_slope_late is None — "
            "Rule 2 cannot be evaluated (insufficient late-window data); skipping to Rule 3."
        )
    elif median_diversity_slope_late <= 0:
        return (
            "generative_capacity",
            f"Rule 2: median_diversity_slope_late={median_diversity_slope_late:.4f} <= 0",
        )
    if novelty_halflife is None:
        print(
            "WARNING: novelty_halflife is None — "
            "Rule 3 cannot be evaluated (curve_fit failed or no births); skipping to Rule 4."
        )
    elif novelty_halflife < 5_000:
        return (
            "niche_construction",
            f"Rule 3: novelty_halflife_steps={novelty_halflife:.1f} < 5000",
        )
    return "open_ended_already", "Rule 4: none of the above triggered"


# ---------------------------------------------------------------------------
# Pairwise memory comparison
# ---------------------------------------------------------------------------


def _compute_memory_comparisons(
    loaded: dict[str, list[dict]],
) -> dict:
    """Pairwise comparisons: normal vs normal+memory, shift vs shift+memory.

    Uses survival AUC as the primary metric, plus extinction fraction and
    diversity slope as secondary metrics.
    """
    pairs = [
        ("normal_graph", "normal_graph_memory"),
        ("shift_graph", "shift_graph_memory"),
    ]

    comparisons: dict[str, dict] = {}
    raw_pvalues: list[float] = []
    pair_labels: list[str] = []

    for base_cond, mem_cond in pairs:
        base_results = loaded.get(base_cond, [])
        mem_results = loaded.get(mem_cond, [])
        if not base_results or not mem_results:
            continue

        label = f"{mem_cond}_vs_{base_cond}"
        pair_labels.append(label)

        base_aucs = [_survival_auc(r) for r in base_results]
        mem_aucs = [_survival_auc(r) for r in mem_results]

        u_stat, p_val = _mwu_test(mem_aucs, base_aucs)
        d = _cohen_d(mem_aucs, base_aucs)
        raw_pvalues.append(p_val)

        # Extinction comparison
        base_ext = compute_extinction(base_results)
        mem_ext = compute_extinction(mem_results)

        # Diversity slope comparison
        base_div = compute_diversity_slope(base_results)
        mem_div = compute_diversity_slope(mem_results)

        comparisons[label] = {
            "base_condition": base_cond,
            "memory_condition": mem_cond,
            "n_seeds_base": len(base_results),
            "n_seeds_memory": len(mem_results),
            "survival_auc": {
                "base_median": float(np.median(base_aucs)),
                "memory_median": float(np.median(mem_aucs)),
                "mwu_u": u_stat,
                "mwu_p_raw": p_val,
                "cohen_d": d,
            },
            "extinction_fraction_10k": {
                "base": base_ext["fraction_extinct_by_10k"],
                "memory": mem_ext["fraction_extinct_by_10k"],
            },
            "diversity_slope_median": {
                "base": base_div["median_slope_per_1k_steps"],
                "memory": mem_div["median_slope_per_1k_steps"],
            },
        }

    # Holm-Bonferroni correction across all pairwise tests
    adjusted = _holm_bonferroni(raw_pvalues)
    for adj_p, label in zip(adjusted, pair_labels, strict=True):
        comparisons[label]["survival_auc"]["mwu_p_adj"] = adj_p
        comparisons[label]["survival_auc"]["significant_adj005"] = (
            adj_p < 0.05 if not (adj_p != adj_p) else None
        )

    return comparisons


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    condition_paths: dict[str, Path] | None = None,
    out_path: Path = _ANALYSIS_OUT,
) -> dict:
    """Compute all metrics, apply the decision rule, write JSON, return result dict."""
    if condition_paths is None:
        condition_paths = _CONDITION_PATHS

    normal_path = condition_paths["normal_graph"]

    if not normal_path.exists():
        print(f"ERROR: {normal_path} not found — run experiment_lifelikeness.py --tier 1 first.")
        sys.exit(1)

    # Load all conditions
    loaded: dict[str, list[dict]] = {}
    for cond, path in condition_paths.items():
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
            print(f"Loaded {cond}: {len(loaded[cond])} seeds")
        else:
            print(f"WARNING: {path} not found — condition '{cond}' will be skipped.")
            loaded[cond] = []

    normal_results = loaded["normal_graph"]
    shift_results = loaded["shift_graph"]

    if not normal_results:
        print(f"ERROR: {normal_path} contains no seed data.")
        sys.exit(1)

    print("Computing metrics...")

    ext = compute_extinction(normal_results)
    div = compute_diversity_slope(normal_results)
    nov = compute_novelty_halflife(normal_results)
    lin = compute_lineage_survival(normal_results)
    bed = compute_bedau_class(normal_results)
    lag = compute_adaptation_lag(shift_results) if shift_results else None

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

    # Memory pairwise comparisons (if memory conditions are available)
    memory_comparisons = _compute_memory_comparisons(loaded)
    if memory_comparisons:
        print("\nMemory pairwise comparisons:")
        for label, comp in memory_comparisons.items():
            auc = comp["survival_auc"]
            d_str = f"{auc['cohen_d']:.3f}" if auc["cohen_d"] is not None else "n/a"
            p_adj = auc.get("mwu_p_adj")
            p_str = f"{p_adj:.4f}" if p_adj is not None else "n/a"
            print(
                f"  {label:45s}  p_adj={p_str}  d={d_str}"
                f"  base_med={auc['base_median']:.1f}  mem_med={auc['memory_median']:.1f}"
            )

    # Per-condition summary stats (for all loaded conditions)
    per_condition_summary: dict[str, dict] = {}
    for cond, results in loaded.items():
        if not results:
            continue
        aucs = [_survival_auc(r) for r in results]
        cond_ext = compute_extinction(results)
        cond_div = compute_diversity_slope(results)
        per_condition_summary[cond] = {
            "n_seeds": len(results),
            "survival_auc_median": float(np.median(aucs)),
            "survival_auc_mean": float(np.mean(aucs)),
            "survival_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
            "extinction_fraction_10k": cond_ext["fraction_extinct_by_10k"],
            "diversity_slope_median": cond_div["median_slope_per_1k_steps"],
        }

    analysis = {
        "data_sources": {
            cond: str(path) for cond, path in condition_paths.items() if loaded.get(cond)
        },
        "n_seeds_normal": len(normal_results),
        "n_seeds_shift": len(shift_results),
        "per_condition_summary": per_condition_summary,
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
        "memory_comparisons": memory_comparisons,
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
