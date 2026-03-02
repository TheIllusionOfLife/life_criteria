"""8th Criterion stress-test analysis — statistical comparison under harsh perturbations.

Reads
-----
    experiments/stress_{regime}_{condition}.json  (for each regime × condition)

Writes
------
    experiments/stress_analysis.json   — per-regime statistics, tests, learning curves
    experiments/stress_manifest.json   — run metadata for reproducibility

Metrics (beyond standard survival AUC)
-------
  Post-shock survival rate : fraction alive at step 10k vs step 3000 (famine)
  Post-shock AUC           : sum(alive_count) for steps 3000-10000 only
  Recovery time            : first step after shock where alive_count stabilises
  Per-cycle survival       : alive_count at end of each bust phase (boom-bust)
  Learning curve slope     : regression of per-cycle survival on cycle number

Usage
-----
    uv run python scripts/analyze_criterion8_stress.py
    uv run python scripts/analyze_criterion8_stress.py --no-figure
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_OUT = _EXP_DIR / "stress_analysis.json"
_MANIFEST_OUT = _EXP_DIR / "stress_manifest.json"

_REGIMES = ["famine", "boom_bust"]
_CONDITIONS = ["baseline", "criterion8_on", "criterion8_ablated", "sham"]

# Famine-specific constants
_FAMINE_SHIFT_STEP = 3_000

# Boom-bust constants
_CYCLE_PERIOD = 2_500  # bust at odd half-cycles: [2.5k,5k), [7.5k,10k)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _condition_path(regime: str, condition: str) -> Path:
    return _EXP_DIR / f"stress_{regime}_{condition}.json"


def _load_regime(regime: str) -> dict[str, list[dict]]:
    """Load all condition files for a regime; return empty list for missing."""
    loaded: dict[str, list[dict]] = {}
    for cond in _CONDITIONS:
        path = _condition_path(regime, cond)
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
            print(f"  Loaded {regime}/{cond}: {len(loaded[cond])} seeds")
        else:
            loaded[cond] = []
            print(f"  SKIP: {path.name} not found")
    return loaded


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def _survival_auc(result: dict) -> float:
    """Total survival AUC (sum of alive_count over all samples)."""
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _post_shock_auc(result: dict, shock_step: int = _FAMINE_SHIFT_STEP) -> float:
    """Sum of alive_count for steps >= shock_step."""
    return float(
        sum(s["alive_count"] for s in result.get("samples", []) if s["step"] >= shock_step)
    )


def _post_shock_survival_rate(result: dict, shock_step: int = _FAMINE_SHIFT_STEP) -> float | None:
    """Fraction: alive_count at final step / alive_count at shock_step."""
    samples = result.get("samples", [])
    if not samples:
        return None
    at_shock = None
    at_end = None
    for s in samples:
        if s["step"] == shock_step:
            at_shock = s["alive_count"]
        at_end = s["alive_count"]  # last sample
    if at_shock is None or at_shock == 0:
        return None
    return float(at_end / at_shock)


def _recovery_time(
    result: dict,
    shock_step: int = _FAMINE_SHIFT_STEP,
    window: int = 5,
    threshold: float = 0.05,
) -> int | None:
    """First step after shock where alive_count stabilises.

    Stabilisation = coefficient of variation over `window` consecutive samples
    drops below `threshold`.  Returns None if never stabilises.
    """
    samples = [s for s in result.get("samples", []) if s["step"] >= shock_step]
    if len(samples) < window:
        return None
    counts = [s["alive_count"] for s in samples]
    for i in range(len(counts) - window + 1):
        chunk = counts[i : i + window]
        mean = np.mean(chunk)
        if mean == 0:
            # All dead — consider "stable" at zero
            return int(samples[i]["step"])
        cv = float(np.std(chunk, ddof=1) / mean)
        if cv < threshold:
            return int(samples[i]["step"])
    return None


def _extinction_rate(results: list[dict]) -> float:
    """Fraction of seeds where final alive_count == 0."""
    if not results:
        return float("nan")
    n_dead = sum(1 for r in results if r.get("final_alive_count", 0) == 0)
    return n_dead / len(results)


def _per_cycle_survival(result: dict, period: int = _CYCLE_PERIOD) -> list[float]:
    """Alive count at the end of each bust phase (boom-bust regime).

    Bust phases end at steps: period*2, period*4, ... (end of odd half-cycles).
    With period=2500: bust phases at [2.5k,5k) and [7.5k,10k), ending at 5k and 10k.
    """
    bust_end_steps = [period * (2 * i + 2) for i in range(10_000 // (period * 2))]
    sample_map = {s["step"]: s["alive_count"] for s in result.get("samples", [])}
    return [float(sample_map.get(step, 0)) for step in bust_end_steps]


def _learning_curve_slope(per_cycle: list[float]) -> float | None:
    """OLS slope of per-cycle survival vs cycle number (0-indexed).

    Positive slope = organisms improve across successive bust cycles.
    """
    n = len(per_cycle)
    if n < 3:
        return None
    x = np.arange(n, dtype=float)
    y = np.array(per_cycle, dtype=float)
    # Ignore seeds where all values are zero (100% extinction)
    if np.all(y == 0):
        return 0.0
    # OLS: slope = cov(x,y) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return None
    return float(ss_xy / ss_xx)


# ---------------------------------------------------------------------------
# Statistical helpers (reused from analyze_criterion8.py)
# ---------------------------------------------------------------------------


def _cohen_d(a: list[float], b: list[float]) -> float | None:
    """Pooled-variance Cohen's d (positive = a > b)."""
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
    """Holm-Bonferroni step-down correction.  NaN-safe."""
    n = len(p_values)
    if n == 0:
        return []
    adjusted = [float("nan")] * n
    finite = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    if not finite:
        return adjusted
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    previous_adj = 0.0
    for rank, (orig_idx, p) in enumerate(finite):
        multiplier = m - rank
        adj = min(1.0, max(previous_adj, p * multiplier))
        adjusted[orig_idx] = adj
        previous_adj = adj
    return adjusted


def _mwu_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Per-regime analysis
# ---------------------------------------------------------------------------


def _analyze_famine(loaded: dict[str, list[dict]]) -> dict:
    """Analyse the famine regime: post-shock AUC, survival rate, recovery."""
    baseline = loaded.get("baseline", [])
    if len(baseline) < 2:
        return {"error": "baseline needs >= 2 seeds"}

    summaries: dict[str, dict] = {}
    for cond in _CONDITIONS:
        results = loaded.get(cond, [])
        if not results:
            summaries[cond] = {"n_seeds": 0}
            continue
        aucs = [_survival_auc(r) for r in results]
        post_aucs = [_post_shock_auc(r) for r in results]
        survival_rates = [_post_shock_survival_rate(r) for r in results]
        recovery_times = [_recovery_time(r) for r in results]
        valid_rates = [v for v in survival_rates if v is not None]
        valid_recovery = [v for v in recovery_times if v is not None]

        summaries[cond] = {
            "n_seeds": len(results),
            "extinction_rate": _extinction_rate(results),
            "survival_auc": {
                "per_seed": aucs,
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
            },
            "post_shock_auc": {
                "per_seed": post_aucs,
                "mean": float(np.mean(post_aucs)),
                "std": float(np.std(post_aucs, ddof=1)) if len(post_aucs) >= 2 else None,
            },
            "post_shock_survival_rate": {
                "per_seed": survival_rates,
                "mean": float(np.mean(valid_rates)) if valid_rates else None,
            },
            "recovery_time": {
                "per_seed": recovery_times,
                "mean": float(np.mean(valid_recovery)) if valid_recovery else None,
                "n_stabilised": len(valid_recovery),
            },
        }

    # Pairwise comparisons vs baseline (on post-shock AUC — more sensitive)
    baseline_post_aucs = summaries["baseline"]["post_shock_auc"]["per_seed"]
    comparison_conds = [
        c for c in ["criterion8_on", "criterion8_ablated", "sham"] if summaries[c]["n_seeds"] >= 2
    ]
    raw_pvalues: list[float] = []
    comparisons: dict[str, dict] = {}

    for cond in comparison_conds:
        other = summaries[cond]["post_shock_auc"]["per_seed"]
        u, p = _mwu_test(other, baseline_post_aucs)
        d = _cohen_d(other, baseline_post_aucs)
        raw_pvalues.append(p)
        comparisons[cond] = {
            "metric": "post_shock_auc",
            "vs_baseline_mwu_u": u,
            "vs_baseline_mwu_p_raw": p,
            "vs_baseline_cohen_d": d,
        }

    adjusted = _holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, comparison_conds, strict=True):
        comparisons[cond]["vs_baseline_mwu_p_adj"] = adj_p
        comparisons[cond]["significant_adj005"] = adj_p < 0.05 if not np.isnan(adj_p) else None

    return {
        "regime": "famine",
        "shift_step": _FAMINE_SHIFT_STEP,
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
    }


def _analyze_boom_bust(loaded: dict[str, list[dict]]) -> dict:
    """Analyse the boom-bust regime: per-cycle survival + learning curves."""
    baseline = loaded.get("baseline", [])
    if len(baseline) < 2:
        return {"error": "baseline needs >= 2 seeds"}

    summaries: dict[str, dict] = {}
    for cond in _CONDITIONS:
        results = loaded.get(cond, [])
        if not results:
            summaries[cond] = {"n_seeds": 0}
            continue
        aucs = [_survival_auc(r) for r in results]
        per_cycle_all = [_per_cycle_survival(r) for r in results]
        slopes = [_learning_curve_slope(pc) for pc in per_cycle_all]
        valid_slopes = [s for s in slopes if s is not None]

        # Mean per-cycle across seeds (element-wise)
        n_cycles = 5
        mean_per_cycle = []
        for ci in range(n_cycles):
            vals = [pc[ci] for pc in per_cycle_all if ci < len(pc)]
            mean_per_cycle.append(float(np.mean(vals)) if vals else 0.0)

        summaries[cond] = {
            "n_seeds": len(results),
            "extinction_rate": _extinction_rate(results),
            "survival_auc": {
                "per_seed": aucs,
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
            },
            "per_cycle_survival": {
                "mean_per_cycle": mean_per_cycle,
                "per_seed": per_cycle_all,
            },
            "learning_curve_slope": {
                "per_seed": slopes,
                "mean": float(np.mean(valid_slopes)) if valid_slopes else None,
                "std": float(np.std(valid_slopes, ddof=1)) if len(valid_slopes) >= 2 else None,
            },
        }

    # Pairwise: survival AUC comparisons
    baseline_aucs = summaries["baseline"]["survival_auc"]["per_seed"]
    comparison_conds = [
        c for c in ["criterion8_on", "criterion8_ablated", "sham"] if summaries[c]["n_seeds"] >= 2
    ]
    raw_pvalues: list[float] = []
    comparisons: dict[str, dict] = {}

    for cond in comparison_conds:
        other = summaries[cond]["survival_auc"]["per_seed"]
        u, p = _mwu_test(other, baseline_aucs)
        d = _cohen_d(other, baseline_aucs)
        raw_pvalues.append(p)
        comparisons[cond] = {
            "metric": "survival_auc",
            "vs_baseline_mwu_u": u,
            "vs_baseline_mwu_p_raw": p,
            "vs_baseline_cohen_d": d,
        }

    adjusted = _holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, comparison_conds, strict=True):
        comparisons[cond]["vs_baseline_mwu_p_adj"] = adj_p
        comparisons[cond]["significant_adj005"] = adj_p < 0.05 if not np.isnan(adj_p) else None

    # Learning curve comparison: is slope for criterion8_on > baseline?
    learning_comparison: dict = {}
    baseline_slopes = summaries["baseline"]["learning_curve_slope"]["per_seed"]
    baseline_slopes_valid = [s for s in baseline_slopes if s is not None]
    for cond in ["criterion8_on", "sham"]:
        lc_data = summaries[cond].get("learning_curve_slope")
        if not lc_data:
            continue
        cond_slopes = lc_data["per_seed"]
        cond_slopes_valid = [s for s in cond_slopes if s is not None]
        if len(cond_slopes_valid) >= 2 and len(baseline_slopes_valid) >= 2:
            u, p = _mwu_test(cond_slopes_valid, baseline_slopes_valid)
            d = _cohen_d(cond_slopes_valid, baseline_slopes_valid)
        else:
            u, p, d = float("nan"), float("nan"), None
        learning_comparison[cond] = {
            "metric": "learning_curve_slope",
            "vs_baseline_mwu_u": u,
            "vs_baseline_mwu_p": p,
            "vs_baseline_cohen_d": d,
        }

    return {
        "regime": "boom_bust",
        "cycle_period": _CYCLE_PERIOD,
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
        "learning_curve_comparison": learning_comparison,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(analysis: dict) -> None:
    """Write a compact manifest summarising run metadata."""
    manifest: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "regimes": {},
    }
    for regime_key in _REGIMES:
        regime_data = analysis.get(regime_key)
        if not regime_data or "error" in regime_data:
            continue
        summaries = regime_data.get("summaries", {})
        seeds_by_cond: dict[str, list[int]] = {}
        for cond in _CONDITIONS:
            cond_data = summaries.get(cond, {})
            # Seeds not stored in summary, derive from n_seeds
            seeds_by_cond[cond] = {"n_seeds": cond_data.get("n_seeds", 0)}
        manifest["regimes"][regime_key] = {
            "conditions": seeds_by_cond,
            "extinction_rates": {
                cond: summaries.get(cond, {}).get("extinction_rate")
                for cond in _CONDITIONS
                if summaries.get(cond, {}).get("n_seeds", 0) > 0
            },
        }

    with open(_MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {_MANIFEST_OUT}")


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


def _print_famine_summary(famine: dict) -> None:
    summaries = famine.get("summaries", {})
    comparisons = famine.get("pairwise_vs_baseline", {})

    print("\n=== FAMINE REGIME ===")
    print(f"  Shift at step {famine.get('shift_step', '?')}")

    print("\n  Extinction rates:")
    for cond in _CONDITIONS:
        s = summaries.get(cond, {})
        if s.get("n_seeds", 0) > 0:
            print(f"    {cond:25s}  {s['extinction_rate']:.1%}  (n={s['n_seeds']})")

    print("\n  Post-shock AUC:")
    for cond in _CONDITIONS:
        s = summaries.get(cond, {})
        ps = s.get("post_shock_auc", {})
        if ps.get("mean") is not None:
            print(f"    {cond:25s}  mean={ps['mean']:10.1f}  std={ps.get('std', 0):8.1f}")

    print("\n  Pairwise vs baseline (post-shock AUC, Holm-Bonferroni):")
    for cond in ["criterion8_on", "criterion8_ablated", "sham"]:
        c = comparisons.get(cond, {})
        if not c:
            continue
        d_val = c.get("vs_baseline_cohen_d")
        d_str = f"{d_val:.3f}" if d_val is not None else "n/a"
        print(
            f"    {cond:25s}  p_raw={c['vs_baseline_mwu_p_raw']:.4f}"
            f"  p_adj={c['vs_baseline_mwu_p_adj']:.4f}"
            f"  d={d_str}"
        )


def _print_boom_bust_summary(bb: dict) -> None:
    summaries = bb.get("summaries", {})
    learning = bb.get("learning_curve_comparison", {})

    print("\n=== BOOM-BUST REGIME ===")
    print(f"  Cycle period: {bb.get('cycle_period', '?')}")

    print("\n  Extinction rates:")
    for cond in _CONDITIONS:
        s = summaries.get(cond, {})
        if s.get("n_seeds", 0) > 0:
            print(f"    {cond:25s}  {s['extinction_rate']:.1%}  (n={s['n_seeds']})")

    print("\n  Mean per-cycle survival (bust-end alive count):")
    for cond in _CONDITIONS:
        s = summaries.get(cond, {})
        pcs = s.get("per_cycle_survival", {}).get("mean_per_cycle", [])
        if pcs:
            vals = "  ".join(f"{v:6.1f}" for v in pcs)
            print(f"    {cond:25s}  [{vals}]")

    print("\n  Learning curve slope (per-cycle survival vs cycle #):")
    for cond in _CONDITIONS:
        s = summaries.get(cond, {})
        lc = s.get("learning_curve_slope", {})
        if lc.get("mean") is not None:
            print(f"    {cond:25s}  mean={lc['mean']:.4f}  std={lc.get('std', 0):.4f}")

    if learning:
        print("\n  Learning curve slope comparison vs baseline:")
        for cond in ["criterion8_on", "sham"]:
            c = learning.get(cond, {})
            if not c:
                continue
            d_val = c.get("vs_baseline_cohen_d")
            d_str = f"{d_val:.3f}" if d_val is not None else "n/a"
            print(f"    {cond:25s}  p={c['vs_baseline_mwu_p']:.4f}  d={d_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_analysis() -> dict:
    """Run full stress-test analysis for all available regimes."""
    analysis: dict = {}

    for regime in _REGIMES:
        print(f"\nLoading regime: {regime}")
        loaded = _load_regime(regime)

        # Skip regime if baseline is missing or too small
        if len(loaded.get("baseline", [])) < 2:
            print(f"  SKIP: {regime} baseline has < 2 seeds")
            analysis[regime] = {"error": "insufficient baseline data"}
            continue

        if regime == "famine":
            analysis[regime] = _analyze_famine(loaded)
            _print_famine_summary(analysis[regime])
        elif regime == "boom_bust":
            analysis[regime] = _analyze_boom_bust(loaded)
            _print_boom_bust_summary(analysis[regime])

    # Write outputs
    _EXP_DIR.mkdir(exist_ok=True)
    with open(_ANALYSIS_OUT, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {_ANALYSIS_OUT}")

    _write_manifest(analysis)

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse 8th-criterion stress-test experiment results."
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis()

    if not args.no_figure:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from figures.fig_criterion8_stress import generate_criterion8_stress

            generate_criterion8_stress()
        except ImportError as exc:
            print(f"WARNING: Could not import figure module ({exc}); skipping figure.")


if __name__ == "__main__":
    main()
