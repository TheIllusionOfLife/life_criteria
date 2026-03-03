"""Seasonal cycle stress-test analysis — both Candidates A and B.

Reads
-----
    experiments/seasonal_{A,B}_{condition}.json

Writes
------
    experiments/seasonal_analysis.json

Metrics
-------
  Survival AUC             : sum(alive_count) over all samples
  Per-cycle survival       : alive_count at end of each "winter" (low-resource) phase
  Learning curve slope     : regression of per-cycle survival on cycle number
  Memory/Kin diagnostics   : memory_mean variance, kin_fraction_mean

Usage
-----
    uv run python scripts/analyze_seasonal_stress.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.results.statistics import (
    distribution_stats,
    holm_bonferroni,
    run_paired_comparison,
)

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_OUT = _EXP_DIR / "seasonal_analysis.json"

_CYCLE_PERIOD = 1_000  # seasonal cycle period
_CONDITIONS_A = ["baseline", "criterion8_on", "criterion8_ablated", "sham"]
_CONDITIONS_B = ["baseline", "candidateB_on", "candidateB_ablated", "sham"]


def _survival_auc(result: dict) -> float:
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _extinction_rate(results: list[dict]) -> float:
    if not results:
        return float("nan")
    n_dead = sum(1 for r in results if r.get("final_alive_count", 0) == 0)
    return n_dead / len(results)


def _per_cycle_survival(result: dict, period: int = _CYCLE_PERIOD) -> list[float]:
    """Alive count at the end of each low-resource (winter) phase.

    Winter ends at steps: period, 2*period, 3*period, ...
    With period=1000: winter phases end at 1000, 2000, ..., 10000.
    """
    cycle_end_steps = [period * (i + 1) for i in range(10_000 // period)]
    sample_map = {s["step"]: s["alive_count"] for s in result.get("samples", [])}
    return [float(sample_map.get(step, 0)) for step in cycle_end_steps]


def _learning_curve_slope(per_cycle: list[float]) -> float | None:
    n = len(per_cycle)
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    y = np.array(per_cycle, dtype=float)
    if np.all(y == 0):
        return 0.0
    x_mean = np.mean(x)
    ss_xy = np.sum((x - x_mean) * (y - np.mean(y)))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return None
    return float(ss_xy / ss_xx)


def _analyze_candidate(
    candidate: str,
    prefix: str,
    conditions: list[str],
    comparison_conds: list[str],
) -> dict | None:
    """Analyse one candidate's seasonal results."""
    loaded: dict[str, list[dict]] = {}
    for cond in conditions:
        path = _EXP_DIR / f"{prefix}{cond}.json"
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
            print(f"  Loaded {prefix}{cond}: {len(loaded[cond])} seeds")
        else:
            loaded[cond] = []

    baseline = loaded.get("baseline", [])
    if len(baseline) < 2:
        return None

    summaries: dict[str, dict] = {}
    for cond in conditions:
        results = loaded.get(cond, [])
        if not results:
            summaries[cond] = {"n_seeds": 0}
            continue
        aucs = [_survival_auc(r) for r in results]
        per_cycle_all = [_per_cycle_survival(r) for r in results]
        slopes = [_learning_curve_slope(pc) for pc in per_cycle_all]
        valid_slopes = [s for s in slopes if s is not None]

        n_cycles = max((len(pc) for pc in per_cycle_all), default=0)
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
            "per_cycle_survival": {"mean_per_cycle": mean_per_cycle},
            "learning_curve_slope": {
                "per_seed": slopes,
                "mean": float(np.mean(valid_slopes)) if valid_slopes else None,
                "std": float(np.std(valid_slopes, ddof=1)) if len(valid_slopes) >= 2 else None,
            },
        }

    # Paired comparisons vs baseline
    baseline_aucs = summaries["baseline"]["survival_auc"]["per_seed"]
    active_conds = [c for c in comparison_conds if summaries.get(c, {}).get("n_seeds", 0) >= 2]
    comparisons: dict[str, dict] = {}
    raw_pvalues: list[float] = []

    for cond in active_conds:
        other = summaries[cond]["survival_auc"]["per_seed"]
        paired = run_paired_comparison(np.array(other), np.array(baseline_aucs))
        raw_pvalues.append(paired["wilcoxon_p"])
        comparisons[cond] = {"metric": "survival_auc", **paired}

    adjusted = holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, active_conds, strict=True):
        comparisons[cond]["wilcoxon_p_adj"] = adj_p

    return {
        "candidate": candidate,
        "regime": "seasonal",
        "cycle_period": _CYCLE_PERIOD,
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
    }


def run_analysis() -> dict:
    analysis: dict = {}

    print("\n=== Candidate A (memory/EMA) — Seasonal ===")
    result_a = _analyze_candidate("A", "seasonal_A_", _CONDITIONS_A,
                                   ["criterion8_on", "criterion8_ablated", "sham"])
    if result_a:
        analysis["candidate_A"] = result_a
        for cond in ["criterion8_on", "criterion8_ablated", "sham"]:
            c = result_a["pairwise_vs_baseline"].get(cond, {})
            if c:
                print(
                    f"  {cond:25s}  wilcoxon_p_adj={c.get('wilcoxon_p_adj', 'n/a'):.4f}"
                    f"  d_paired={c['paired_cohens_d']:.3f}"
                    f"  tost_p={c['tost_p']:.4f}"
                )
    else:
        print("  SKIP: insufficient data for Candidate A seasonal")

    print("\n=== Candidate B (kin-sensing) — Seasonal ===")
    result_b = _analyze_candidate("B", "seasonal_B_", _CONDITIONS_B,
                                   ["candidateB_on", "candidateB_ablated", "sham"])
    if result_b:
        analysis["candidate_B"] = result_b
        for cond in ["candidateB_on", "candidateB_ablated", "sham"]:
            c = result_b["pairwise_vs_baseline"].get(cond, {})
            if c:
                print(
                    f"  {cond:25s}  wilcoxon_p_adj={c.get('wilcoxon_p_adj', 'n/a'):.4f}"
                    f"  d_paired={c['paired_cohens_d']:.3f}"
                    f"  tost_p={c['tost_p']:.4f}"
                )
    else:
        print("  SKIP: insufficient data for Candidate B seasonal")

    _EXP_DIR.mkdir(exist_ok=True)
    with open(_ANALYSIS_OUT, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {_ANALYSIS_OUT}")
    return analysis


if __name__ == "__main__":
    run_analysis()
