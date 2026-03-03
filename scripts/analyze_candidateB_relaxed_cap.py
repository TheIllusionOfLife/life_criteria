"""Candidate B pop-cap relaxation analysis.

Reads
-----
    experiments/relaxed_cap_{regime}_{condition}.json

Writes
------
    experiments/candidateB_relaxed_cap_analysis.json

Key diagnostic: kin_fraction distributions under cap=400 vs cap=100.
If kin_fraction is substantially higher (>> 0.03/0.19 from cap=100),
the kin signal is now viable.  If Candidate B still shows null with
a viable signal, the result is robust.

Usage
-----
    uv run python scripts/analyze_candidateB_relaxed_cap.py
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
_ANALYSIS_OUT = _EXP_DIR / "candidateB_relaxed_cap_analysis.json"

_REGIMES = ["famine", "boom_bust", "seasonal"]
_CONDITIONS = ["baseline", "candidateB_on", "candidateB_ablated", "sham"]


def _survival_auc(result: dict) -> float:
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _extinction_rate(results: list[dict]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.get("final_alive_count", 0) == 0) / len(results)


def _kin_fraction_trajectory(result: dict) -> list[float]:
    return [float(s.get("kin_fraction_mean", 0.0)) for s in result.get("samples", [])]


def _analyze_regime(regime: str) -> dict | None:
    loaded: dict[str, list[dict]] = {}
    for cond in _CONDITIONS:
        path = _EXP_DIR / f"relaxed_cap_{regime}_{cond}.json"
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
            print(f"  Loaded relaxed_cap_{regime}/{cond}: {len(loaded[cond])} seeds")
        else:
            loaded[cond] = []

    baseline = loaded.get("baseline", [])
    if len(baseline) < 2:
        return None

    summaries: dict[str, dict] = {}
    for cond in _CONDITIONS:
        results = loaded.get(cond, [])
        if not results:
            summaries[cond] = {"n_seeds": 0}
            continue
        aucs = [_survival_auc(r) for r in results]
        kf_trajectories = [_kin_fraction_trajectory(r) for r in results]
        kf_means = [float(np.mean(t)) if t else 0.0 for t in kf_trajectories]
        kf_final = [t[-1] if t else 0.0 for t in kf_trajectories]

        summaries[cond] = {
            "n_seeds": len(results),
            "extinction_rate": _extinction_rate(results),
            "survival_auc": {
                "per_seed": aucs,
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
            },
            "kin_fraction_overall_mean": distribution_stats(np.array(kf_means)),
            "kin_fraction_final": distribution_stats(np.array(kf_final)),
        }

    # Paired comparisons
    baseline_aucs = summaries["baseline"]["survival_auc"]["per_seed"]
    comparison_conds = [
        c for c in ["candidateB_on", "candidateB_ablated", "sham"]
        if summaries.get(c, {}).get("n_seeds", 0) >= 2
    ]
    comparisons: dict[str, dict] = {}
    raw_pvalues: list[float] = []

    for cond in comparison_conds:
        other = summaries[cond]["survival_auc"]["per_seed"]
        paired = run_paired_comparison(np.array(other), np.array(baseline_aucs))
        raw_pvalues.append(paired["wilcoxon_p"])
        comparisons[cond] = {"metric": "survival_auc", **paired}

    adjusted = holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, comparison_conds, strict=True):
        comparisons[cond]["wilcoxon_p_adj"] = adj_p

    return {
        "regime": regime,
        "population_cap": 400,
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
    }


def run_analysis() -> dict:
    analysis: dict = {"regimes": {}}

    for regime in _REGIMES:
        print(f"\nAnalysing relaxed-cap: {regime}")
        result = _analyze_regime(regime)
        if result is None:
            print(f"  SKIP: insufficient data for {regime}")
            continue
        analysis["regimes"][regime] = result

        # Print summary
        print(f"\n  Kin-fraction (cap=400):")
        for cond in _CONDITIONS:
            s = result["summaries"].get(cond, {})
            kf = s.get("kin_fraction_overall_mean", {})
            if kf.get("mean") is not None:
                print(f"    {cond:25s}  mean={kf['mean']:.4f}  median={kf['median']:.4f}")

        print(f"\n  Survival AUC comparisons (paired Wilcoxon):")
        for cond in ["candidateB_on", "candidateB_ablated", "sham"]:
            c = result["pairwise_vs_baseline"].get(cond, {})
            if c:
                print(
                    f"    {cond:25s}  wilcoxon_p_adj={c.get('wilcoxon_p_adj', 'n/a'):.4f}"
                    f"  d_paired={c['paired_cohens_d']:.3f}"
                    f"  tost_p={c['tost_p']:.4f}"
                )

    _EXP_DIR.mkdir(exist_ok=True)
    with open(_ANALYSIS_OUT, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {_ANALYSIS_OUT}")
    return analysis


if __name__ == "__main__":
    run_analysis()
