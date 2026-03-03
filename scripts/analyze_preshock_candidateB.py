"""Pre-shock window analysis for Candidate B (kin-sensing).

Extracts survival AUC for steps 0–3000 only from existing Candidate B stress
data.  In this early window, organisms are still multi-agent (population cap
not yet binding), so kin_fraction should be a viable signal.  If Candidate B
still shows null in the pre-shock window, the result is robust to the pop-cap
confound identified by reviewers.

Reads
-----
    experiments/candidateB_{regime}_{condition}.json

Writes
------
    experiments/candidateB_preshock_analysis.json

Usage
-----
    uv run python scripts/analyze_preshock_candidateB.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.results.statistics import (
    holm_bonferroni,
    run_paired_comparison,
)

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_OUT = _EXP_DIR / "candidateB_preshock_analysis.json"

_REGIMES = ["famine", "boom_bust"]
_CONDITIONS = ["baseline", "candidateB_on", "candidateB_ablated", "sham"]
_PRESHOCK_END = 3_000


def _condition_path(regime: str, condition: str) -> Path:
    return _EXP_DIR / f"candidateB_{regime}_{condition}.json"


def _preshock_auc(result: dict, end_step: int = _PRESHOCK_END) -> float:
    """Sum of alive_count for steps <= end_step."""
    return float(
        sum(s["alive_count"] for s in result.get("samples", []) if s["step"] <= end_step)
    )


def _preshock_kin_fraction_mean(result: dict, end_step: int = _PRESHOCK_END) -> float:
    """Mean kin_fraction_mean in the pre-shock window."""
    vals = [
        s.get("kin_fraction_mean", 0.0)
        for s in result.get("samples", [])
        if s["step"] <= end_step
    ]
    return float(np.mean(vals)) if vals else 0.0


def _analyze_regime(regime: str) -> dict | None:
    """Analyse one regime's pre-shock window."""
    loaded: dict[str, list[dict]] = {}
    for cond in _CONDITIONS:
        path = _condition_path(regime, cond)
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
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
        aucs = [_preshock_auc(r) for r in results]
        kf_means = [_preshock_kin_fraction_mean(r) for r in results]
        summaries[cond] = {
            "n_seeds": len(results),
            "preshock_auc": {
                "per_seed": aucs,
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
            },
            "preshock_kin_fraction_mean": {
                "per_seed": kf_means,
                "mean": float(np.mean(kf_means)),
            },
        }

    # Paired comparisons vs baseline on pre-shock AUC
    baseline_aucs = summaries["baseline"]["preshock_auc"]["per_seed"]
    comparison_conds = [
        c for c in ["candidateB_on", "candidateB_ablated", "sham"] if summaries[c]["n_seeds"] >= 2
    ]
    comparisons: dict[str, dict] = {}
    raw_pvalues: list[float] = []

    for cond in comparison_conds:
        other = summaries[cond]["preshock_auc"]["per_seed"]
        paired = run_paired_comparison(np.array(other), np.array(baseline_aucs))
        raw_pvalues.append(paired["wilcoxon_p"])
        comparisons[cond] = {"metric": "preshock_auc", **paired}

    adjusted = holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, comparison_conds, strict=True):
        comparisons[cond]["wilcoxon_p_adj"] = adj_p

    return {
        "regime": regime,
        "preshock_end_step": _PRESHOCK_END,
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
    }


def run_analysis() -> dict:
    analysis: dict = {}
    for regime in _REGIMES:
        print(f"\nAnalysing pre-shock window: {regime}")
        result = _analyze_regime(regime)
        if result is None:
            print(f"  SKIP: insufficient baseline data for {regime}")
            analysis[regime] = {"error": "insufficient baseline data"}
            continue
        analysis[regime] = result

        # Print summary
        for cond in ["candidateB_on", "candidateB_ablated", "sham"]:
            c = result["pairwise_vs_baseline"].get(cond, {})
            if not c:
                continue
            print(
                f"  {cond:25s}  wilcoxon_p_adj={c.get('wilcoxon_p_adj', 'n/a'):.4f}"
                f"  d_paired={c['paired_cohens_d']:.3f}"
                f"  tost_p={c['tost_p']:.4f}"
            )
        # Kin-fraction diagnostic
        for cond in _CONDITIONS:
            s = result["summaries"].get(cond, {})
            kf = s.get("preshock_kin_fraction_mean", {})
            if kf.get("mean") is not None:
                print(f"  {cond:25s}  pre-shock kin_frac={kf['mean']:.4f}")

    _EXP_DIR.mkdir(exist_ok=True)
    with open(_ANALYSIS_OUT, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {_ANALYSIS_OUT}")
    return analysis


if __name__ == "__main__":
    run_analysis()
