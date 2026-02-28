"""8th Criterion (Learning/Memory) analysis — statistical comparison of 4 conditions.

Reads
-----
    experiments/criterion8_baseline.json
    experiments/criterion8_criterion8_on.json
    experiments/criterion8_criterion8_ablated.json
    experiments/criterion8_sham.json

Writes
------
    experiments/criterion8_analysis.json  — per-condition AUC statistics + tests

Statistical plan
----------------
  Primary outcome 1 — Survival AUC:
      Sum of alive_count over all sample steps.  Higher = better long-run survival.

  Primary outcome 2 — Memory trace stability (criterion8_on vs sham):
      Variance of memory_mean over the late window [5k, 10k].
      Real memory should show low variance (converged EMA); sham should be high.

  Tests (all pairwise comparisons vs baseline):
      Mann-Whitney U, two-sided.  Holm-Bonferroni correction over 3 comparisons.
      Effect size: Cohen's d (pooled SD).

  Orthogonality check:
      Within criterion8_on, regress survival AUC on mean genome_diversity and
      mean energy_mean to verify memory benefit is not explained by the 7-criteria
      signals alone.  Report partial correlation coefficient.

Usage
-----
    uv run python scripts/analyze_criterion8.py
    uv run python scripts/analyze_criterion8.py --no-figure
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_OUT = _EXP_DIR / "criterion8_analysis.json"

_CONDITION_FILES = {
    "baseline": _EXP_DIR / "criterion8_baseline.json",
    "criterion8_on": _EXP_DIR / "criterion8_criterion8_on.json",
    "criterion8_ablated": _EXP_DIR / "criterion8_criterion8_ablated.json",
    "sham": _EXP_DIR / "criterion8_sham.json",
}

# ---------------------------------------------------------------------------
# AUC and trajectory helpers
# ---------------------------------------------------------------------------


def _survival_auc(result: dict) -> float:
    """Sum of alive_count across all sample steps (trapezoidal ≈ rectangular at uniform spacing)."""
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _memory_late_variance(result: dict, start: int = 5_000, end: int = 10_000) -> float | None:
    """Variance of memory_mean in the late window for one seed."""
    vals = [
        s["memory_mean"]
        for s in result.get("samples", [])
        if start <= s["step"] <= end and "memory_mean" in s
    ]
    if len(vals) < 2:
        return None
    return float(np.var(vals, ddof=1))


def _field_mean(result: dict, field: str) -> float:
    """Mean of a step-metric field across all samples."""
    vals = [s[field] for s in result.get("samples", []) if field in s]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Statistical helpers
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
    """Holm-Bonferroni step-down correction.  Returns adjusted p-values in the same order."""
    n = len(p_values)
    if n == 0:
        return []
    # Sort ascending; multiply smallest p by n, next by n-1, etc.
    # Enforce monotonicity via running maximum (step-down).
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    previous_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = n - rank
        if np.isnan(p):
            adjusted[orig_idx] = float("nan")
        else:
            adj = min(1.0, max(previous_adj, p * multiplier))
            adjusted[orig_idx] = adj
            previous_adj = adj
    return adjusted


def _mwu_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided).  Returns (U_statistic, p_value)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _partial_correlation(x: list[float], y: list[float], z: list[float]) -> float | None:
    """Partial correlation of x with y controlling for z (via residuals)."""
    n = len(x)
    if n < 4 or len(y) != n or len(z) != n:
        return None
    xa, ya, za = np.array(x), np.array(y), np.array(z)

    def _residuals(arr: np.ndarray, predictor: np.ndarray) -> np.ndarray:
        if np.std(predictor) == 0:
            return arr - np.mean(arr)
        slope = np.cov(arr, predictor)[0, 1] / np.var(predictor)
        return arr - (np.mean(arr) + slope * (predictor - np.mean(predictor)))

    res_x = _residuals(xa, za)
    res_y = _residuals(ya, za)
    denom = np.std(res_x) * np.std(res_y)
    if denom == 0:
        return None
    return float(np.corrcoef(res_x, res_y)[0, 1])


# ---------------------------------------------------------------------------
# Per-condition summary
# ---------------------------------------------------------------------------


def _summarize_condition(results: list[dict]) -> dict:
    """Compute per-seed AUC + memory variance for a single condition."""
    aucs = [_survival_auc(r) for r in results]
    mem_vars = [_memory_late_variance(r) for r in results]
    mem_vars_valid = [v for v in mem_vars if v is not None]

    return {
        "n_seeds": len(results),
        "survival_auc": {
            "per_seed": aucs,
            "mean": float(np.mean(aucs)) if aucs else None,
            "median": float(np.median(aucs)) if aucs else None,
            "std": float(np.std(aucs, ddof=1)) if len(aucs) >= 2 else None,
        },
        "memory_late_variance": {
            "per_seed": mem_vars,
            "mean": float(np.mean(mem_vars_valid)) if mem_vars_valid else None,
            "n_valid": len(mem_vars_valid),
        },
        "mean_alive_final": float(
            np.mean([r["final_alive_count"] for r in results]) if results else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Orthogonality check
# ---------------------------------------------------------------------------


def _orthogonality_check(criterion8_results: list[dict]) -> dict:
    """Partial correlation of survival AUC with genome_diversity controlling for energy_mean.

    A near-zero partial correlation suggests memory benefit is not reducible to
    the genome_diversity or energy signals already captured by the 7 criteria.
    """
    aucs = [_survival_auc(r) for r in criterion8_results]
    diversities = [_field_mean(r, "genome_diversity") for r in criterion8_results]
    energies = [_field_mean(r, "energy_mean") for r in criterion8_results]

    pcorr_auc_div_controlling_energy = _partial_correlation(aucs, diversities, energies)
    pcorr_auc_energy_controlling_div = _partial_correlation(aucs, energies, diversities)

    return {
        "partial_corr_auc_vs_diversity_controlling_energy": pcorr_auc_div_controlling_energy,
        "partial_corr_auc_vs_energy_controlling_diversity": pcorr_auc_energy_controlling_div,
        "n_seeds": len(criterion8_results),
        "note": (
            "Values near zero indicate survival AUC gain from memory is not explained "
            "by the 7-criteria metabolic/genomic signals alone."
        ),
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    condition_files: dict[str, Path] = _CONDITION_FILES,
    out_path: Path = _ANALYSIS_OUT,
) -> dict:
    """Load all conditions, run stats, write JSON, return result dict."""
    # Load all conditions — abort if baseline is missing
    if not condition_files["baseline"].exists():
        print(
            f"ERROR: {condition_files['baseline']} not found — run experiment_criterion8.py first."
        )
        sys.exit(1)

    loaded: dict[str, list[dict]] = {}
    for cond, path in condition_files.items():
        if path.exists():
            with open(path) as f:
                loaded[cond] = json.load(f)
            print(f"Loaded {cond}: {len(loaded[cond])} seeds")
        else:
            print(f"WARNING: {path} not found — condition '{cond}' will be skipped.")
            loaded[cond] = []

    baseline_results = loaded["baseline"]
    if not baseline_results:
        print("ERROR: baseline data is empty.")
        sys.exit(1)

    print("\nComputing per-condition summaries...")
    summaries = {cond: _summarize_condition(results) for cond, results in loaded.items()}

    # Pairwise comparisons vs baseline — skip conditions with no data
    baseline_aucs: list[float] = summaries["baseline"]["survival_auc"]["per_seed"]
    comparisons: dict[str, dict] = {}
    _all_comparison_conds = ["criterion8_on", "criterion8_ablated", "sham"]
    comparison_conds = [
        c for c in _all_comparison_conds if len(summaries[c]["survival_auc"]["per_seed"]) >= 2
    ]
    raw_pvalues: list[float] = []

    for cond in comparison_conds:
        other_aucs = summaries[cond]["survival_auc"]["per_seed"]
        u_stat, p_val = _mwu_test(other_aucs, baseline_aucs)
        d = _cohen_d(other_aucs, baseline_aucs)
        raw_pvalues.append(p_val)
        comparisons[cond] = {
            "vs_baseline_mwu_u": u_stat,
            "vs_baseline_mwu_p_raw": p_val,
            "vs_baseline_cohen_d": d,
        }

    # Holm-Bonferroni correction
    adjusted = _holm_bonferroni(raw_pvalues)
    for adj_p, cond in zip(adjusted, comparison_conds, strict=True):
        comparisons[cond]["vs_baseline_mwu_p_adj"] = adj_p
        comparisons[cond]["vs_baseline_significant_adj005"] = (
            adj_p < 0.05 if not (adj_p != adj_p) else None  # NaN check
        )

    # Memory stability check: real EMA should have lower variance than sham
    c8_mem_var = summaries["criterion8_on"]["memory_late_variance"]["mean"]
    sham_mem_var = summaries["sham"]["memory_late_variance"]["mean"]
    c8_mem_vars = [
        v for v in summaries["criterion8_on"]["memory_late_variance"]["per_seed"] if v is not None
    ]
    sham_mem_vars = [
        v for v in summaries["sham"]["memory_late_variance"]["per_seed"] if v is not None
    ]
    if c8_mem_vars and sham_mem_vars:
        mem_u, mem_p = _mwu_test(c8_mem_vars, sham_mem_vars)
    else:
        mem_u, mem_p = None, None

    memory_stability = {
        "criterion8_on_mean_late_variance": c8_mem_var,
        "sham_mean_late_variance": sham_mem_var,
        "mwu_c8_vs_sham_u": mem_u,
        "mwu_c8_vs_sham_p": mem_p,
        "interpretation": (
            "criterion8_on should show significantly LOWER variance (converged EMA) "
            "than sham (random trace)."
        ),
    }

    # Orthogonality
    orthogonality = _orthogonality_check(loaded["criterion8_on"])

    # Human-readable summary
    print("\nSurvival AUC summary:")
    for cond in ["baseline", "criterion8_on", "criterion8_ablated", "sham"]:
        s = summaries[cond]["survival_auc"]
        n = summaries[cond]["n_seeds"]
        med = f"{s['median']:8.1f}" if s["median"] is not None else "     n/a"
        mean = f"{s['mean']:8.1f}" if s["mean"] is not None else "     n/a"
        print(f"  {cond:25s}  median={med}  mean={mean}  n={n}")

    print("\nPairwise vs baseline (Holm-Bonferroni corrected):")
    for cond in comparison_conds:
        c = comparisons[cond]
        d_val = c["vs_baseline_cohen_d"]
        d_str = "n/a" if d_val is None else f"{d_val:.3f}"
        print(
            f"  {cond:25s}  p_raw={c['vs_baseline_mwu_p_raw']:.4f}"
            f"  p_adj={c['vs_baseline_mwu_p_adj']:.4f}"
            f"  d={d_str}"
        )

    mem_p_str = f"{mem_p:.4f}" if mem_p is not None else "n/a"
    print(f"\nMemory stability (c8_on var vs sham var): p={mem_p_str}")
    orth_val = orthogonality["partial_corr_auc_vs_diversity_controlling_energy"]
    print(f"Orthogonality: partial_corr(AUC, diversity | energy) = {orth_val}")

    analysis = {
        "data_sources": {k: str(v) for k, v in condition_files.items()},
        "summaries": summaries,
        "pairwise_vs_baseline": comparisons,
        "memory_stability": memory_stability,
        "orthogonality": orthogonality,
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
        description="Compute 8th-criterion memory statistics from experiment data."
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
            from figures import generate_criterion8

            generate_criterion8()
        except ImportError as exc:
            print(f"WARNING: Could not import figures package ({exc}); skipping figure.")


if __name__ == "__main__":
    main()
