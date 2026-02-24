"""Criterion-ablation statistical analysis package.

Entry point: call ``main()`` to run the full analysis CLI, or import
individual functions for use in notebooks and tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from .auc import extract_alive_at_step, extract_auc, extract_final_alive, extract_median_lifespan
from .statistics import (
    bootstrap_cliffs_delta_ci,
    cliffs_delta,
    cohens_d,
    cohens_d_ci,
    distribution_stats,
    holm_bonferroni,
    jonckheere_terpstra,
)

CONDITIONS = [
    "no_metabolism",
    "no_boundary",
    "no_homeostasis",
    "no_response",
    "no_reproduction",
    "no_evolution",
    "no_growth",
]


def load_condition(prefix: str, condition: str) -> list[dict]:
    """Load experiment results for a condition from JSON file."""
    path = Path(f"{prefix}_{condition}.json")
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_experiment_data(prefix: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load normal baseline and all condition data."""
    normal_results = load_condition(prefix, "normal")
    if not normal_results:
        print("ERROR: no normal baseline results found", file=sys.stderr)
        sys.exit(1)

    condition_data = {}
    for condition in CONDITIONS:
        condition_data[condition] = load_condition(prefix, condition)

    return normal_results, condition_data


def compute_condition_stats(
    condition: str,
    normal_alive: np.ndarray,
    normal_auc: np.ndarray,
    normal_median_lifespan: float,
    ablated_results: list[dict],
) -> dict | None:
    """Compute stats for a single ablation condition vs normal baseline."""
    ablated_alive = extract_final_alive(ablated_results)
    n_ablated = len(ablated_alive)

    if n_ablated < 2:
        print(f"  {condition}: SKIPPED (n={n_ablated} < 2)", file=sys.stderr)
        return None

    u_stat, p_value = stats.mannwhitneyu(normal_alive, ablated_alive, alternative="greater")
    d = cohens_d(normal_alive, ablated_alive)
    d_ci = cohens_d_ci(normal_alive, ablated_alive)
    cliff_d = cliffs_delta(normal_alive, ablated_alive)
    cliff_ci = bootstrap_cliffs_delta_ci(normal_alive, ablated_alive)
    ablated_auc = extract_auc(ablated_results)
    ablated_median_lifespan = extract_median_lifespan(ablated_results)

    print(
        f"  {condition}: U={u_stat:.1f}, p={p_value:.6f}, d={d:.3f}, "
        f"cliff={cliff_d:.3f}, "
        f"normal={np.mean(normal_alive):.1f}, ablated={np.mean(ablated_alive):.1f}",
        file=sys.stderr,
    )

    return {
        "condition": condition,
        "n_normal": len(normal_alive),
        "n_ablated": n_ablated,
        "normal_mean": float(np.mean(normal_alive)),
        "ablation_mean": float(np.mean(ablated_alive)),
        "normal_dist": distribution_stats(normal_alive),
        "ablation_dist": distribution_stats(ablated_alive),
        "U": float(u_stat),
        "p_raw": float(p_value),
        "cohens_d": round(d, 4),
        "cohens_d_ci_lo": round(d_ci[0], 4),
        "cohens_d_ci_hi": round(d_ci[1], 4),
        "cliffs_delta": round(cliff_d, 4),
        "cliffs_delta_ci_lo": round(cliff_ci[0], 4),
        "cliffs_delta_ci_hi": round(cliff_ci[1], 4),
        "normal_auc_mean": round(float(np.mean(normal_auc)), 2),
        "ablation_auc_mean": round(float(np.mean(ablated_auc)), 2),
        "normal_median_lifespan": round(normal_median_lifespan, 1),
        "ablation_median_lifespan": round(ablated_median_lifespan, 1),
    }


def perform_main_analysis(
    normal_results: list[dict], condition_data: dict[str, list[dict]], alpha: float = 0.05
) -> tuple[list[dict], int]:
    """Perform statistical analysis for all main ablation conditions."""
    normal_alive = extract_final_alive(normal_results)
    normal_auc = extract_auc(normal_results)
    normal_median_lifespan = extract_median_lifespan(normal_results)

    comparisons = []
    raw_p_values = []

    for condition in CONDITIONS:
        results = condition_data[condition]
        if not results:
            print(f"  {condition}: SKIPPED (no data)", file=sys.stderr)
            continue

        stats_res = compute_condition_stats(
            condition, normal_alive, normal_auc, normal_median_lifespan, results
        )
        if stats_res:
            comparisons.append(stats_res)
            raw_p_values.append(stats_res["p_raw"])

    # Apply Holm-Bonferroni correction
    corrected = holm_bonferroni(raw_p_values)
    significant_count = 0
    for comp, p_corr in zip(comparisons, corrected, strict=True):
        comp["p_corrected"] = round(p_corr, 6)
        comp["significant"] = bool(p_corr < alpha)
        if comp["significant"]:
            significant_count += 1

    return comparisons, significant_count


def perform_short_horizon_analysis(
    normal_results: list[dict],
    condition_data: dict[str, list[dict]],
    step: int = 500,
    alpha: float = 0.05,
) -> list[dict]:
    """Perform short-horizon analysis (T=500) for individual-level viability."""
    normal_alive_500 = extract_alive_at_step(normal_results, step)
    short_horizon = []
    short_raw_p = []

    for condition in CONDITIONS:
        results = condition_data[condition]
        if not results:
            continue
        ablated_alive_500 = extract_alive_at_step(results, step)
        if len(ablated_alive_500) < 2:
            continue

        u_stat_500, p_500 = stats.mannwhitneyu(
            normal_alive_500, ablated_alive_500, alternative="greater"
        )
        short_horizon.append(
            {
                "condition": condition,
                "normal_mean_500": round(float(np.mean(normal_alive_500)), 2),
                "ablation_mean_500": round(float(np.mean(ablated_alive_500)), 2),
                "U": float(u_stat_500),
                "p_raw": float(p_500),
            }
        )
        short_raw_p.append(p_500)

    short_corrected = holm_bonferroni(short_raw_p)
    for sh, p_corr in zip(short_horizon, short_corrected, strict=True):
        sh["p_corrected"] = round(p_corr, 6)
        sh["significant"] = bool(p_corr < alpha)

    print(f"\nShort-horizon (T={step}):", file=sys.stderr)
    for sh in short_horizon:
        status = "SIG" if sh["significant"] else "n.s."
        print(
            f"  [{status}] {sh['condition']}: mean={sh['ablation_mean_500']:.1f}, "
            f"p_corr={sh['p_corrected']:.6f}",
            file=sys.stderr,
        )

    return short_horizon


def analyze_graded(exp_dir: Path) -> dict | None:
    """Analyze graded ablation: dose-response + Jonckheere-Terpstra trend test."""
    levels = [1.0, 0.75, 0.5, 0.25, 0.0]
    groups = []
    level_stats = []

    for level in levels:
        path = exp_dir / f"graded_graded_{level:.2f}.json"
        if not path.exists():
            print(f"  SKIP graded {level}: {path} not found", file=sys.stderr)
            return None
        with open(path) as f:
            results = json.load(f)
        alive = extract_final_alive(results)
        groups.append(alive)
        level_stats.append(
            {
                "level": level,
                "n": len(alive),
                **distribution_stats(alive),
            }
        )

    jt_stat, jt_p = jonckheere_terpstra(groups)
    # Also pairwise: each level vs full (1.0)
    baseline = groups[0]
    pairwise = []
    for i, level in enumerate(levels[1:], 1):
        u_stat, p_val = stats.mannwhitneyu(baseline, groups[i], alternative="greater")
        d = cohens_d(baseline, groups[i])
        pairwise.append(
            {
                "level": level,
                "U": float(u_stat),
                "p_raw": float(p_val),
                "cohens_d": round(d, 4),
            }
        )

    print(f"Graded ablation: JT={jt_stat:.1f}, p={jt_p:.6f}", file=sys.stderr)
    return {
        "experiment": "graded_ablation",
        "levels": level_stats,
        "jonckheere_terpstra_stat": round(jt_stat, 2),
        "jonckheere_terpstra_p": round(jt_p, 6),
        "monotonic_trend": bool(jt_p < 0.05),
        "pairwise_vs_full": pairwise,
    }


def analyze_cyclic(exp_dir: Path) -> dict | None:
    """Analyze cyclic environment: per-cycle recovery comparison."""
    conditions = ["cyclic_evo_on", "cyclic_evo_off"]
    cond_data = {}
    for cond in conditions:
        path = exp_dir / f"cyclic_{cond}.json"
        if not path.exists():
            print(f"  SKIP cyclic {cond}: {path} not found", file=sys.stderr)
            return None
        with open(path) as f:
            cond_data[cond] = json.load(f)

    on_alive = extract_final_alive(cond_data["cyclic_evo_on"])
    off_alive = extract_final_alive(cond_data["cyclic_evo_off"])
    u_stat, p_val = stats.mannwhitneyu(on_alive, off_alive, alternative="greater")
    d = cohens_d(on_alive, off_alive)
    cliff_d = cliffs_delta(on_alive, off_alive)

    print(f"Cyclic: U={u_stat:.1f}, p={p_val:.6f}, d={d:.3f}", file=sys.stderr)
    return {
        "experiment": "cyclic_environment",
        "evo_on_dist": distribution_stats(on_alive),
        "evo_off_dist": distribution_stats(off_alive),
        "U": float(u_stat),
        "p_raw": round(float(p_val), 6),
        "cohens_d": round(d, 4),
        "cliffs_delta": round(cliff_d, 4),
        "significant": bool(p_val < 0.05),
    }


def analyze_sham(exp_dir: Path) -> dict | None:
    """Analyze sham ablation: expect non-significant difference."""
    conditions = ["sham_on", "sham_off"]
    cond_data = {}
    for cond in conditions:
        path = exp_dir / f"sham_{cond}.json"
        if not path.exists():
            print(f"  SKIP sham {cond}: {path} not found", file=sys.stderr)
            return None
        with open(path) as f:
            cond_data[cond] = json.load(f)

    on_alive = extract_final_alive(cond_data["sham_on"])
    off_alive = extract_final_alive(cond_data["sham_off"])
    u_stat, p_val = stats.mannwhitneyu(on_alive, off_alive, alternative="two-sided")
    d = cohens_d(on_alive, off_alive)

    print(f"Sham: U={u_stat:.1f}, p={p_val:.6f}, d={d:.3f}", file=sys.stderr)
    return {
        "experiment": "sham_ablation",
        "sham_on_dist": distribution_stats(on_alive),
        "sham_off_dist": distribution_stats(off_alive),
        "U": float(u_stat),
        "p_raw": round(float(p_val), 6),
        "cohens_d": round(d, 4),
        "non_significant": bool(p_val > 0.05),
    }


def generate_report(
    n_normal: int,
    alpha: float,
    significant_count: int,
    comparisons: list[dict],
    short_horizon_data: dict,
    extended_results: dict[str, dict],
) -> dict:
    """Construct the final analysis report dictionary."""
    output = {
        "experiment": "criterion_ablation",
        "n_per_condition": n_normal,
        "alpha": alpha,
        "correction": "holm_bonferroni",
        "significant_count": significant_count,
        "total_comparisons": len(comparisons),
        "comparisons": comparisons,
        "short_horizon": short_horizon_data,
    }
    output.update(extended_results)
    return output


def print_summary(comparisons: list[dict], significant_count: int) -> None:
    """Print a summary of significant findings to stderr."""
    print(f"\nSignificant: {significant_count}/{len(comparisons)}", file=sys.stderr)
    for comp in comparisons:
        status = "SIG" if comp["significant"] else "n.s."
        print(
            f"  [{status}] {comp['condition']}: p_corr={comp['p_corrected']:.6f}, "
            f"d={comp['cohens_d']:.3f}, cliff={comp['cliffs_delta']:.3f}",
            file=sys.stderr,
        )


def main() -> None:
    """Analyze criterion-ablation results with statistical tests and effect sizes."""
    parser = argparse.ArgumentParser(
        description="Analyze criterion-ablation results with statistical tests."
    )
    parser.add_argument("prefix", type=str, help="Experiment prefix (e.g. experiments/final)")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    args = parser.parse_args()

    prefix = args.prefix
    alpha = args.alpha

    normal_results, condition_data = load_experiment_data(prefix)

    normal_alive = extract_final_alive(normal_results)
    n_normal = len(normal_alive)
    print(f"Normal baseline: n={n_normal}, mean={np.mean(normal_alive):.1f}", file=sys.stderr)

    comparisons, significant_count = perform_main_analysis(normal_results, condition_data, alpha)

    short_horizon_step = 500
    short_horizon = perform_short_horizon_analysis(
        normal_results, condition_data, short_horizon_step, alpha
    )

    # ── Extended analyses: graded, cyclic, sham ──
    exp_dir = Path(prefix).resolve().parent
    extended_results = {}

    if graded := analyze_graded(exp_dir):
        extended_results["graded_ablation"] = graded
    if cyclic := analyze_cyclic(exp_dir):
        extended_results["cyclic_environment"] = cyclic
    if sham := analyze_sham(exp_dir):
        extended_results["sham_ablation"] = sham

    output = generate_report(
        n_normal=n_normal,
        alpha=alpha,
        significant_count=significant_count,
        comparisons=comparisons,
        short_horizon_data={
            "step": short_horizon_step,
            "comparisons": short_horizon,
        },
        extended_results=extended_results,
    )

    print(json.dumps(output, indent=2))
    print_summary(comparisons, significant_count)
