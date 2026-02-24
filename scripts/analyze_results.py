"""Statistical analysis for criterion-ablation experiments.

Computes Mann-Whitney U tests, Cohen's d / Cliff's delta effect sizes,
AUC (area under alive-count curve), median lifespan, and
Holm-Bonferroni corrected p-values for each ablation condition
vs the normal baseline.

Usage:
    uv run python scripts/analyze_results.py experiments/final > experiments/final_statistics.json

The prefix argument (e.g. experiments/final) is used to find JSON files
named {prefix}_{condition}.json for each condition.

Implementation lives in ``analyses/results/``; this file is a thin dispatcher.
Public functions are re-exported here so that ``analyze_orthogonal.py`` and
``analyze_evolution_evidence.py`` (which import from this module) continue to work.
"""

from __future__ import annotations

from analyses.results import main
from analyses.results.auc import (
    extract_alive_at_step,
    extract_auc,
    extract_final_alive,
    extract_median_lifespan,
)
from analyses.results.statistics import (
    bootstrap_cliffs_delta_ci,
    cliffs_delta,
    cohens_d,
    cohens_d_ci,
    distribution_stats,
    holm_bonferroni,
    jonckheere_terpstra,
)

__all__ = [
    "main",
    "bootstrap_cliffs_delta_ci",
    "cliffs_delta",
    "cohens_d",
    "cohens_d_ci",
    "distribution_stats",
    "holm_bonferroni",
    "jonckheere_terpstra",
    "extract_auc",
    "extract_alive_at_step",
    "extract_final_alive",
    "extract_median_lifespan",
]

if __name__ == "__main__":
    main()
