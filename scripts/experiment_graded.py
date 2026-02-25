"""Graded metabolic ablation experiment.

Sweeps metabolism_efficiency_multiplier over [1.0, 0.75, 0.5, 0.25, 0.0]
to produce a dose-response curve for metabolic efficiency.

Usage:
    uv run python scripts/experiment_graded.py > experiments/graded_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/graded_{level}.json.
"""

import life_criteria
from experiment_common import log, run_condition_suite

STEPS = 1000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]

CONDITIONS = {
    f"graded_{level:.2f}": {
        "metabolism_mode": "graph",
        "metabolism_efficiency_multiplier": level,
    }
    for level in LEVELS
}


def main():
    """Run graded metabolic ablation experiment (5 levels x 30 seeds)."""
    log(f"Digital Life v{life_criteria.version()}")
    log(
        f"Graded ablation: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log(f"Levels: {LEVELS}")
    log("")
    run_condition_suite("graded_", CONDITIONS, STEPS, SEEDS, SAMPLE_EVERY)


if __name__ == "__main__":
    main()
