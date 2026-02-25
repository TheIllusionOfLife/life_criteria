"""Sham ablation control experiment.

Compares enable_sham_process=true vs false to validate that ablation
effects are functional, not computational artifacts.

Usage:
    uv run python scripts/experiment_sham.py > experiments/sham_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/sham_{condition}.json.
"""

import life_criteria
from experiment_common import log, run_condition_suite

STEPS = 1000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

CONDITIONS = {
    "sham_on": {
        "metabolism_mode": "graph",
        "enable_sham_process": True,
    },
    "sham_off": {
        "metabolism_mode": "graph",
        "enable_sham_process": False,
    },
}


def main():
    """Run sham ablation control experiment (2 conditions x 30 seeds)."""
    log(f"Digital Life v{life_criteria.version()}")
    log(
        f"Sham control: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite("sham_", CONDITIONS, STEPS, SEEDS, SAMPLE_EVERY)


if __name__ == "__main__":
    main()
