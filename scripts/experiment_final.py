"""Final criterion-ablation experiment (2000 steps, n=30, test set).

Runs 8 conditions (normal baseline + 7 criterion ablations) with
seeds 100-129 (test set) and 2000 steps for stronger evolution signal.

Usage:
    uv run python scripts/experiment_final.py > experiments/final_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/final_{condition}.json.
"""

import digital_life
from experiment_common import CONDITIONS, log, run_condition_suite

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}


def main():
    """Run final criterion-ablation experiment (8 conditions x 30 seeds)."""
    log(f"Digital Life v{digital_life.version()}")
    log(
        f"Final experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")
    run_condition_suite(
        "final_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
