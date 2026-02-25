"""Cyclic environment experiment.

10,000-step runs with periodic resource modulation (period=2000).
Compares evolution-on (baseline) vs evolution-off under cyclic stress.

Usage:
    uv run python scripts/experiment_cyclic.py > experiments/cyclic_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/cyclic_{condition}.json.
"""

import json
from pathlib import Path

import life_criteria
from experiment_common import log, make_config, run_condition_suite
from experiment_manifest import write_manifest

STEPS = 10000
SAMPLE_EVERY = 100
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

CYCLE_PERIOD = 2000
NORMAL_RATE = 0.01
LOW_RATE = 0.005

CONDITIONS = {
    "cyclic_evo_on": {
        "metabolism_mode": "graph",
        "environment_cycle_period": CYCLE_PERIOD,
        "environment_cycle_low_rate": LOW_RATE,
    },
    "cyclic_evo_off": {
        "metabolism_mode": "graph",
        "enable_evolution": False,
        "environment_cycle_period": CYCLE_PERIOD,
        "environment_cycle_low_rate": LOW_RATE,
    },
}


def main():
    """Run cyclic environment experiment (2 conditions x 30 seeds)."""
    log(f"Digital Life v{life_criteria.version()}")
    log(
        f"Cyclic environment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log(f"Cycle period: {CYCLE_PERIOD}, normal rate: {NORMAL_RATE}, low rate: {LOW_RATE}")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    base_config = json.loads(make_config(SEEDS[0], CONDITIONS["cyclic_evo_on"]))
    write_manifest(
        out_dir / "cyclic_manifest.json",
        experiment_name="cyclic_environment",
        steps=STEPS,
        sample_every=SAMPLE_EVERY,
        seeds=SEEDS,
        base_config=base_config,
        condition_overrides=CONDITIONS,
        report_bindings=[
            {
                "result_id": "cyclic_recovery",
                "paper_ref": "fig:evolution",
                "source_files": [
                    "experiments/cyclic_data.tsv",
                    "experiments/cyclic_cyclic_evo_on.json",
                    "experiments/cyclic_cyclic_evo_off.json",
                ],
            }
        ],
    )

    run_condition_suite("cyclic_", CONDITIONS, STEPS, SEEDS, SAMPLE_EVERY, out_dir=out_dir)


if __name__ == "__main__":
    main()
