"""Extended evolution experiment.

Two sub-experiments to demonstrate evolution's contribution:
  1. Long run: 10,000 steps (vs current 2,000), normal vs no_evolution
  2. Environmental shift: 5,000 steps, resource rate halved at step 2,500,
     normal vs no_evolution on post-shift recovery

Usage:
    uv run python scripts/experiment_evolution.py > experiments/evolution_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/evolution_{condition}.json.
"""

import json
import time
from pathlib import Path

import life_criteria
from experiment_common import log, make_config, print_header, run_condition_common
from experiment_manifest import write_manifest

SAMPLE_EVERY = 100
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

# Sub-experiment 1: Long run (10,000 steps)
LONG_STEPS = 10000
LONG_CONDITIONS = {
    "long_normal": {},
    "long_no_evolution": {"enable_evolution": False},
}

# Sub-experiment 2: Environmental shift (5,000 steps, shift at 2,500)
SHIFT_STEPS = 5000
SHIFT_STEP = 2500
SHIFT_RESOURCE_RATE = 0.005  # halved from 0.01
SHIFT_CONDITIONS = {
    "shift_normal": {
        "environment_shift_step": SHIFT_STEP,
        "environment_shift_resource_rate": SHIFT_RESOURCE_RATE,
    },
    "shift_no_evolution": {
        "enable_evolution": False,
        "environment_shift_step": SHIFT_STEP,
        "environment_shift_resource_rate": SHIFT_RESOURCE_RATE,
    },
}


def main():
    """Run extended evolution experiments: long run and environmental shift."""
    log(f"Digital Life v{life_criteria.version()}")
    log("Evolution strengthening experiment")
    log(f"  Long run: {LONG_STEPS} steps, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    log(f"  Shift run: {SHIFT_STEPS} steps, shift at {SHIFT_STEP}")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    write_manifest(
        out_dir / "evolution_long_manifest.json",
        experiment_name="evolution_long_run",
        steps=LONG_STEPS,
        sample_every=SAMPLE_EVERY,
        seeds=SEEDS,
        base_config=json.loads(make_config(SEEDS[0], {})),
        condition_overrides=LONG_CONDITIONS,
        report_bindings=[
            {
                "result_id": "evolution_long_run",
                "paper_ref": "fig:evolution",
                "source_files": [
                    "experiments/evolution_long_normal.json",
                    "experiments/evolution_long_no_evolution.json",
                    "experiments/evolution_evidence.json",
                ],
            },
            {
                "result_id": "phenotype_persistence",
                "paper_ref": "fig:persistent_clusters",
                "source_files": [
                    "experiments/niche_normal.json",
                    "experiments/niche_normal_long.json",
                    "experiments/phenotype_analysis.json",
                    "docs/research/zenodo_niche_long_horizon_metadata.json",
                ],
            },
        ],
    )
    write_manifest(
        out_dir / "evolution_shift_manifest.json",
        experiment_name="evolution_shift_run",
        steps=SHIFT_STEPS,
        sample_every=SAMPLE_EVERY,
        seeds=SEEDS,
        base_config=json.loads(make_config(SEEDS[0], SHIFT_CONDITIONS["shift_normal"])),
        condition_overrides=SHIFT_CONDITIONS,
        report_bindings=[
            {
                "result_id": "evolution_shift_run",
                "paper_ref": "fig:evolution",
                "source_files": [
                    "experiments/evolution_shift_normal.json",
                    "experiments/evolution_shift_no_evolution.json",
                    "experiments/evolution_evidence.json",
                ],
            }
        ],
    )

    print_header()
    total_start = time.perf_counter()

    # Sub-experiment 1: Long run
    for cond_name, overrides in LONG_CONDITIONS.items():
        run_condition_common(
            cond_name, overrides, out_dir, "evolution_", SEEDS, LONG_STEPS, SAMPLE_EVERY
        )

    # Sub-experiment 2: Environmental shift
    for cond_name, overrides in SHIFT_CONDITIONS.items():
        run_condition_common(
            cond_name, overrides, out_dir, "evolution_", SEEDS, SHIFT_STEPS, SAMPLE_EVERY
        )

    log(f"Total experiment time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
