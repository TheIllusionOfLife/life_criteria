"""Final criterion-ablation experiment with GraphMetabolism (2000 steps, n=30, test set).

Runs 8 conditions (normal baseline + 7 criterion ablations) with
metabolism_mode="graph", seeds 100-129, 2000 steps.

Usage:
    uv run python scripts/experiment_final_graph.py > experiments/final_graph_data.tsv
"""

import json
from pathlib import Path

import life_criteria
from experiment_common import CONDITIONS, log, make_config, run_condition_suite
from experiment_manifest import write_manifest

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}


def main():
    log(f"Digital Life v{life_criteria.version()}")
    log(
        f"Final GraphMetabolism experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    base_config = json.loads(make_config(SEEDS[0], GRAPH_OVERRIDES))
    write_manifest(
        out_dir / "final_graph_manifest.json",
        experiment_name="final_graph_ablation",
        steps=STEPS,
        sample_every=SAMPLE_EVERY,
        seeds=SEEDS,
        base_config=base_config,
        condition_overrides={name: {**GRAPH_OVERRIDES, **ov} for name, ov in CONDITIONS.items()},
        report_bindings=[
            {
                "result_id": "ablation_primary",
                "paper_ref": "tab:ablation",
                "source_files": [
                    "experiments/final_graph_data.tsv",
                    "experiments/final_graph_statistics.json",
                ],
            },
            {
                "result_id": "coupling_main",
                "paper_ref": "fig:coupling",
                "source_files": [
                    "experiments/final_graph_normal.json",
                    "experiments/coupling_analysis.json",
                ],
            },
        ],
    )

    run_condition_suite(
        "final_graph_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        out_dir=out_dir,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
