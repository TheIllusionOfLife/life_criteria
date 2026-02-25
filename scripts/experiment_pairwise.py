"""Pairwise criterion-ablation experiment with Graph metabolism.

Tests interaction effects between pairs of criteria to prove
interdependence (not just independent necessity).  Uses Graph metabolism
for consistency with the main ablation table (Table 3).

Pairs tested (top criteria by effect size):
  (metabolism, homeostasis), (metabolism, response),
  (reproduction, growth), (boundary, homeostasis),
  (response, homeostasis), (reproduction, evolution)

Usage:
    uv run python scripts/experiment_pairwise.py > experiments/pairwise_graph_data.tsv

Output: TSV data to stdout + summary report to stderr.
        Raw JSON saved to experiments/pairwise_graph_{pair}.json.
"""

import json
from pathlib import Path

import life_criteria
from experiment_common import (
    CRITERION_TO_FLAG,
    PAIRS,
    log,
    make_config,
    run_condition_suite,
)
from experiment_manifest import write_manifest

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 130))  # test set: seeds 100-129, n=30

GRAPH_OVERRIDES = {"metabolism_mode": "graph"}

CONDITIONS: dict[str, dict] = {"normal": {}}
for _a, _b in PAIRS:
    _cond = f"no_{_a}_no_{_b}"
    CONDITIONS[_cond] = {CRITERION_TO_FLAG[_a]: False, CRITERION_TO_FLAG[_b]: False}


def main():
    """Run pairwise criterion-ablation experiment for 6 criterion pairs."""
    log(f"Digital Life v{life_criteria.version()}")
    log(
        f"Pairwise ablation experiment: {STEPS} steps, sample every {SAMPLE_EVERY}, "
        f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})"
    )
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    base_config = json.loads(make_config(SEEDS[0], GRAPH_OVERRIDES))
    write_manifest(
        out_dir / "pairwise_graph_manifest.json",
        experiment_name="pairwise_graph_ablation",
        steps=STEPS,
        sample_every=SAMPLE_EVERY,
        seeds=SEEDS,
        base_config=base_config,
        condition_overrides={name: {**GRAPH_OVERRIDES, **ov} for name, ov in CONDITIONS.items()},
        report_bindings=[
            {
                "result_id": "pairwise_interaction",
                "paper_ref": "tab:intervention",
                "source_files": [
                    "experiments/pairwise_graph_data.tsv",
                    "experiments/pairwise_graph_statistics.json",
                ],
            }
        ],
    )

    run_condition_suite(
        "pairwise_graph_",
        CONDITIONS,
        STEPS,
        SEEDS,
        SAMPLE_EVERY,
        out_dir=out_dir,
        extra_overrides=GRAPH_OVERRIDES,
    )


if __name__ == "__main__":
    main()
