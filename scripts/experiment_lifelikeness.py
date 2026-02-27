"""Life-likeness gap analysis experiment — long-horizon diagnostic runs.

PRE-REGISTERED DECISION RULES (locked 2026-02-25, BEFORE execution)
=====================================================================
These rules determine which 8th-criterion candidate is selected from
the Tier 1 (10k step) diagnostic data.  They are written here before
any simulation results exist.  First-match-wins; applied in order.

  Rule 1 — Extinction:
      If extinction_fraction_at_10k > 0.50 (majority of seeds extinct)
      → candidate = "adaptive_robustness"
      Rationale: organisms lack within-lifetime regulation of their own
      homeostasis thresholds in response to sustained stress — analogous
      to epigenetic regulation.

  Rule 2 — Complexity plateau:
      Elif median genome_diversity slope over steps 5k–10k <= 0
      → candidate = "generative_capacity"
      Rationale: fixed-length genomes cannot generate qualitatively new
      functions; variable-length genome with gene duplication is required.

  Rule 3 — Novelty rate collapse:
      Elif birth-rate half-life (exponential decay fit) < 5000 steps
      → candidate = "niche_construction"
      Rationale: organisms do not deposit environmental signals that
      create new selective pressures for future generations.

  Rule 4 — None triggered:
      Else → candidate = "open_ended_already"
      The 7 criteria are sufficient for open-ended evolution; the paper
      proves their necessity instead of proposing an 8th.
=====================================================================

Conditions
----------
  normal_graph        : baseline 7-criteria system with graph metabolism
  shift_graph         : same + resource-regeneration shift at step 5 000
                        (for adaptation-lag measurement, Panel D of figure)
  normal_graph_memory : normal_graph + enable_memory=True (8th criterion)
  shift_graph_memory  : shift_graph + enable_memory=True (8th criterion)

Tiers
-----
  --tier 1  (diagnostic)     : 10 000 steps, sample_every=100, n=30 seeds
  --tier 2  (open-endedness) : 100 000 steps, sample_every=500, n=3 seeds,
                               normal_graph only

Usage
-----
    uv run python scripts/experiment_lifelikeness.py --tier 1 --dry-run
    uv run python scripts/experiment_lifelikeness.py --tier 1 --seeds 0 1 2
    uv run python scripts/experiment_lifelikeness.py --tier 1
    uv run python scripts/experiment_lifelikeness.py --tier 2

Output (experiments/)
---------------------
    lifelikeness_t1_normal_graph.json
    lifelikeness_t1_shift_graph.json
    lifelikeness_t2_normal_graph.json

Each JSON file is a list of per-seed RunSummary dicts, each augmented
with a top-level ``"seed"`` key.
"""

import argparse
import json
import time
from pathlib import Path

import life_criteria
from experiment_common import log, make_config_dict, run_single

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

TIER_PARAMS: dict[int, dict] = {
    # Tier 1 uses held-out seeds 100–129 (calibration seeds 0–99 reserved).
    1: {"steps": 10_000, "sample_every": 100, "default_seeds": list(range(100, 130))},
    2: {"steps": 100_000, "sample_every": 500, "default_seeds": list(range(3))},
}

# max_alive_organisms caps the alive population so per-step cost stays bounded.
# Without it a 20k-step run grows to ~386 organisms, making each step ~18× more
# expensive than the 2k-step baseline (O(n_agents × k_neighbors) NN query phase).
# Value 100 = 2× the initial 50, allowing visible population growth while keeping
# runtime tractable (~8 min/seed vs 87 min/seed uncapped).
_POPULATION_CAP = 100

CONDITIONS: dict[str, dict] = {
    "normal_graph": {
        "metabolism_mode": "graph",
        "max_alive_organisms": _POPULATION_CAP,
    },
    "shift_graph": {
        "metabolism_mode": "graph",
        "max_alive_organisms": _POPULATION_CAP,
        # Shift at step 5 000 so there are 5 000 post-shift steps to observe recovery.
        "environment_shift_step": 5_000,
        "environment_shift_resource_rate": 0.003,
    },
    # Memory-enabled variants for before/after comparison on lifelikeness metrics.
    "normal_graph_memory": {
        "metabolism_mode": "graph",
        "max_alive_organisms": _POPULATION_CAP,
        "enable_memory": True,
    },
    "shift_graph_memory": {
        "metabolism_mode": "graph",
        "max_alive_organisms": _POPULATION_CAP,
        "enable_memory": True,
        "environment_shift_step": 5_000,
        "environment_shift_resource_rate": 0.003,
    },
}

# Tier 2 only needs the baseline condition (no adaptation-lag measurement needed)
_TIER2_CONDITIONS = {"normal_graph": CONDITIONS["normal_graph"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Life-likeness gap analysis — long-horizon experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for pre-registered decision rules.",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        required=True,
        help="1 = diagnostic (20k steps, n=30 seeds); 2 = open-endedness (100k steps, n=3 seeds).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="N",
        help="Override default seed list (e.g. --seeds 0 1 2 for smoke test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config JSON for seed 0 of each condition without running.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_condition(
    cond_name: str,
    overrides: dict,
    seeds: list[int],
    steps: int,
    sample_every: int,
    out_dir: Path,
    tier: int,
    dry_run: bool,
) -> None:
    """Run one condition over all seeds and write a single JSON results file."""
    out_path = out_dir / f"lifelikeness_t{tier}_{cond_name}.json"
    log(f"--- Condition: {cond_name} ---")
    log(f"  Steps: {steps}  sample_every: {sample_every}  seeds: {seeds}")
    log(f"  Output: {out_path}")

    if dry_run:
        sample_config = make_config_dict(seeds[0], overrides)
        log("  [dry-run] Resolved config (seed=0):")
        log(json.dumps(sample_config, indent=2))
        log("")
        return

    cond_start = time.perf_counter()
    results: list[dict] = []

    for seed in seeds:
        t0 = time.perf_counter()
        result = run_single(seed, overrides, steps=steps, sample_every=sample_every)
        elapsed = time.perf_counter() - t0

        # Augment with seed so downstream analysis can identify per-seed entries
        result["seed"] = seed
        results.append(result)

        log(
            f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
            f"  samples={len(result['samples']):5d}"
            f"  lineage_events={len(result.get('lineage_events', [])):6d}"
            f"  {elapsed:.1f}s"
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"  Saved {out_path}  ({time.perf_counter() - cond_start:.1f}s total)")
    log("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    params = TIER_PARAMS[args.tier]
    seeds: list[int] = args.seeds if args.seeds is not None else params["default_seeds"]
    steps: int = params["steps"]
    sample_every: int = params["sample_every"]
    conditions = CONDITIONS if args.tier == 1 else _TIER2_CONDITIONS

    log(f"Life Criteria v{life_criteria.version()}")
    log(f"Life-likeness gap analysis — Tier {args.tier}")
    log(f"  Steps: {steps}  sample_every: {sample_every}")
    log(f"  Seeds: {seeds}  (n={len(seeds)})")
    log(f"  Conditions: {list(conditions)}")
    if args.dry_run:
        log("  [dry-run mode — no simulation will be executed]")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    total_start = time.perf_counter()
    for cond_name, overrides in conditions.items():
        _run_condition(
            cond_name=cond_name,
            overrides=overrides,
            seeds=seeds,
            steps=steps,
            sample_every=sample_every,
            out_dir=out_dir,
            tier=args.tier,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
