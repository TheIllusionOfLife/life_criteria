"""8th Criterion (Learning/Memory) experiment — 4-condition comparative run.

Conditions
----------
  baseline         : 7-criteria system, memory disabled (enable_memory=False)
  criterion8_on    : 8-criteria system, memory enabled  (enable_memory=True)
  criterion8_ablated : memory enabled but zeroed mid-run at step 5 000
                      (ablation_target="memory", ablation_step=5000)
  sham             : compute-matched random memory updates, no temporal learning
                     (enable_memory=True, enable_sham_process=True)

All conditions use graph metabolism and the population cap from Phase 0.

Tiers
-----
  --tier 1  (main)  : 10 000 steps, sample_every=100, n=30 seeds
  --tier smoke      : 10 000 steps, sample_every=100, n=3 seeds (quick sanity)

Usage
-----
    uv run python scripts/experiment_criterion8.py --tier smoke
    uv run python scripts/experiment_criterion8.py --tier 1
    uv run python scripts/experiment_criterion8.py --tier 1 --seeds 0 1 2
    uv run python scripts/experiment_criterion8.py --tier 1 --dry-run

Output (experiments/)
---------------------
    criterion8_baseline.json
    criterion8_criterion8_on.json
    criterion8_criterion8_ablated.json
    criterion8_sham.json

Each JSON file is a list of per-seed RunSummary dicts, each augmented with a
top-level "seed" key and the condition name.
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

STEPS = 10_000
SAMPLE_EVERY = 100

# Reuse Phase 0 population cap so runtimes remain tractable.
_POPULATION_CAP = 100

# Mid-run ablation step: enough pre-experience to accumulate a useful memory
# trace, and enough post-ablation steps to observe degradation.
_ABLATION_STEP = 5_000

# Shared overrides applied to every condition
_BASE_OVERRIDES: dict = {
    "metabolism_mode": "graph",
    "max_alive_organisms": _POPULATION_CAP,
}

CONDITIONS: dict[str, dict] = {
    # 7-criteria baseline: memory flag off
    "baseline": {
        **_BASE_OVERRIDES,
        "enable_memory": False,
    },
    # 8-criteria: memory on
    "criterion8_on": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
    },
    # Memory on but zeroed at step 5k via scheduled ablation
    "criterion8_ablated": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
        "ablation_targets": ["memory"],
        "ablation_step": _ABLATION_STEP,
    },
    # Sham: compute-matched random memory updates (no temporal learning)
    "sham": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
        "enable_sham_process": True,
    },
}

TIER_SEEDS: dict[str, list[int]] = {
    "smoke": [0, 1, 2],
    "1": list(range(30)),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="8th criterion (memory) experiment — 4-condition comparative run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for condition descriptions.",
    )
    parser.add_argument(
        "--tier",
        choices=["smoke", "1"],
        required=True,
        help="'smoke' = 3 seeds for sanity check; '1' = full 30-seed run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="N",
        help="Override seed list (e.g. --seeds 0 1 2).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        metavar="COND",
        help="Run only the named conditions (default: all 4).",
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
    out_dir: Path,
    dry_run: bool,
) -> None:
    """Run one condition over all seeds and write results to experiments/."""
    out_path = out_dir / f"criterion8_{cond_name}.json"
    log(f"--- Condition: {cond_name} ---")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}  seeds: {seeds}")
    log(f"  Overrides: {json.dumps({k: v for k, v in overrides.items() if k != 'metabolism_mode'})}")
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
        result = run_single(seed, overrides, steps=STEPS, sample_every=SAMPLE_EVERY)
        elapsed = time.perf_counter() - t0

        result["seed"] = seed
        result["condition"] = cond_name
        results.append(result)

        # Log final memory_mean if available (will be non-zero when memory is on)
        last_sample = result["samples"][-1] if result["samples"] else {}
        mem_str = f"  mem={last_sample.get('memory_mean', 0.0):.3f}" if "memory_mean" in last_sample else ""
        log(
            f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
            f"  samples={len(result['samples']):5d}"
            f"{mem_str}"
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
    seeds: list[int] = args.seeds if args.seeds is not None else TIER_SEEDS[args.tier]
    active_conditions = {
        k: v for k, v in CONDITIONS.items()
        if args.conditions is None or k in args.conditions
    }

    log(f"Life Criteria v{life_criteria.version()}")
    log(f"8th Criterion (memory) experiment — Tier {args.tier}")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}")
    log(f"  Seeds: {seeds}  (n={len(seeds)})")
    log(f"  Conditions: {list(active_conditions)}")
    if args.dry_run:
        log("  [dry-run mode — no simulation will be executed]")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    total_start = time.perf_counter()
    for cond_name, overrides in active_conditions.items():
        _run_condition(
            cond_name=cond_name,
            overrides=overrides,
            seeds=seeds,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
