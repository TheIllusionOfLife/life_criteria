"""Seasonal cycle stress-test experiment — learnable temporal structure.

A 3rd perturbation regime with predictable, shorter-period resource cycles
that create a learnable temporal pattern.  This is the "fairest test" for
Candidate A (memory/EMA): if memory can track seasonal resource fluctuation,
it should improve survival during predictable downturns.

Seasonal Cycle
--------------
  Period 1,000 steps → 10 full cycles in a 10k-step run.
  Resource rate oscillates between baseline (0.003) and low (0.0005).
  Moderate stress calibrated to ~40-60% baseline extinction.

  EMA compatibility: period=1000 → angular freq ω≈0.006.
  For EMA to track, need α ≈ 0.003-0.02 (effective horizon 50-333 steps).

Both Candidates Tested
----------------------
  Candidate A (memory/EMA): 4 conditions × seeds
  Candidate B (kin-sensing): 4 conditions × seeds

  Rationale: seasonality modulates density/relatedness, potentially creating
  usable kin structure for Candidate B.

Tiers
-----
  --tier calibration : 5 seeds (0-4) for parameter tuning
  --tier 1           : 30 seeds (100-129) held-out final test

Usage
-----
    uv run python scripts/experiment_seasonal_stress.py --candidate A --tier calibration
    uv run python scripts/experiment_seasonal_stress.py --candidate B --tier 1 --workers 6
    uv run python scripts/experiment_seasonal_stress.py --candidate A --tier 1 --workers 6
"""

import argparse
import json
import time
from pathlib import Path

import life_criteria
from experiment_common import log, make_config_dict, run_seeds_parallel, run_single

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

STEPS = 10_000
SAMPLE_EVERY = 100

_POPULATION_CAP = 100
_ABLATION_STEP = 5_000

_BASE_OVERRIDES: dict = {
    "metabolism_mode": "graph",
    "max_alive_organisms": _POPULATION_CAP,
}

# ---------------------------------------------------------------------------
# Seasonal cycle config
# ---------------------------------------------------------------------------

# Calibration history (baseline extinction rate, 5 seeds):
#   v1: regen=0.003, period=1000, low=0.0005, eff=0.7, thresh=0.20
#       → calibrate first, then adjust
_SEASONAL_OVERRIDES: dict = {
    "resource_regeneration_rate": 0.003,
    "environment_cycle_period": 1_000,
    "environment_cycle_low_rate": 0.0005,
    "metabolism_efficiency_multiplier": 0.7,
    "death_energy_threshold": 0.20,
}

# ---------------------------------------------------------------------------
# Conditions — Candidate A (memory/EMA)
# ---------------------------------------------------------------------------

CONDITIONS_A: dict[str, dict] = {
    "baseline": {
        **_BASE_OVERRIDES,
        "enable_memory": False,
    },
    "criterion8_on": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
    },
    "criterion8_ablated": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
        "ablation_targets": ["memory"],
        "ablation_step": _ABLATION_STEP,
    },
    "sham": {
        **_BASE_OVERRIDES,
        "enable_memory": True,
        "enable_sham_process": True,
    },
}

# ---------------------------------------------------------------------------
# Conditions — Candidate B (kin-sensing)
# ---------------------------------------------------------------------------

CONDITIONS_B: dict[str, dict] = {
    "baseline": {
        **_BASE_OVERRIDES,
        "enable_collective_sensing": False,
    },
    "candidateB_on": {
        **_BASE_OVERRIDES,
        "enable_collective_sensing": True,
    },
    "candidateB_ablated": {
        **_BASE_OVERRIDES,
        "enable_collective_sensing": True,
        "ablation_targets": ["collective_sensing"],
        "ablation_step": _ABLATION_STEP,
    },
    "sham": {
        **_BASE_OVERRIDES,
        "enable_sham_collective": True,
    },
}

# ---------------------------------------------------------------------------
# Tier seed lists
# ---------------------------------------------------------------------------

TIER_SEEDS: dict[str, list[int]] = {
    "calibration": list(range(5)),
    "1": list(range(100, 130)),
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seasonal cycle stress-test for Candidates A and B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidate",
        choices=["A", "B"],
        required=True,
        help="Candidate: 'A' (memory/EMA) or 'B' (kin-sensing).",
    )
    parser.add_argument(
        "--tier",
        choices=list(TIER_SEEDS),
        required=True,
        help="'calibration' = 5 seeds; '1' = 30-seed held-out run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="N",
        help="Override seed list.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="COND",
        help="Run only named conditions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core runner (reused from experiment_criterion8_stress.py)
# ---------------------------------------------------------------------------


def _run_condition(
    cond_name: str,
    overrides: dict,
    seeds: list[int],
    out_dir: Path,
    filename_prefix: str,
    dry_run: bool,
    workers: int = 1,
) -> None:
    out_path = out_dir / f"{filename_prefix}{cond_name}.json"
    log(f"--- Condition: {cond_name} ---")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}  seeds: {seeds}")
    filtered = {k: v for k, v in overrides.items() if k != "metabolism_mode"}
    log(f"  Overrides: {json.dumps(filtered)}")
    log(f"  Output: {out_path}  workers: {workers}")

    if dry_run:
        sample_config = make_config_dict(seeds[0], overrides)
        log(f"  [dry-run] Resolved config (seed={seeds[0]}):")
        log(json.dumps(sample_config, indent=2))
        log("")
        return

    cond_start = time.perf_counter()

    if workers > 1:
        raw_results = run_seeds_parallel(
            seeds, overrides, steps=STEPS, sample_every=SAMPLE_EVERY, max_workers=workers
        )
        results: list[dict] = []
        for seed, result in zip(sorted(seeds), raw_results, strict=True):
            result["seed"] = seed
            result["condition"] = cond_name
            results.append(result)
            last_sample = result["samples"][-1] if result["samples"] else {}
            mem_str = (
                f"  mem={last_sample.get('memory_mean', 0.0):.3f}"
                if "memory_mean" in last_sample
                else ""
            )
            kf_str = (
                f"  kf={last_sample.get('kin_fraction_mean', 0.0):.3f}"
                if "kin_fraction_mean" in last_sample
                else ""
            )
            log(
                f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
                f"{mem_str}{kf_str}"
            )
        log(f"  Batch completed in {time.perf_counter() - cond_start:.1f}s")
    else:
        results = []
        for seed in seeds:
            t0 = time.perf_counter()
            result = run_single(seed, overrides, steps=STEPS, sample_every=SAMPLE_EVERY)
            elapsed = time.perf_counter() - t0
            result["seed"] = seed
            result["condition"] = cond_name
            results.append(result)
            last_sample = result["samples"][-1] if result["samples"] else {}
            mem_str = (
                f"  mem={last_sample.get('memory_mean', 0.0):.3f}"
                if "memory_mean" in last_sample
                else ""
            )
            kf_str = (
                f"  kf={last_sample.get('kin_fraction_mean', 0.0):.3f}"
                if "kin_fraction_mean" in last_sample
                else ""
            )
            log(
                f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
                f"{mem_str}{kf_str}  {elapsed:.1f}s"
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
    seeds = args.seeds if args.seeds is not None else TIER_SEEDS[args.tier]
    candidate = args.candidate

    if candidate == "A":
        all_conditions = CONDITIONS_A
        prefix = "seasonal_A_"
    else:
        all_conditions = CONDITIONS_B
        prefix = "seasonal_B_"

    active_conditions = {
        k: v
        for k, v in all_conditions.items()
        if args.conditions is None or k in args.conditions
    }

    log(f"Life Criteria v{life_criteria.version()}")
    log(f"Seasonal stress-test — Candidate {candidate}, Tier: {args.tier}")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}")
    log(f"  Seeds: {seeds}  (n={len(seeds)})")
    log(f"  Seasonal overrides: {json.dumps(_SEASONAL_OVERRIDES)}")
    log(f"  Conditions: {list(active_conditions)}")
    log(f"  Workers: {args.workers}")
    if args.dry_run:
        log("  [dry-run mode]")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    total_start = time.perf_counter()
    for cond_name, cond_overrides in active_conditions.items():
        combined = {**cond_overrides, **_SEASONAL_OVERRIDES}
        _run_condition(
            cond_name=cond_name,
            overrides=combined,
            seeds=seeds,
            out_dir=out_dir,
            filename_prefix=prefix,
            dry_run=args.dry_run,
            workers=args.workers,
        )

    if not args.dry_run:
        log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
