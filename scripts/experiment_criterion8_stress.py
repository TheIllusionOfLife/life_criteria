"""8th Criterion stress-test experiment — harsh perturbation regimes.

Two regimes test whether memory provides survival benefit under genuine stress:

Famine
------
  Sharp resource crash at step 3,000.  Regen drops to 0.001 (1.0/step)
  vs consumption ~4.0/step, creating a real deficit that should cause
  50-80% baseline extinction over the remaining 7,000 steps.

Boom-Bust
---------
  Cyclic resource availability with period 1,000 (5 bust phases in 10k steps).
  During bust, regen = 0.001.  Organisms that "learned" from bust #1 should
  survive bust #2 better — ideal for testing memory's experience-dependence.

Conditions (same 4 as criterion8 experiment)
---------------------------------------------
  baseline          : 7-criteria system, memory disabled
  criterion8_on     : 8-criteria system, memory enabled
  criterion8_ablated: memory enabled but zeroed mid-run at step 5,000
  sham              : compute-matched random memory updates

Tiers
-----
  --tier calibration : 5 seeds (0-4) for parameter tuning
  --tier 1           : 30 seeds (100-129) held-out final test

Usage
-----
    uv run python scripts/experiment_criterion8_stress.py --regime famine --tier calibration
    uv run python scripts/experiment_criterion8_stress.py --regime boom_bust --tier calibration
    uv run python scripts/experiment_criterion8_stress.py --regime famine --tier 1
    uv run python scripts/experiment_criterion8_stress.py --regime famine --tier 1 --seeds 100 101
    uv run python scripts/experiment_criterion8_stress.py \
        --regime famine --tier calibration --dry-run

Output (experiments/)
---------------------
    stress_famine_baseline.json
    stress_famine_criterion8_on.json
    stress_boom_bust_baseline.json
    ...
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

_POPULATION_CAP = 100
_ABLATION_STEP = 5_000

# Shared overrides applied to every condition (all regimes)
_BASE_OVERRIDES: dict = {
    "metabolism_mode": "graph",
    "max_alive_organisms": _POPULATION_CAP,
}

# ---------------------------------------------------------------------------
# Perturbation regime configs
# ---------------------------------------------------------------------------

# Famine: sharp resource crash at step 3,000
_FAMINE_OVERRIDES: dict = {
    "environment_shift_step": 3_000,
    "environment_shift_resource_rate": 0.001,
}

# Boom-Bust: cyclic resource with period 1,000 (10 half-cycles, 5 busts)
_BOOM_BUST_OVERRIDES: dict = {
    "environment_cycle_period": 1_000,
    "environment_cycle_low_rate": 0.001,
}

REGIME_OVERRIDES: dict[str, dict] = {
    "famine": _FAMINE_OVERRIDES,
    "boom_bust": _BOOM_BUST_OVERRIDES,
}

# ---------------------------------------------------------------------------
# 4 conditions (identical to criterion8 experiment)
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, dict] = {
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
# Tier seed lists
# ---------------------------------------------------------------------------

TIER_SEEDS: dict[str, list[int]] = {
    "calibration": list(range(5)),  # seeds 0-4
    "1": list(range(100, 130)),  # held-out seeds 100-129
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="8th criterion stress-test — famine/boom-bust perturbation regimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for regime and condition descriptions.",
    )
    parser.add_argument(
        "--regime",
        choices=list(REGIME_OVERRIDES),
        required=True,
        help="Perturbation regime: 'famine' (sharp crash) or 'boom_bust' (cyclic).",
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
        help="Print resolved config JSON for the first seed of each condition.",
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
    filename_prefix: str,
    dry_run: bool,
) -> None:
    """Run one condition over all seeds and write results JSON."""
    out_path = out_dir / f"{filename_prefix}{cond_name}.json"
    log(f"--- Condition: {cond_name} ---")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}  seeds: {seeds}")
    filtered = {k: v for k, v in overrides.items() if k != "metabolism_mode"}
    log(f"  Overrides: {json.dumps(filtered)}")
    log(f"  Output: {out_path}")

    if dry_run:
        seed = seeds[0]
        sample_config = make_config_dict(seed, overrides)
        log(f"  [dry-run] Resolved config (seed={seed}):")
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

        last_sample = result["samples"][-1] if result["samples"] else {}
        mem_str = (
            f"  mem={last_sample.get('memory_mean', 0.0):.3f}"
            if "memory_mean" in last_sample
            else ""
        )
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
    regime = args.regime
    regime_ovr = REGIME_OVERRIDES[regime]

    active_conditions = {
        k: v for k, v in CONDITIONS.items() if args.conditions is None or k in args.conditions
    }

    log(f"Life Criteria v{life_criteria.version()}")
    log(f"8th Criterion stress-test — Regime: {regime}, Tier: {args.tier}")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}")
    log(f"  Seeds: {seeds}  (n={len(seeds)})")
    log(f"  Regime overrides: {json.dumps(regime_ovr)}")
    log(f"  Conditions: {list(active_conditions)}")
    if args.dry_run:
        log("  [dry-run mode — no simulation will be executed]")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    filename_prefix = f"stress_{regime}_"

    total_start = time.perf_counter()
    for cond_name, cond_overrides in active_conditions.items():
        # Merge: base condition overrides + regime-specific overrides
        combined = {**cond_overrides, **regime_ovr}
        _run_condition(
            cond_name=cond_name,
            overrides=combined,
            seeds=seeds,
            out_dir=out_dir,
            filename_prefix=filename_prefix,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        log(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
