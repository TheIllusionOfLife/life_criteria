"""Candidate B pop-cap relaxation experiment — test with viable kin signal.

Increases max_alive_organisms from 100 → 400 to allow multi-agent organisms
to persist longer, keeping kin_fraction as a meaningful signal.  This directly
addresses the reviewer concern (I4) that Candidate B null may be an artifact
of the population cap forcing organisms to ~1 agent each (degenerate kin signal).

Three regimes
-------------
  famine     : sharp resource crash at step 3,000 (same params as PR #9)
  boom_bust  : cyclic resource with period 2,500
  seasonal   : predictable period-1000 cycle (from Phase 3)

Power justification
-------------------
  n=30 paired achieves ~80% power for d=0.5 (adequate for detecting
  medium effects).  n=15 would only give ~30-45% power — insufficient.

Tiers
-----
  --tier calibration : 5 seeds (0-4) for timing/feasibility check
  --tier 1           : 30 seeds (100-129) held-out final test

Steps: 10,000 (matched to primary stress protocols)

Usage
-----
    uv run python scripts/experiment_candidateB_relaxed_cap.py \\
        --regime famine --tier calibration
    uv run python scripts/experiment_candidateB_relaxed_cap.py \\
        --regime famine --tier 1 --workers 6
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

_POPULATION_CAP = 400  # relaxed from 100
_ABLATION_STEP = STEPS // 2

_BASE_OVERRIDES: dict = {
    "metabolism_mode": "graph",
    "max_alive_organisms": _POPULATION_CAP,
}

# ---------------------------------------------------------------------------
# Regime configs
# ---------------------------------------------------------------------------

_FAMINE_OVERRIDES: dict = {
    "environment_shift_step": STEPS * 3 // 10,
    "environment_shift_resource_rate": 0.0,
    "metabolism_efficiency_multiplier": 0.8,
    "death_energy_threshold": 0.25,
}

_BOOM_BUST_OVERRIDES: dict = {
    "resource_regeneration_rate": 0.002,
    "environment_cycle_period": STEPS // 4,
    "environment_cycle_low_rate": 0.0,
    "metabolism_efficiency_multiplier": 0.5,
    "death_energy_threshold": 0.25,
}

_SEASONAL_OVERRIDES: dict = {
    "resource_regeneration_rate": 0.003,
    "environment_cycle_period": STEPS // 10,
    "environment_cycle_low_rate": 0.0005,
    "metabolism_efficiency_multiplier": 0.7,
    "death_energy_threshold": 0.20,
}

REGIME_OVERRIDES: dict[str, dict] = {
    "famine": _FAMINE_OVERRIDES,
    "boom_bust": _BOOM_BUST_OVERRIDES,
    "seasonal": _SEASONAL_OVERRIDES,
}

# ---------------------------------------------------------------------------
# Conditions (Candidate B only)
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, dict] = {
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

TIER_SEEDS: dict[str, list[int]] = {
    "calibration": list(range(5)),
    "1": list(range(100, 130)),
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate B pop-cap relaxation (cap=400, 10k steps).",
    )
    parser.add_argument(
        "--regime",
        choices=list(REGIME_OVERRIDES),
        required=True,
    )
    parser.add_argument(
        "--tier",
        choices=list(TIER_SEEDS),
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", metavar="N")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), metavar="COND")
    parser.add_argument("--workers", type=int, default=1, metavar="N")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runner
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
            kf = last_sample.get("kin_fraction_mean", 0.0)
            log(
                f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
                f"  kf={kf:.3f}"
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
            kf = last_sample.get("kin_fraction_mean", 0.0)
            log(
                f"  seed={seed:3d}  alive={result['final_alive_count']:4d}"
                f"  kf={kf:.3f}  {elapsed:.1f}s"
            )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Saved {out_path}  ({time.perf_counter() - cond_start:.1f}s total)")
    log("")


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else TIER_SEEDS[args.tier]
    regime = args.regime
    regime_ovr = REGIME_OVERRIDES[regime]

    active_conditions = {
        k: v for k, v in CONDITIONS.items() if args.conditions is None or k in args.conditions
    }

    log(f"Life Criteria v{life_criteria.version()}")
    log(f"Candidate B pop-cap relaxation — Regime: {regime}, Tier: {args.tier}")
    log(f"  Steps: {STEPS}  sample_every: {SAMPLE_EVERY}  cap: {_POPULATION_CAP}")
    log(f"  Seeds: {seeds}  (n={len(seeds)})")
    log(f"  Regime: {json.dumps(regime_ovr)}")
    log(f"  Workers: {args.workers}")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    prefix = f"relaxed_cap_{regime}_"
    total_start = time.perf_counter()
    for cond_name, cond_overrides in active_conditions.items():
        combined = {**cond_overrides, **regime_ovr}
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
