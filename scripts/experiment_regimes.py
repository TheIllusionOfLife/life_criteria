"""Environment regime shift robustness experiment.

Runs full 8-condition criterion-ablation under multiple environment regimes
with GraphMetabolism mode to test external validity.

Usage:
    uv run python scripts/experiment_regimes.py
"""

import json
import sys
import time
from pathlib import Path

import digital_life

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(100, 110))  # test set subset: n=10 for robustness check

TUNED_BASELINE = {
    "boundary_decay_base_rate": 0.001,
    "boundary_repair_rate": 0.05,
    "metabolic_viability_floor": 0.1,
    "crowding_neighbor_threshold": 50.0,
    "homeostasis_decay_rate": 0.01,
    "growth_maturation_steps": 200,
    "growth_immature_metabolic_efficiency": 0.3,
    "resource_regeneration_rate": 0.01,
    "metabolism_mode": "graph",
}

REGIMES = {
    "default": {},
    "sparse": {"resource_regeneration_rate": 0.005, "world_size": 150.0},
    "crowded": {"num_organisms": 80, "agents_per_organism": 30, "world_size": 80.0},
    "scarce": {"resource_regeneration_rate": 0.003},
}

CONDITIONS = {
    "normal": {},
    "no_metabolism": {"enable_metabolism": False},
    "no_boundary": {"enable_boundary_maintenance": False},
    "no_homeostasis": {"enable_homeostasis": False},
    "no_response": {"enable_response": False},
    "no_reproduction": {"enable_reproduction": False},
    "no_evolution": {"enable_evolution": False},
    "no_growth": {"enable_growth": False},
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def make_config(seed: int, regime_overrides: dict, condition_overrides: dict) -> str:
    config = json.loads(digital_life.default_config_json())
    config["seed"] = seed
    config.update(TUNED_BASELINE)
    config.update(regime_overrides)
    config.update(condition_overrides)
    return json.dumps(config)


def run_single(seed: int, regime_overrides: dict, condition_overrides: dict) -> dict:
    config_json = make_config(seed, regime_overrides, condition_overrides)
    result_json = digital_life.run_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    return json.loads(result_json)


def main():
    log(f"Digital Life v{digital_life.version()}")
    log(f"Regime shift experiment: {STEPS} steps, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    log(f"Regimes: {list(REGIMES.keys())}")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)

    total_start = time.perf_counter()

    for regime_name, regime_overrides in REGIMES.items():
        log(f"=== Regime: {regime_name} ===")
        regime_start = time.perf_counter()

        for cond_name, cond_overrides in CONDITIONS.items():
            log(f"  --- {cond_name} ---")
            results = []

            for seed in SEEDS:
                t0 = time.perf_counter()
                result = run_single(seed, regime_overrides, cond_overrides)
                elapsed = time.perf_counter() - t0
                results.append(result)
                log(f"    seed={seed:3d}  alive={result['final_alive_count']:4d}  {elapsed:.2f}s")

            raw_path = out_dir / f"regime_{regime_name}_{cond_name}.json"
            with open(raw_path, "w") as f:
                json.dump(results, f, indent=2)

        regime_elapsed = time.perf_counter() - regime_start
        log(f"  Regime time: {regime_elapsed:.1f}s")
        log("")

    total_elapsed = time.perf_counter() - total_start
    log(f"Total experiment time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
