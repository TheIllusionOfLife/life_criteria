"""Benchmark: sequential vs parallel seed execution."""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import life_criteria


def _load_config(seed: int, overrides: dict) -> str:
    config = json.loads(life_criteria.default_config_json())
    config["seed"] = seed
    config_path = Path(__file__).resolve().parent.parent / "configs" / "tuned_baseline.json"
    with open(config_path) as f:
        baseline = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    config.update(baseline)
    config.update(overrides)
    return json.dumps(config)


def run_one(args: tuple) -> tuple:
    seed, overrides, steps, sample_every = args
    config_json = _load_config(seed, overrides)
    t0 = time.perf_counter()
    result = json.loads(life_criteria.run_experiment_json(config_json, steps, sample_every))
    elapsed = time.perf_counter() - t0
    return seed, result["final_alive_count"], elapsed


def main() -> None:
    overrides = {
        "metabolism_mode": "graph",
        "max_alive_organisms": 100,
        "enable_memory": False,
        "environment_shift_step": 4000,
        "environment_shift_resource_rate": 0.0,
        "metabolism_efficiency_multiplier": 0.8,
        "death_energy_threshold": 0.10,
    }
    seeds4 = [200, 201, 202, 203]
    steps = 2000
    sample_every = 100

    # Sequential
    print("=== Sequential (4 seeds, 2k steps) ===")
    t0 = time.perf_counter()
    for s in seeds4:
        _, alive, elapsed = run_one((s, overrides, steps, sample_every))
        print(f"  seed={s} alive={alive} {elapsed:.1f}s")
    seq_time = time.perf_counter() - t0
    print(f"  Total: {seq_time:.1f}s")
    print()

    for n_workers in [2, 4, 6]:
        seeds = list(range(200, 200 + n_workers))
        print(f"=== Parallel {n_workers} workers ({n_workers} seeds, 2k steps) ===")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            results = list(exe.map(run_one, [(s, overrides, steps, sample_every) for s in seeds]))
        par_time = time.perf_counter() - t0
        for seed, alive, elapsed in results:
            print(f"  seed={seed} alive={alive} {elapsed:.1f}s")
        print(f"  Total: {par_time:.1f}s")
        # Compare: how long would these seeds take sequentially?
        sum_elapsed = sum(e for _, _, e in results)
        print(f"  Sum of individual times: {sum_elapsed:.1f}s")
        print(f"  Speedup: {sum_elapsed / par_time:.2f}x")
        print()


if __name__ == "__main__":
    main()
