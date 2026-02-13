"""Calibration experiment for GraphMetabolism mode.

Runs normal condition with metabolism_mode="graph", seeds 0-29, 2000 steps.
Compares alive counts and energy dynamics to ToyMetabolism baseline.

Usage:
    uv run python scripts/experiment_graph_calibration.py
"""

import json
import sys
import time
from pathlib import Path

import digital_life

STEPS = 2000
SAMPLE_EVERY = 50
SEEDS = list(range(0, 30))  # calibration set

TUNED_BASELINE = {
    "boundary_decay_base_rate": 0.001,
    "boundary_repair_rate": 0.05,
    "metabolic_viability_floor": 0.1,
    "crowding_neighbor_threshold": 50.0,
    "homeostasis_decay_rate": 0.01,
    "growth_maturation_steps": 200,
    "growth_immature_metabolic_efficiency": 0.3,
    "resource_regeneration_rate": 0.01,
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def make_config(seed: int, overrides: dict) -> str:
    config = json.loads(digital_life.default_config_json())
    config["seed"] = seed
    config.update(TUNED_BASELINE)
    config.update(overrides)
    return json.dumps(config)


def run_single(seed: int, overrides: dict) -> dict:
    config_json = make_config(seed, overrides)
    result_json = digital_life.run_experiment_json(config_json, STEPS, SAMPLE_EVERY)
    return json.loads(result_json)


def summarize_results(label: str, results: list[dict]) -> dict:
    alive_counts = [r["final_alive_count"] for r in results]
    energies = [r["samples"][-1]["energy_mean"] for r in results if r["samples"]]
    return {
        "label": label,
        "n": len(results),
        "alive_mean": sum(alive_counts) / len(alive_counts),
        "alive_min": min(alive_counts),
        "alive_max": max(alive_counts),
        "energy_mean": sum(energies) / len(energies) if energies else 0.0,
    }


def main():
    log(f"Digital Life v{digital_life.version()}")
    log(f"GraphMetabolism calibration: {STEPS} steps, seeds {SEEDS[0]}-{SEEDS[-1]}")
    log("")

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)

    modes = {
        "toy": {"metabolism_mode": "toy"},
        "graph": {"metabolism_mode": "graph"},
    }

    summaries = []
    for mode_name, overrides in modes.items():
        log(f"--- Mode: {mode_name} ---")
        results = []
        t0 = time.perf_counter()

        for seed in SEEDS:
            ts = time.perf_counter()
            result = run_single(seed, overrides)
            elapsed = time.perf_counter() - ts
            results.append(result)
            log(f"  seed={seed:3d}  alive={result['final_alive_count']:4d}  {elapsed:.2f}s")

        elapsed = time.perf_counter() - t0
        log(f"  Mode time: {elapsed:.1f}s")

        raw_path = out_dir / f"calibration_{mode_name}.json"
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  Saved: {raw_path}")

        summary = summarize_results(mode_name, results)
        summaries.append(summary)
        log(f"  alive: mean={summary['alive_mean']:.1f} "
            f"[{summary['alive_min']}, {summary['alive_max']}]  "
            f"energy_mean={summary['energy_mean']:.4f}")
        log("")

    log("=== Comparison ===")
    for s in summaries:
        log(f"  {s['label']:6s}: alive_mean={s['alive_mean']:.1f}, "
            f"energy_mean={s['energy_mean']:.4f}")

    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
