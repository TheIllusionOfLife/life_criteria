"""Shared utilities for experiment scripts.

Provides common constants, configuration helpers, logging, and TSV output
functions used across experiment_final.py, experiment_proxy.py,
experiment_pairwise.py, and experiment_evolution.py.
"""

import json
import sys
import time
from pathlib import Path

import digital_life

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_tuned_baseline() -> dict:
    """Load tuned baseline parameters from configs/tuned_baseline.json."""
    path = _CONFIGS_DIR / "tuned_baseline.json"
    with open(path) as f:
        data = json.load(f)
    # Strip metadata keys that begin with '_'
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Tuned baseline parameters — loaded from configs/tuned_baseline.json.
# Calibrated via param_sweep_thresholds.py on 2026-02-12 (seeds 0-99).
TUNED_BASELINE = _load_tuned_baseline()

# Criterion name to config flag mapping
CRITERION_TO_FLAG = {
    "metabolism": "enable_metabolism",
    "boundary": "enable_boundary_maintenance",
    "homeostasis": "enable_homeostasis",
    "response": "enable_response",
    "reproduction": "enable_reproduction",
    "evolution": "enable_evolution",
    "growth": "enable_growth",
}

# Pairwise ablation pairs (top criteria by effect size)
PAIRS = [
    ("metabolism", "homeostasis"),
    ("metabolism", "response"),
    ("reproduction", "growth"),
    ("boundary", "homeostasis"),
    ("response", "homeostasis"),
    ("reproduction", "evolution"),
]

# Standard criterion-ablation conditions shared across experiment drivers.
CONDITIONS = {
    "normal": {},
    **{f"no_{criterion}": {flag: False} for criterion, flag in CRITERION_TO_FLAG.items()},
}

# TSV column headers for experiment output
TSV_COLUMNS = [
    "condition",
    "seed",
    "step",
    "alive_count",
    "energy_mean",
    "waste_mean",
    "boundary_mean",
    "birth_count",
    "death_count",
    "population_size",
    "mean_generation",
    "mean_genome_drift",
    "energy_std",
    "waste_std",
    "boundary_std",
    "mean_age",
    "genome_diversity",
    "max_generation",
]


def log(msg: str) -> None:
    """Write a message to stderr for progress reporting."""
    print(msg, file=sys.stderr)


def make_config(seed: int, overrides: dict) -> str:
    """Build a JSON config string with tuned baseline, seed, and overrides."""
    config = make_config_dict(seed, overrides)
    return json.dumps(config)


def make_config_dict(seed: int, overrides: dict) -> dict:
    """Build a config dict with tuned baseline, seed, and overrides."""
    config = json.loads(digital_life.default_config_json())
    config["seed"] = seed
    config.update(TUNED_BASELINE)
    config.update(overrides)
    return config


def run_single(seed: int, overrides: dict, steps: int = 2000, sample_every: int = 50) -> dict:
    """Run a single experiment and return parsed results."""
    config_json = make_config(seed, overrides)
    result_json = digital_life.run_experiment_json(config_json, steps, sample_every)
    return json.loads(result_json)


def print_header() -> None:
    """Print TSV column header to stdout."""
    print("\t".join(TSV_COLUMNS))


def print_sample(condition: str, seed: int, s: dict) -> None:
    """Print a single sample row as TSV to stdout."""
    vals = [
        condition,
        str(seed),
        str(s["step"]),
        str(s["alive_count"]),
        f"{s['energy_mean']:.4f}",
        f"{s['waste_mean']:.4f}",
        f"{s['boundary_mean']:.4f}",
        str(s["birth_count"]),
        str(s["death_count"]),
        str(s["population_size"]),
        f"{s['mean_generation']:.2f}",
        f"{s['mean_genome_drift']:.4f}",
        f"{s.get('energy_std', 0):.4f}",
        f"{s.get('waste_std', 0):.4f}",
        f"{s.get('boundary_std', 0):.4f}",
        f"{s.get('mean_age', 0):.1f}",
        f"{s.get('genome_diversity', 0):.4f}",
        str(s.get("max_generation", 0)),
    ]
    print("\t".join(vals))


def safe_path(base_dir: Path, *parts: str) -> Path:
    """Safely join path parts and ensure the result is within base_dir.

    Args:
        base_dir: The base directory that the resulting path must be under.
        *parts: Path components to join to base_dir.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If the resulting path escapes base_dir.
    """
    # Use resolve() to handle '..' and symlinks for absolute comparison
    base_resolved = base_dir.resolve()
    target = base_resolved.joinpath(*parts).resolve()

    # Check if target is still under base_resolved
    if not target.is_relative_to(base_resolved):
        raise ValueError(f"Security error: path {target} escapes base directory {base_resolved}")

    return target


def experiment_output_dir() -> Path:
    """Return the experiments output directory, creating it if needed."""
    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def load_json(path: Path) -> list[dict]:
    """Load a JSON file and return its contents, or empty list if missing."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def extract_final_alive(results: list[dict]) -> list[int]:
    """Extract final_alive_count from each seed's result."""
    return [r["final_alive_count"] for r in results if "samples" in r]


def run_condition_common(
    cond_name: str,
    overrides: dict,
    out_dir: Path,
    filename_prefix: str,
    seeds: list[int],
    steps: int,
    sample_every: int,
) -> None:
    """Run one condition, stream samples to stdout, and write JSON results."""
    log(f"--- Condition: {cond_name} ---")
    start = time.perf_counter()
    results = []

    for seed in seeds:
        t0 = time.perf_counter()
        result = run_single(seed, overrides, steps=steps, sample_every=sample_every)
        elapsed = time.perf_counter() - t0
        results.append(result)

        for sample in result["samples"]:
            print_sample(cond_name, seed, sample)

        log(f"  seed={seed:3d}  alive={result['final_alive_count']:4d}  {elapsed:.2f}s")

    with open(out_dir / f"{filename_prefix}{cond_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"  Condition time: {time.perf_counter() - start:.1f}s")
    log("")


def run_condition_suite(
    filename_prefix: str,
    conditions: dict,
    steps: int,
    seeds: list[int],
    sample_every: int,
    out_dir: Path | None = None,
    extra_overrides: dict | None = None,
) -> None:
    """Run all conditions in a suite.

    Streams TSV rows to stdout and writes per-condition JSON to *out_dir*.
    This is the recommended entry point for experiment scripts: define only
    ``conditions``, ``steps``, ``seeds``, and ``sample_every``, then call here.

    Args:
        filename_prefix: Output JSON files are named ``{prefix}{cond_name}.json``.
        conditions: Mapping of condition name → override dict.
        steps: Number of simulation steps per seed.
        seeds: Seed list.
        sample_every: Sampling interval.
        out_dir: Directory for JSON output. Defaults to ``experiments/``.
        extra_overrides: Config overrides applied to *every* condition (e.g.
            ``{"metabolism_mode": "graph"}``). Condition-specific overrides
            take precedence.
    """
    if out_dir is None:
        out_dir = experiment_output_dir()
    print_header()
    total_start = time.perf_counter()
    for cond_name, overrides in conditions.items():
        combined = {**(extra_overrides or {}), **overrides}
        run_condition_common(
            cond_name, combined, out_dir, filename_prefix, seeds, steps, sample_every
        )
    log(f"Total time: {time.perf_counter() - total_start:.1f}s")
