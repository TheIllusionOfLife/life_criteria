"""Time-lagged correlation and population-mean utilities for coupling analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def fisher_combine(p_values: list[float]) -> float:
    """Combine p-values using Fisher's method."""
    if not p_values:
        return 1.0
    clipped = [min(max(p, 1e-12), 1.0) for p in p_values]
    stat = -2.0 * sum(np.log(p) for p in clipped)
    return float(stats.chi2.sf(stat, 2 * len(clipped)))


def load_seed_timeseries(
    path: Path,
) -> tuple[list[int], list[dict[str, np.ndarray]], dict[str, int | float]]:
    """Load per-seed time series from normal condition output."""
    import json

    with open(path) as f:
        results = json.load(f)

    seed_series: list[dict[str, np.ndarray]] = []
    steps_ref: list[int] | None = None
    total_runs = len(results)
    dropped_missing_samples = 0
    dropped_step_mismatch = 0

    for run in results:
        samples = run.get("samples", [])
        if not samples:
            dropped_missing_samples += 1
            continue
        steps = [int(s["step"]) for s in samples]
        if steps_ref is None:
            steps_ref = steps
        elif steps != steps_ref:
            dropped_step_mismatch += 1
            continue

        energy = np.array([float(s["energy_mean"]) for s in samples], dtype=float)
        boundary = np.array([float(s["boundary_mean"]) for s in samples], dtype=float)
        internal = np.array(
            [
                float(s.get("internal_state_mean", [0.0])[0])
                if s.get("internal_state_mean")
                else 0.0
                for s in samples
            ],
            dtype=float,
        )
        seed_series.append(
            {
                "energy_mean": energy,
                "boundary_mean": boundary,
                "internal_state_mean_0": internal,
            }
        )

    accepted_runs = len(seed_series)
    dropped_runs = dropped_missing_samples + dropped_step_mismatch
    quality = {
        "total_runs": total_runs,
        "accepted_runs": accepted_runs,
        "dropped_runs": dropped_runs,
        "dropped_missing_samples": dropped_missing_samples,
        "dropped_step_mismatch": dropped_step_mismatch,
        "dropped_fraction": (dropped_runs / total_runs) if total_runs else 0.0,
    }

    if steps_ref is None:
        return [], [], quality
    return steps_ref, seed_series, quality


def mean_timeseries(seed_series: list[dict[str, np.ndarray]], var_name: str) -> np.ndarray:
    """Compute mean time series across seeds for a named variable."""
    arr = np.stack([seed[var_name] for seed in seed_series], axis=0)
    return arr.mean(axis=0)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> list[dict]:
    """Compute lagged Pearson/Spearman correlations on aggregate means."""
    rows: list[dict] = []
    for lag in range(max_lag + 1):
        if lag == 0:
            x_slice, y_slice = x, y
        else:
            x_slice, y_slice = x[:-lag], y[lag:]
        if len(x_slice) < 3:
            continue
        r_p, p_p = stats.pearsonr(x_slice, y_slice)
        r_s, p_s = stats.spearmanr(x_slice, y_slice)
        rows.append(
            {
                "lag": lag,
                "pearson_r": round(float(r_p), 4),
                "pearson_p": float(p_p),
                "spearman_r": round(float(r_s), 4),
                "spearman_p": float(p_s),
                "n": len(x_slice),
            }
        )
    return rows


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean over seeds."""
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(2026)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        boot[i] = float(np.mean(sample))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def extract_final_step_means(path: Path) -> dict[str, float]:
    """Extract mean of final step values across all runs in a JSON file."""
    import json

    if not path.exists():
        return {}
    with open(path) as f:
        results = json.load(f)
    vals: dict[str, list[float]] = defaultdict(list)
    for run in results:
        samples = run.get("samples", [])
        if not samples:
            continue
        last = samples[-1]
        vals["energy_mean"].append(float(last["energy_mean"]))
        vals["waste_mean"].append(float(last["waste_mean"]))
        vals["boundary_mean"].append(float(last["boundary_mean"]))
        internal_state = last.get("internal_state_mean")
        if internal_state:
            vals["internal_state_mean_0"].append(float(internal_state[0]))
    return {k: float(np.mean(v)) for k, v in vals.items() if v}
