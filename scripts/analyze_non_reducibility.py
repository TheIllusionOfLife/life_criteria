"""Non-reducibility analysis: can candidate-specific signals be predicted from 7-criteria proxies?

For each candidate, we define a candidate-specific signal:
  - Candidate A: late-window memory variance (memory_mean stability)
  - Candidate B: late-window kin_fraction mean

Then we fit cross-validated regression: signal ~ 7-criteria proxies.
Low R² → candidate signal is genuinely new information (non-reducible).
High R² → candidate signal is predictable from existing criteria (reducible).

Usage:
    uv run python scripts/analyze_non_reducibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"

_PROXY_FEATURES = [
    "energy_mean",
    "waste_mean",
    "boundary_mean",
    "genome_diversity",
    "spatial_cohesion_mean",
    "mean_generation",
    "mean_genome_drift",
]

# Data sources: only "enabled" conditions (where candidate mechanism is active)
_CANDIDATE_A_FILES = [
    "criterion8_criterion8_on.json",
    "stress_famine_criterion8_on.json",
    "stress_boom_bust_criterion8_on.json",
]

_CANDIDATE_B_FILES = [
    "candidateB_famine_candidateB_on.json",
    "candidateB_boom_bust_candidateB_on.json",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _late_proxies(result: dict, start: int = 5000, end: int = 10_000) -> list[float] | None:
    """Extract mean of each proxy feature in the late window."""
    samples = [s for s in result.get("samples", []) if start <= s["step"] <= end]
    if len(samples) < 2:
        return None
    row = []
    for feat in _PROXY_FEATURES:
        vals = [s[feat] for s in samples if feat in s]
        row.append(float(np.mean(vals)) if vals else 0.0)
    return row


def _memory_late_variance(result: dict, start: int = 5000, end: int = 10_000) -> float | None:
    """Candidate A signal: variance of memory_mean in late window."""
    vals = [
        s["memory_mean"]
        for s in result.get("samples", [])
        if start <= s["step"] <= end and "memory_mean" in s
    ]
    if len(vals) < 2:
        return None
    return float(np.var(vals, ddof=1))


def _kin_fraction_late_mean(result: dict, start: int = 5000, end: int = 10_000) -> float | None:
    """Candidate B signal: mean of kin_fraction_mean in late window."""
    vals = [
        s["kin_fraction_mean"]
        for s in result.get("samples", [])
        if start <= s["step"] <= end and "kin_fraction_mean" in s
    ]
    if not vals:
        return None
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Cross-validated R²
# ---------------------------------------------------------------------------


def _cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5) -> tuple[float, list[float]]:
    """K-fold cross-validated R². Returns (mean_r2, per_fold_r2)."""
    rng = np.random.default_rng(42)
    n = len(y)
    indices = rng.permutation(n)
    fold_size = n // k
    fold_r2s = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < 3 or len(test_idx) < 1:
            continue

        X_train = np.column_stack([X[train_idx], np.ones(len(train_idx))])
        X_test = np.column_stack([X[test_idx], np.ones(len(test_idx))])
        y_train = y[train_idx]
        y_test = y[test_idx]

        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        y_pred = X_test @ beta

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fold_r2s.append(r2)

    mean_r2 = float(np.mean(fold_r2s)) if fold_r2s else 0.0
    return mean_r2, fold_r2s


def _permutation_r2(X: np.ndarray, y: np.ndarray, n_perm: int = 5000) -> tuple[float, float]:
    """Permutation baseline: shuffle y and compute R², return (mean_null_r2, p)."""
    rng = np.random.default_rng(42)

    # Observed R²
    X_aug = np.column_stack([X, np.ones(len(X))])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    observed_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    null_r2s = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        beta_p = np.linalg.lstsq(X_aug, y_perm, rcond=None)[0]
        y_hat_p = X_aug @ beta_p
        ss_res_p = np.sum((y_perm - y_hat_p) ** 2)
        ss_tot_p = np.sum((y_perm - np.mean(y_perm)) ** 2)
        null_r2s[i] = 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else 0.0

    p = float(np.mean(null_r2s >= observed_r2))
    return float(np.mean(null_r2s)), p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_and_analyze(
    name: str,
    files: list[str],
    signal_fn,
) -> dict | None:
    """Load data and run non-reducibility analysis for one candidate."""
    rows_X: list[list[float]] = []
    rows_y: list[float] = []

    for filename in files:
        path = EXP_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)
        for r in results:
            proxies = _late_proxies(r)
            signal = signal_fn(r)
            if proxies is None or signal is None:
                continue
            rows_X.append(proxies)
            rows_y.append(signal)

    X = np.array(rows_X)
    y = np.array(rows_y)

    if len(y) < 10:
        print(f"  SKIP {name}: insufficient data ({len(y)} rows)")
        return None

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # In-sample R²
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_insample = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cross-validated R²
    cv_r2, fold_r2s = _cv_r2(X_norm, y)

    # Permutation baseline
    null_mean_r2, perm_p = _permutation_r2(X_norm, y)

    result = {
        "candidate": name,
        "n_observations": int(len(y)),
        "signal_mean": round(float(np.mean(y)), 6),
        "signal_std": round(float(np.std(y, ddof=1)), 6),
        "proxy_features": _PROXY_FEATURES,
        "r2_insample": round(r2_insample, 6),
        "r2_cv_mean": round(cv_r2, 6),
        "r2_cv_per_fold": [round(v, 4) for v in fold_r2s],
        "permutation_null_r2_mean": round(null_mean_r2, 6),
        "permutation_p": round(perm_p, 4),
        "interpretation": (
            "reducible" if cv_r2 > 0.5 else
            "partially_reducible" if cv_r2 > 0.2 else
            "non_reducible"
        ),
    }

    print(f"\n  {name}:")
    print(f"    n = {result['n_observations']}")
    print(f"    Signal: mean={result['signal_mean']:.4f}, std={result['signal_std']:.4f}")
    print(f"    In-sample R² = {result['r2_insample']:.4f}")
    print(f"    Cross-validated R² = {result['r2_cv_mean']:.4f}")
    null_r2 = result["permutation_null_r2_mean"]
    perm_p = result["permutation_p"]
    print(f"    Permutation null R² = {null_r2:.4f}, p = {perm_p:.4f}")
    print(f"    Interpretation: {result['interpretation']}")

    return result


def main() -> None:
    """Run non-reducibility analysis for both candidates."""
    print("Non-reducibility analysis: can candidate signals be predicted from 7-criteria proxies?")
    print("=" * 80)

    results = {}

    cand_a = _load_and_analyze(
        "Candidate A (Memory — late variance)",
        _CANDIDATE_A_FILES,
        _memory_late_variance,
    )
    if cand_a:
        results["candidate_a"] = cand_a

    cand_b = _load_and_analyze(
        "Candidate B (Kin-Sensing — late kin_fraction)",
        _CANDIDATE_B_FILES,
        _kin_fraction_late_mean,
    )
    if cand_b:
        results["candidate_b"] = cand_b

    if results:
        out_path = EXP_DIR / "non_reducibility_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
