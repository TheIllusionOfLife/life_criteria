"""Orthogonality analysis: incremental predictive validity of 8th-criterion candidates.

Tests whether the 8th-candidate indicator adds R² beyond 7-criteria proxy
signals when predicting survival AUC.  Includes permutation test, bootstrap
CIs, and cross-validated ΔR².

Usage:
    uv run python scripts/analyze_orthogonality_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"

# ---------------------------------------------------------------------------
# 7-criteria proxy features (late-window averages per seed)
# ---------------------------------------------------------------------------

_PROXY_FEATURES = [
    "energy_mean",
    "waste_mean",
    "boundary_mean",
    "genome_diversity",
    "spatial_cohesion_mean",
    "mean_generation",
    "mean_genome_drift",
]

# ---------------------------------------------------------------------------
# Experiment data sources
# ---------------------------------------------------------------------------

# Candidate A (memory): normal + stress regimes
_CANDIDATE_A_FILES = {
    "normal": {
        "baseline": "criterion8_baseline.json",
        "enabled": "criterion8_criterion8_on.json",
    },
    "famine": {
        "baseline": "stress_famine_baseline.json",
        "enabled": "stress_famine_criterion8_on.json",
    },
    "boom_bust": {
        "baseline": "stress_boom_bust_baseline.json",
        "enabled": "stress_boom_bust_criterion8_on.json",
    },
}

# Candidate B (kin-sensing): stress regimes only
_CANDIDATE_B_FILES = {
    "famine": {
        "baseline": "candidateB_famine_baseline.json",
        "enabled": "candidateB_famine_candidateB_on.json",
    },
    "boom_bust": {
        "baseline": "candidateB_boom_bust_baseline.json",
        "enabled": "candidateB_boom_bust_candidateB_on.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _survival_auc(result: dict) -> float:
    """Sum of alive_count across all sample steps."""
    return float(sum(s["alive_count"] for s in result.get("samples", [])))


def _late_window_proxies(result: dict, start: int = 5000, end: int = 10_000) -> dict | None:
    """Extract mean of each proxy feature in the late window for one seed."""
    samples = [
        s for s in result.get("samples", []) if start <= s["step"] <= end
    ]
    if len(samples) < 2:
        return None
    proxies = {}
    for feat in _PROXY_FEATURES:
        vals = [s[feat] for s in samples if feat in s]
        if vals:
            proxies[feat] = float(np.mean(vals))
        else:
            proxies[feat] = 0.0
    return proxies


def _load_condition_data(
    files_map: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load proxy features (X), survival AUC (y), and candidate indicator (z).

    Returns (X, y, z) arrays aligned by seed order.
    """
    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    rows_z: list[float] = []

    for _regime, file_pair in files_map.items():
        for cond_label, filename in file_pair.items():
            path = EXP_DIR / filename
            if not path.exists():
                continue
            with open(path) as f:
                results = json.load(f)
            indicator = 1.0 if cond_label == "enabled" else 0.0
            for r in results:
                auc = _survival_auc(r)
                proxies = _late_window_proxies(r)
                if proxies is None:
                    continue
                rows_X.append([proxies[f] for f in _PROXY_FEATURES])
                rows_y.append(auc)
                rows_z.append(indicator)

    return np.array(rows_X), np.array(rows_y), np.array(rows_z)


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least squares R² using pseudoinverse."""
    X_aug = np.column_stack([X, np.ones(len(X))])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _partial_f_test(
    X_reduced: np.ndarray, X_full: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Partial F-test comparing reduced vs full model. Returns (F, p)."""
    from scipy import stats

    n = len(y)
    p_r = X_reduced.shape[1] + 1  # +1 for intercept
    p_f = X_full.shape[1] + 1

    X_r = np.column_stack([X_reduced, np.ones(n)])
    X_f = np.column_stack([X_full, np.ones(n)])

    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]

    ssr_r = np.sum((y - X_r @ beta_r) ** 2)
    ssr_f = np.sum((y - X_f @ beta_f) ** 2)

    df_diff = p_f - p_r
    df_resid = n - p_f

    if df_resid <= 0 or ssr_f == 0:
        return 0.0, 1.0

    f_stat = ((ssr_r - ssr_f) / df_diff) / (ssr_f / df_resid)
    p_val = 1.0 - stats.f.cdf(f_stat, df_diff, df_resid)
    return float(f_stat), float(p_val)


def _permutation_test(
    X: np.ndarray, y: np.ndarray, z: np.ndarray, n_perm: int = 10_000
) -> tuple[float, float]:
    """Permutation test: shuffle candidate indicator, compute null ΔR² distribution."""
    rng = np.random.default_rng(42)
    X_reduced = X
    X_full = np.column_stack([X, z])

    r2_reduced = _ols_r2(X_reduced, y)
    r2_full = _ols_r2(X_full, y)
    observed_delta = r2_full - r2_reduced

    null_deltas = np.empty(n_perm)
    for i in range(n_perm):
        z_perm = rng.permutation(z)
        X_full_perm = np.column_stack([X, z_perm])
        null_deltas[i] = _ols_r2(X_full_perm, y) - r2_reduced

    p_perm = float(np.mean(null_deltas >= observed_delta))
    return observed_delta, p_perm


def _bootstrap_delta_r2(
    X: np.ndarray, y: np.ndarray, z: np.ndarray, n_boot: int = 2000
) -> tuple[float, float]:
    """Bootstrap 95% CI for ΔR²."""
    rng = np.random.default_rng(42)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b, y_b, z_b = X[idx], y[idx], z[idx]
        r2_r = _ols_r2(X_b, y_b)
        r2_f = _ols_r2(np.column_stack([X_b, z_b]), y_b)
        deltas[i] = r2_f - r2_r
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _cv_delta_r2(
    X: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 5
) -> float:
    """K-fold cross-validated ΔR² (mean across folds)."""
    rng = np.random.default_rng(42)
    n = len(y)
    indices = rng.permutation(n)
    fold_size = n // k
    deltas = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < 3 or len(test_idx) < 1:
            continue

        # Train reduced model
        X_train_r = X[train_idx]
        X_train_f = np.column_stack([X[train_idx], z[train_idx]])
        y_train = y[train_idx]

        # Test
        X_test_r = X[test_idx]
        X_test_f = np.column_stack([X[test_idx], z[test_idx]])
        y_test = y[test_idx]

        # Fit + predict
        for X_tr, X_te, label in [
            (X_train_r, X_test_r, "reduced"),
            (X_train_f, X_test_f, "full"),
        ]:
            X_aug = np.column_stack([X_tr, np.ones(len(X_tr))])
            beta = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
            X_te_aug = np.column_stack([X_te, np.ones(len(X_te))])
            y_pred = X_te_aug @ beta
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if label == "reduced":
                r2_reduced = r2
            else:
                r2_full = r2
        deltas.append(r2_full - r2_reduced)

    return float(np.mean(deltas)) if deltas else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_candidate(name: str, files_map: dict) -> dict | None:
    """Run full orthogonality analysis for one candidate."""
    X, y, z = _load_condition_data(files_map)
    if len(y) < 10:
        print(f"  SKIP {name}: insufficient data ({len(y)} rows)")
        return None

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    X_reduced = X_norm
    X_full = np.column_stack([X_norm, z])

    r2_reduced = _ols_r2(X_reduced, y)
    r2_full = _ols_r2(X_full, y)
    delta_r2 = r2_full - r2_reduced

    # Partial F-test
    f_stat, f_pval = _partial_f_test(X_reduced, X_full, y)

    # Permutation test
    perm_delta, perm_p = _permutation_test(X_norm, y, z)

    # Bootstrap CI
    boot_lo, boot_hi = _bootstrap_delta_r2(X_norm, y, z)

    # Cross-validated ΔR²
    cv_delta = _cv_delta_r2(X_norm, y, z)

    result = {
        "candidate": name,
        "n_observations": int(len(y)),
        "n_enabled": int(np.sum(z)),
        "n_baseline": int(len(z) - np.sum(z)),
        "proxy_features": _PROXY_FEATURES,
        "r2_reduced": round(r2_reduced, 6),
        "r2_full": round(r2_full, 6),
        "delta_r2": round(delta_r2, 6),
        "partial_f_stat": round(f_stat, 4),
        "partial_f_pval": round(f_pval, 6),
        "permutation_delta_r2": round(perm_delta, 6),
        "permutation_p": round(perm_p, 4),
        "bootstrap_ci_lo": round(boot_lo, 6),
        "bootstrap_ci_hi": round(boot_hi, 6),
        "cv_delta_r2": round(cv_delta, 6),
    }

    print(f"\n  {name}:")
    n = result["n_observations"]
    n_en = result["n_enabled"]
    n_bl = result["n_baseline"]
    print(f"    n = {n} (enabled={n_en}, baseline={n_bl})")
    print(f"    R² reduced (7 proxies) = {result['r2_reduced']:.4f}")
    print(f"    R² full (+8th ind.)    = {result['r2_full']:.4f}")
    print(f"    ΔR²                    = {result['delta_r2']:.6f}")
    f_s = result["partial_f_stat"]
    f_p = result["partial_f_pval"]
    print(f"    Partial F = {f_s:.4f}, p = {f_p:.6f}")
    perm_d = result["permutation_delta_r2"]
    perm_p = result["permutation_p"]
    print(f"    Permutation ΔR² = {perm_d:.6f}, p = {perm_p:.4f}")
    b_lo = result["bootstrap_ci_lo"]
    b_hi = result["bootstrap_ci_hi"]
    print(f"    Bootstrap 95% CI: [{b_lo:.6f}, {b_hi:.6f}]")
    print(f"    CV ΔR² = {result['cv_delta_r2']:.6f}")

    return result


def main() -> None:
    """Run orthogonality analysis for both candidates."""
    print("Orthogonality analysis: incremental predictive validity")
    print("=" * 60)

    results = {}

    cand_a = analyze_candidate("Candidate A (Memory)", _CANDIDATE_A_FILES)
    if cand_a:
        results["candidate_a"] = cand_a

    cand_b = analyze_candidate("Candidate B (Kin-Sensing)", _CANDIDATE_B_FILES)
    if cand_b:
        results["candidate_b"] = cand_b

    if results:
        out_path = EXP_DIR / "orthogonality_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
