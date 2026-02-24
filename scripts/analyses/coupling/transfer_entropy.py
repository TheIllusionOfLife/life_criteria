"""Transfer entropy estimation with permutation significance and robustness analysis."""

from __future__ import annotations

import numpy as np


def discretize_series(values: np.ndarray, bins: int) -> np.ndarray:
    """Quantile discretization to integer bins in [0, bins-1]."""
    if len(values) == 0:
        return np.array([], dtype=int)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(values, dtype=int)
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.asarray(idx, dtype=int)


def transfer_entropy_from_discrete(
    x_prev: np.ndarray, y_prev: np.ndarray, y_curr: np.ndarray
) -> float:
    """Compute TE = I(Y_t ; X_{t-1} | Y_{t-1}) in bits."""
    n = len(y_curr)
    if n == 0:
        return 0.0

    x_bins = int(np.max(x_prev)) + 1 if len(x_prev) > 0 else 1
    y_bins = int(np.max(np.concatenate((y_prev, y_curr)))) + 1

    joint_y = np.bincount(y_prev, minlength=y_bins).astype(float)
    joint_xy = (
        np.bincount(x_prev * y_bins + y_prev, minlength=x_bins * y_bins)
        .astype(float)
        .reshape(x_bins, y_bins)
    )
    joint_yy = (
        np.bincount(y_prev * y_bins + y_curr, minlength=y_bins * y_bins)
        .astype(float)
        .reshape(y_bins, y_bins)
    )
    joint_xxy = (
        np.bincount(
            (x_prev * y_bins + y_prev) * y_bins + y_curr,
            minlength=x_bins * y_bins * y_bins,
        )
        .astype(float)
        .reshape(x_bins, y_bins, y_bins)
    )

    p_xxy = joint_xxy / n
    p_yc_given_xy = np.divide(
        joint_xxy,
        joint_xy[:, :, None],
        out=np.zeros_like(joint_xxy),
        where=joint_xy[:, :, None] > 0,
    )
    p_yc_given_y = np.divide(
        joint_yy[None, :, :],
        joint_y[None, :, None],
        out=np.zeros((1, y_bins, y_bins), dtype=float),
        where=joint_y[None, :, None] > 0,
    )

    ratio = np.divide(
        p_yc_given_xy,
        p_yc_given_y,
        out=np.zeros_like(joint_xxy),
        where=p_yc_given_y > 0,
    )
    mask = (p_xxy > 0) & (ratio > 0)
    if not np.any(mask):
        return 0.0
    te = np.sum(p_xxy[mask] * np.log2(ratio[mask]))
    return float(max(float(te), 0.0))


def transfer_entropy_lag1(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict | None:
    """Estimate TE(X->Y) with permutation p-value."""
    if len(x) < 4 or len(x) != len(y):
        return None

    # Use a shared discretization per full series to keep y(t-1), y(t) on
    # the same state space.
    x_disc = discretize_series(x, bins)
    y_disc = discretize_series(y, bins)
    x_prev = x_disc[:-1]
    y_prev = y_disc[:-1]
    y_curr = y_disc[1:]

    observed = transfer_entropy_from_discrete(x_prev, y_prev, y_curr)

    null_vals = np.empty(permutations, dtype=float)
    for i in range(permutations):
        perm_x_prev = rng.permutation(x_prev)
        null_vals[i] = transfer_entropy_from_discrete(perm_x_prev, y_prev, y_curr)

    p_val = float((np.sum(null_vals >= observed) + 1) / (permutations + 1))
    return {
        "te": float(observed),
        "p_value": p_val,
        "null_mean": float(np.mean(null_vals)),
        "null_std": float(np.std(null_vals, ddof=1)) if permutations > 1 else 0.0,
    }


def phase_randomize(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a phase-randomized surrogate preserving power spectrum."""
    n = len(series)
    if n < 4:
        return series.copy()
    spectrum = np.fft.rfft(series)
    randomized = spectrum.copy()
    if n % 2 == 0:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum) - 2)
        randomized[1:-1] = np.abs(randomized[1:-1]) * np.exp(1j * phases)
    else:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum) - 1)
        randomized[1:] = np.abs(randomized[1:]) * np.exp(1j * phases)
    surrogate = np.fft.irfft(randomized, n=n)
    return np.asarray(surrogate, dtype=float)


def te_robustness_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_settings: list[int],
    permutation_settings: list[int],
    rng_seed: int,
    phase_surrogate_samples: int,
    surrogate_permutation_floor: int,
    surrogate_permutation_divisor: int,
) -> list[dict]:
    """Compute TE sensitivity and phase-surrogate robustness grid for one pair."""
    rows: list[dict] = []
    for bins in bin_settings:
        for permutations in permutation_settings:
            te_seed = np.random.SeedSequence([rng_seed, bins, permutations, 1])
            te_rng = np.random.default_rng(te_seed)
            te = transfer_entropy_lag1(x, y, bins=bins, permutations=permutations, rng=te_rng)
            if te is None:
                continue

            phase_seed = np.random.SeedSequence([rng_seed, bins, permutations, 2])
            phase_rng = np.random.default_rng(phase_seed)
            surrogate_te = np.empty(phase_surrogate_samples, dtype=float)
            surrogate_te.fill(np.nan)
            for i in range(phase_surrogate_samples):
                x_surrogate = phase_randomize(x, phase_rng)
                y_surrogate = phase_randomize(y, phase_rng)
                te_surrogate = transfer_entropy_lag1(
                    x_surrogate,
                    y_surrogate,
                    bins=bins,
                    permutations=max(
                        surrogate_permutation_floor,
                        permutations // surrogate_permutation_divisor,
                    ),
                    rng=phase_rng,
                )
                if te_surrogate is not None:
                    surrogate_te[i] = te_surrogate["te"]

            valid_surrogates = surrogate_te[~np.isnan(surrogate_te)]
            phase_p = float(
                (np.sum(valid_surrogates >= float(te["te"])) + 1) / (len(valid_surrogates) + 1)
            )
            rows.append(
                {
                    "bins": bins,
                    "permutations": permutations,
                    "te": round(float(te["te"]), 6),
                    "p_value": float(te["p_value"]),
                    "phase_surrogate_p_value": phase_p,
                    "phase_surrogate_te_mean": round(float(np.nanmean(surrogate_te)), 6)
                    if len(valid_surrogates) > 0
                    else None,
                    "phase_surrogate_valid_n": int(len(valid_surrogates)),
                }
            )
    return rows
