"""Pure statistical functions for criterion-ablation analysis.

All functions are stateless and depend only on numpy/scipy — safe to unit-test
with synthetic data without any experiment files on disk.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cohens_d_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wald-type CI for Cohen's d with approximate standard error.

    Uses the Hedges & Olkin (1985) approximation for the SE of d.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (0.0, 0.0)
    d = cohens_d(a, b)
    df = na + nb - 2
    se_d = np.sqrt((na + nb) / (na * nb) + d**2 / (2 * (na + nb - 2)))
    t_lo = stats.t.ppf(alpha / 2, df)
    t_hi = stats.t.ppf(1 - alpha / 2, df)
    return (float(d + t_lo * se_d), float(d + t_hi * se_d))


def bootstrap_cliffs_delta_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute bootstrap CI for Cliff's delta using percentile method."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        a_boot = a[rng.integers(0, na, size=na)]
        b_boot = b[rng.integers(0, nb, size=nb)]
        boot_deltas[i] = cliffs_delta(a_boot, b_boot)
    lo = float(np.percentile(boot_deltas, 100 * alpha / 2))
    hi = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta (nonparametric effect size).

    delta = (#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)
    Range: [-1, 1]. Positive means group a tends to be larger.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    more = 0
    less = 0
    for ai in a:
        for bj in b:
            if ai > bj:
                more += 1
            elif ai < bj:
                less += 1
    return (more - less) / (na * nb)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns corrected p-values in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)
    return corrected


def distribution_stats(arr: np.ndarray) -> dict:
    """Compute median, IQR, mean, and SD for an array."""
    if len(arr) == 0:
        return {"median": 0.0, "q25": 0.0, "q75": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def jonckheere_terpstra(groups: list[np.ndarray]) -> tuple[float, float]:
    """Jonckheere-Terpstra trend test for ordered groups.

    Tests whether there is a monotonic trend across ordered groups.
    Returns (JT statistic, two-sided p-value via normal approximation).
    """
    k = len(groups)
    if k < 2:
        return (0.0, 1.0)
    # JT statistic: sum of Mann-Whitney U for all i<j pairs
    jt = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for yj in groups[j]:
                    if xi > yj:
                        jt += 1.0
                    elif xi == yj:
                        jt += 0.5
    # Expected value and variance under null (no-tie approximation).
    # Tie correction omitted: with continuous-valued alive counts across
    # 30 seeds per group, exact ties are rare and impact is negligible.
    n_total = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]
    e_jt = (n_total**2 - sum(n**2 for n in ns)) / 4.0
    var_num = n_total**2 * (2 * n_total + 3) - sum(n**2 * (2 * n + 3) for n in ns)
    var_jt = var_num / 72.0
    if var_jt <= 0:
        return (jt, 1.0)
    z = (jt - e_jt) / np.sqrt(var_jt)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return (float(jt), float(p_value))
