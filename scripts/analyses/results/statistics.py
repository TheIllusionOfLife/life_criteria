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
    NaN p-values are ignored for ranking and returned as NaN.
    """
    n = len(p_values)
    if n == 0:
        return []

    corrected = [float("nan")] * n
    finite = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    if not finite:
        return corrected

    finite.sort(key=lambda x: x[1])
    m = len(finite)
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(finite):
        adjusted = p * (m - rank)
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


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.

    Returns (statistic, p_value).  If all differences are zero, returns (0, 1.0).
    """
    diffs = np.asarray(a) - np.asarray(b)
    if np.all(diffs == 0):
        return (0.0, 1.0)
    result = stats.wilcoxon(diffs, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test.

    Returns (statistic, p_value). If either sample has <2 observations,
    returns (NaN, NaN).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def tost_equivalence(
    a: np.ndarray, b: np.ndarray, sesoi: float = 0.5
) -> tuple[float, float, float]:
    """Two One-Sided Tests (TOST) for equivalence using paired Cohen's d.

    Tests whether the paired effect size lies within [-sesoi, +sesoi].
    Returns (p_upper, p_lower, tost_p) where tost_p = max(p_upper, p_lower).
    Significant tost_p (< alpha) means equivalence within the SESOI bounds.
    """
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    if n < 2:
        return (1.0, 1.0, 1.0)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    if sd_d == 0:
        # Zero variance in differences — if mean_d is within bounds, perfectly equivalent
        if abs(mean_d) == 0:
            return (0.0, 0.0, 0.0)
        return (1.0, 1.0, 1.0)
    se = sd_d / np.sqrt(n)
    df = n - 1
    # Convert SESOI from Cohen's d units to raw-score units
    delta = sesoi * sd_d
    # Upper test: H0: mean_diff >= delta → H1: mean_diff < delta
    t_upper = (mean_d - delta) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    # Lower test: H0: mean_diff <= -delta → H1: mean_diff > -delta
    t_lower = (mean_d + delta) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    tost_p = max(p_upper, p_lower)
    return (p_upper, p_lower, tost_p)


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (d_z): mean(diff) / sd(diff)."""
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    if n < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    mean = float(np.mean(diffs))
    if sd == 0:
        return float(np.copysign(np.inf, mean)) if mean != 0 else 0.0
    return float(mean / sd)


def paired_cohens_d_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Wald-type CI for paired Cohen's d_z.

    Uses the approximate SE: sqrt(1/n + d^2 / (2*n)).
    """
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    if n < 2:
        return (0.0, 0.0)
    d = paired_cohens_d(a, b)
    se_d = np.sqrt(1.0 / n + d**2 / (2.0 * n))
    df = n - 1
    t_crit_lo = float(stats.t.ppf(alpha / 2, df))
    t_crit_hi = float(stats.t.ppf(1 - alpha / 2, df))
    return (d + t_crit_lo * se_d, d + t_crit_hi * se_d)


def run_paired_comparison(a: np.ndarray, b: np.ndarray, sesoi: float = 0.5) -> dict:
    """Run full paired comparison suite: Wilcoxon + TOST + paired d + CI + Cliff's delta.

    Arguments a and b must be seed-matched (same length, same seed order).
    Returns a dict of results suitable for JSON serialization.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"run_paired_comparison requires equal-length arrays (got {len(a)} vs {len(b)})"
        )
    w_stat, w_p = wilcoxon_signed_rank(a, b)
    p_upper, p_lower, tost_p = tost_equivalence(a, b, sesoi=sesoi)
    d = paired_cohens_d(a, b)
    d_lo, d_hi = paired_cohens_d_ci(a, b)
    cd = cliffs_delta(a, b)
    return {
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "tost_p_upper": p_upper,
        "tost_p_lower": p_lower,
        "tost_p": tost_p,
        "tost_sesoi": sesoi,
        "paired_cohens_d": d,
        "paired_cohens_d_ci_lo": d_lo,
        "paired_cohens_d_ci_hi": d_hi,
        "cliffs_delta": cd,
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
