"""Tests for the analyses/ subpackage using synthetic data.

All tests run without experiment data files on disk — inputs are constructed
in-memory so the test suite is fast and hermetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is on sys.path so the analyses package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyses.coupling.transfer_entropy import (
    discretize_series,
    transfer_entropy_from_discrete,
    transfer_entropy_lag1,
)
from analyses.results.statistics import (
    cliffs_delta,
    cohens_d,
    distribution_stats,
    holm_bonferroni,
    jonckheere_terpstra,
    paired_cohens_d,
    paired_cohens_d_ci,
    tost_equivalence,
    wilcoxon_signed_rank,
)

# ---------------------------------------------------------------------------
# analyses.results.statistics
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_empty_returns_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_unchanged(self):
        result = holm_bonferroni([0.03])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.03)

    def test_monotone_property(self):
        """Corrected p-values must be non-decreasing when sorted by original rank."""
        raw = [0.001, 0.01, 0.05, 0.2]
        corrected = holm_bonferroni(raw)
        # Holm correction: smallest raw p is multiplied by n, next by n-1, ...
        assert corrected[0] == pytest.approx(0.001 * 4)
        assert corrected[1] == pytest.approx(0.01 * 3)

    def test_caps_at_one(self):
        corrected = holm_bonferroni([0.5, 0.6])
        assert all(p <= 1.0 for p in corrected)

    def test_preserves_order(self):
        """Output index corresponds to input index, not sorted order."""
        raw = [0.2, 0.001]  # intentionally reversed
        corrected = holm_bonferroni(raw)
        # raw[1]=0.001 is smallest: corrected[1] = 0.001*2 = 0.002
        assert corrected[1] == pytest.approx(0.002)

    def test_nan_safe_and_ignored_in_ranking(self):
        raw = [0.01, float("nan"), 0.04]
        corrected = holm_bonferroni(raw)
        assert corrected[1] != corrected[1]  # NaN stays NaN
        # finite values ranked with m=2, not n=3
        assert corrected[0] == pytest.approx(0.02)
        assert corrected[2] == pytest.approx(0.04)


class TestCohensD:
    def test_identical_groups_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        # Manually: a=[10,10], b=[0,0] → d=(10-0)/sqrt((0+0)/2) → undefined,
        # use well-separated groups with SD=1
        a = np.array([10.0, 10.0, 10.0])
        b = np.array([7.0, 7.0, 7.0])
        # pooled SD = 0, so d=0.0 (guard)
        assert cohens_d(a, b) == 0.0

    def test_direction(self):
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0.0  # a > b

    def test_too_small_groups(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert cohens_d(a, b) == 0.0


class TestCliffsDelta:
    def test_perfect_dominance(self):
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(a, b) == pytest.approx(1.0)

    def test_perfect_inferiority(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert cliffs_delta(a, b) == pytest.approx(-1.0)

    def test_equal_groups_zero(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cliffs_delta(a, b) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert cliffs_delta(np.array([]), np.array([1.0])) == 0.0


class TestDistributionStats:
    def test_empty_array(self):
        result = distribution_stats(np.array([]))
        assert result["median"] == 0.0
        assert result["mean"] == 0.0

    def test_known_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = distribution_stats(arr)
        assert result["median"] == pytest.approx(3.0)
        assert result["mean"] == pytest.approx(3.0)
        assert result["q25"] == pytest.approx(2.0)
        assert result["q75"] == pytest.approx(4.0)


class TestJonckheereTerpstra:
    def test_monotone_increasing_significant(self):
        # Clear monotone trend: group means 0, 5, 10
        rng = np.random.default_rng(0)
        groups = [
            rng.normal(0, 0.1, size=20),
            rng.normal(5, 0.1, size=20),
            rng.normal(10, 0.1, size=20),
        ]
        _, p = jonckheere_terpstra(groups)
        assert p < 0.05

    def test_single_group_trivial(self):
        _, p = jonckheere_terpstra([np.array([1.0, 2.0])])
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# analyses.coupling.transfer_entropy
# ---------------------------------------------------------------------------


class TestDiscretizeSeries:
    def test_constant_signal_all_same_bin(self):
        x = np.ones(20)
        result = discretize_series(x, bins=5)
        assert np.all(result == result[0])

    def test_range_within_bins(self):
        x = np.arange(10, dtype=float)
        result = discretize_series(x, bins=5)
        assert result.min() >= 0
        assert result.max() <= 4

    def test_empty(self):
        result = discretize_series(np.array([]), bins=5)
        assert len(result) == 0


class TestTransferEntropyConstantSignals:
    """TE should be ~0 for independent or constant signals."""

    def test_constant_x_constant_y(self):
        x_prev = np.zeros(50, dtype=int)
        y_prev = np.zeros(50, dtype=int)
        y_curr = np.zeros(50, dtype=int)
        te = transfer_entropy_from_discrete(x_prev, y_prev, y_curr)
        assert te == pytest.approx(0.0)

    def test_independent_signals_near_zero(self):
        rng = np.random.default_rng(42)
        n = 200
        x = discretize_series(rng.normal(0, 1, n + 1), bins=5)
        y = discretize_series(rng.normal(0, 1, n + 1), bins=5)
        te = transfer_entropy_from_discrete(x[:-1], y[:-1], y[1:])
        # For independent signals, TE should be small (may be slightly > 0 due to
        # finite-sample estimation bias, but not large)
        assert te < 0.5

    def test_lag1_with_rng_returns_dict(self):
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        result = transfer_entropy_lag1(x, y, bins=3, permutations=20, rng=rng)
        assert result is not None
        assert "te" in result
        assert "p_value" in result
        assert result["te"] >= 0.0

    def test_lag1_too_short_returns_none(self):
        rng = np.random.default_rng(0)
        x = np.array([1.0, 2.0])  # too short (< 4)
        y = np.array([1.0, 2.0])
        result = transfer_entropy_lag1(x, y, bins=3, permutations=10, rng=rng)
        assert result is None


# ---------------------------------------------------------------------------
# analyses.results.statistics — Paired tests + TOST
# ---------------------------------------------------------------------------


class TestWilcoxonSignedRank:
    def test_basic_paired_difference(self):
        """Paired data with consistent positive shift should produce small p-value."""
        a = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        b = np.array([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
        stat, p = wilcoxon_signed_rank(a, b)
        assert p < 0.05
        assert stat >= 0

    def test_identical_arrays_p_one(self):
        """Identical paired data should produce p ≈ 1 (no difference)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = wilcoxon_signed_rank(a, a)
        assert p > 0.9

    def test_returns_tuple(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = wilcoxon_signed_rank(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestTOSTEquivalence:
    def test_within_bounds_significant(self):
        """Data with negligible effect should produce significant TOST (p < 0.05)."""
        rng = np.random.default_rng(42)
        a = rng.normal(100, 10, size=30)
        b = a + rng.normal(0, 1, size=30)  # tiny shift
        p_upper, p_lower, tost_p = tost_equivalence(a, b, sesoi=0.5)
        assert tost_p < 0.05  # equivalence demonstrated
        assert p_upper >= 0.0
        assert p_lower >= 0.0

    def test_outside_bounds_not_significant(self):
        """Data with large effect should NOT produce significant TOST."""
        rng = np.random.default_rng(42)
        a = rng.normal(100, 10, size=30)
        b = a + 20  # huge shift (d >> 0.5)
        _, _, tost_p = tost_equivalence(a, b, sesoi=0.5)
        assert tost_p > 0.05

    def test_returns_three_p_values(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = tost_equivalence(a, b, sesoi=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestPairedCohensD:
    def test_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert paired_cohens_d(a, a) == pytest.approx(0.0)

    def test_known_large_effect(self):
        """Consistent shift relative to within-pair variability → large |d|."""
        # b2 > a, so a-b2 is negative → d should be large negative
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b2 = np.array([11.0, 21.5, 31.0, 41.5, 51.0])
        d = paired_cohens_d(a, b2)
        assert abs(d) > 1.0

    def test_direction(self):
        # Add slight variability so sd_diff > 0
        a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        b = np.array([1.0, 2.1, 2.9, 4.0, 5.1])
        assert paired_cohens_d(a, b) > 0

    def test_too_small(self):
        assert paired_cohens_d(np.array([1.0]), np.array([2.0])) == 0.0


class TestPairedCohensDCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        a = rng.normal(10, 2, size=20)
        b = rng.normal(10, 2, size=20)
        d = paired_cohens_d(a, b)
        lo, hi = paired_cohens_d_ci(a, b)
        assert lo <= d <= hi

    def test_wider_at_lower_alpha(self):
        """99% CI should be wider than 95% CI."""
        rng = np.random.default_rng(42)
        a = rng.normal(10, 2, size=20)
        b = rng.normal(10, 2, size=20)
        lo95, hi95 = paired_cohens_d_ci(a, b, alpha=0.05)
        lo99, hi99 = paired_cohens_d_ci(a, b, alpha=0.01)
        assert (hi99 - lo99) >= (hi95 - lo95)
