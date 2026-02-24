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
