"""Tests for src/commands/nhop_connectivity helper utilities."""

import pytest

from src.commands.nhop_connectivity import _bin_nhop_buckets


class TestBinNhopBuckets:
    """Tests for the _bin_nhop_buckets helper."""

    def _make_buckets(self, values: list[int]) -> tuple[dict[int, int], dict[int, int]]:
        """Build total/sc dicts where every value is SC."""
        return {v: 1 for v in values}, {v: 1 for v in values}

    def test_no_binning_when_below_threshold(self) -> None:
        total = {1: 10, 2: 20, 3: 5}
        sc = {1: 5, 2: 10}
        x, y = _bin_nhop_buckets(total, sc)
        assert x == [1.0, 2.0, 3.0]
        assert y == pytest.approx([0.5, 0.5, 0.0])

    def test_binning_when_above_threshold(self) -> None:
        # 200 distinct integer values → should be binned into 100 buckets
        total = {i: 1 for i in range(200)}
        sc = {i: 1 for i in range(0, 200, 2)}  # even values are SC
        x, y = _bin_nhop_buckets(total, sc)
        assert len(x) == 100
        assert len(y) == 100
        # Every bin should have exactly 2 values (1 SC, 1 non-SC) → ratio = 0.5
        assert all(abs(r - 0.5) < 1e-9 for r in y)

    def test_x_values_are_bin_midpoints(self) -> None:
        # 200 values from 0 to 199: bin_width = (199-0)/100 = 1.99
        # midpoint of first bin  = 0 + 0.5 * 1.99 = 0.995
        # midpoint of second bin = 0 + 1.5 * 1.99 = 2.985
        total = {i: 1 for i in range(200)}
        sc = {}
        x, y = _bin_nhop_buckets(total, sc)
        bin_width = 199 / 100
        assert x[0] == pytest.approx(0 + 0.5 * bin_width)
        assert x[1] == pytest.approx(0 + 1.5 * bin_width)

    def test_empty_bins_are_omitted(self) -> None:
        # Only 3 values but ask for 100 bins → only 3 non-empty bins
        total = {0: 1, 50: 1, 100: 1}
        sc = {0: 1, 50: 1, 100: 1}
        x, y = _bin_nhop_buckets(total, sc, num_bins=100)
        assert len(x) == 3
        assert all(r == pytest.approx(1.0) for r in y)

    def test_single_value_below_threshold(self) -> None:
        total = {42: 7}
        sc = {42: 3}
        x, y = _bin_nhop_buckets(total, sc)
        assert x == [42.0]
        assert y == pytest.approx([3 / 7])

    def test_sc_count_defaults_to_zero(self) -> None:
        total = {1: 5, 2: 5, 3: 5}
        sc: dict[int, int] = {}
        x, y = _bin_nhop_buckets(total, sc)
        assert y == pytest.approx([0.0, 0.0, 0.0])

    def test_exactly_at_threshold_no_binning(self) -> None:
        total = {i: 1 for i in range(100)}
        sc = {i: 1 for i in range(100)}
        x, y = _bin_nhop_buckets(total, sc)
        # Exactly 100 distinct values → no binning
        assert len(x) == 100
        assert all(r == pytest.approx(1.0) for r in y)

    def test_one_more_than_threshold_triggers_binning(self) -> None:
        total = {i: 1 for i in range(101)}
        sc = {}
        x, y = _bin_nhop_buckets(total, sc)
        # 101 distinct values → binned into 100 buckets
        assert len(x) <= 100
