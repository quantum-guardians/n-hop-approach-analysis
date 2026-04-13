"""Tests for the score calculator module."""

import networkx as nx
import pytest

from src.score_calculator import calculate_apsp_sum, calculate_nhop_neighbor_counts


def _directed_triangle_cw() -> nx.DiGraph:
    """Clockwise directed triangle: 0→1→2→0."""
    dg = nx.DiGraph()
    dg.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return dg


def _directed_path() -> nx.DiGraph:
    """Simple directed path: 0→1→2."""
    dg = nx.DiGraph()
    dg.add_edges_from([(0, 1), (1, 2)])
    return dg


class TestCalculateApspSum:
    def test_triangle_apsp_sum(self):
        """In a 3-cycle each node can reach the other two in 1 and 2 hops
        (total 3+3 = 6)."""
        dg = _directed_triangle_cw()
        # Distances: 0→1=1, 0→2=2, 1→2=1, 1→0=2, 2→0=1, 2→1=2  → sum = 9
        assert calculate_apsp_sum(dg) == 9

    def test_single_node_apsp_sum(self):
        dg = nx.DiGraph()
        dg.add_node(0)
        assert calculate_apsp_sum(dg) == 0.0

    def test_disconnected_ignores_unreachable(self):
        """Unreachable pairs (infinite distance) are excluded from the sum."""
        dg = _directed_path()
        # 0→1=1, 0→2=2, 1→2=1  (1→0, 2→0, 2→1 are unreachable)
        assert calculate_apsp_sum(dg) == 4.0

    def test_returns_float(self):
        assert isinstance(calculate_apsp_sum(_directed_triangle_cw()), float)


class TestCalculateNhopNeighborCounts:
    def test_triangle_2hop_count(self):
        """In the 3-cycle: 0→2 in 2 steps, 1→0 in 2 steps, 2→1 in 2 steps → 3 pairs."""
        dg = _directed_triangle_cw()
        counts = calculate_nhop_neighbor_counts(dg, hops=(2,))
        assert counts[2] == 3

    def test_triangle_1hop_count(self):
        dg = _directed_triangle_cw()
        counts = calculate_nhop_neighbor_counts(dg, hops=(1,))
        assert counts[1] == 3

    def test_default_hops(self):
        dg = _directed_triangle_cw()
        counts = calculate_nhop_neighbor_counts(dg)
        assert set(counts.keys()) == {2, 3, 4}

    def test_custom_hops(self):
        dg = _directed_triangle_cw()
        counts = calculate_nhop_neighbor_counts(dg, hops=(1, 2))
        assert set(counts.keys()) == {1, 2}

    def test_no_pairs_beyond_graph_diameter(self):
        """3-cycle has diameter 2; no pairs at hop distance 3 or 4."""
        dg = _directed_triangle_cw()
        counts = calculate_nhop_neighbor_counts(dg, hops=(3, 4))
        assert counts[3] == 0
        assert counts[4] == 0

    def test_path_graph_nhop(self):
        dg = _directed_path()
        counts = calculate_nhop_neighbor_counts(dg, hops=(1, 2))
        # 0→1 and 1→2 are at distance 1; 0→2 is at distance 2
        assert counts[1] == 2
        assert counts[2] == 1
