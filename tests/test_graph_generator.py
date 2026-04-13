"""Tests for the graph generator module."""

import pytest
import networkx as nx

from src.graph_generator import generate_graph


class TestGenerateGraph:
    def test_returns_graph(self):
        g = generate_graph(5, 0.5, seed=0)
        assert isinstance(g, nx.Graph)

    def test_vertex_count(self):
        for n in (1, 5, 10):
            g = generate_graph(n, 0.5, seed=0)
            assert g.number_of_nodes() == n

    def test_full_connectivity(self):
        """With connectivity=1.0 every pair of vertices should be connected."""
        n = 6
        g = generate_graph(n, 1.0, seed=0)
        assert g.number_of_edges() == n * (n - 1) // 2

    def test_no_connectivity(self):
        """With connectivity=0.0 there should be no edges."""
        g = generate_graph(5, 0.0, seed=0)
        assert g.number_of_edges() == 0

    def test_reproducible_with_seed(self):
        g1 = generate_graph(8, 0.5, seed=42)
        g2 = generate_graph(8, 0.5, seed=42)
        assert list(g1.edges()) == list(g2.edges())

    def test_different_seeds_may_differ(self):
        g1 = generate_graph(10, 0.5, seed=1)
        g2 = generate_graph(10, 0.5, seed=2)
        # Not guaranteed to differ, but with 10 nodes at p=0.5 it is very
        # unlikely they produce identical edge sets.
        assert set(g1.edges()) != set(g2.edges())

    def test_invalid_vertices_raises(self):
        with pytest.raises(ValueError):
            generate_graph(0, 0.5)

    def test_invalid_connectivity_low_raises(self):
        with pytest.raises(ValueError):
            generate_graph(5, -0.1)

    def test_invalid_connectivity_high_raises(self):
        with pytest.raises(ValueError):
            generate_graph(5, 1.1)


class TestDelaunayGraph:
    """Tests for the Delaunay-based planar graph (connectivity=None)."""

    def test_returns_graph(self):
        g = generate_graph(6, seed=0)
        assert isinstance(g, nx.Graph)

    def test_vertex_count(self):
        for n in (1, 5, 10, 20):
            g = generate_graph(n, seed=0)
            assert g.number_of_nodes() == n

    def test_reproducible_with_seed(self):
        g1 = generate_graph(10, seed=7)
        g2 = generate_graph(10, seed=7)
        assert set(g1.edges()) == set(g2.edges())

    def test_different_seeds_differ(self):
        g1 = generate_graph(15, seed=1)
        g2 = generate_graph(15, seed=2)
        assert set(g1.edges()) != set(g2.edges())

    def test_has_edges(self):
        """A Delaunay graph with enough vertices should have edges."""
        g = generate_graph(5, seed=0)
        assert g.number_of_edges() > 0

    def test_is_planar(self):
        """The Delaunay graph should always be planar."""
        g = generate_graph(10, seed=42)
        assert nx.is_planar(g)

    def test_is_connected(self):
        """The Delaunay triangulation of distinct random points is connected."""
        g = generate_graph(8, seed=0)
        assert nx.is_connected(g)

    def test_small_graph_two_vertices(self):
        g = generate_graph(2, seed=0)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1

    def test_single_vertex(self):
        g = generate_graph(1, seed=0)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0

    def test_invalid_vertices_raises(self):
        with pytest.raises(ValueError):
            generate_graph(0)
