"""Tests for the case generator module."""

import networkx as nx

from src.case_generator import generate_strongly_connected_orientations


def _triangle() -> nx.Graph:
    """Return a simple undirected triangle (3-cycle)."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return g


def _path_graph() -> nx.Graph:
    """Return a path graph 0-1-2 (not strongly orientable)."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    return g


class TestGenerateStronglyConnectedOrientations:
    def test_triangle_has_two_strongly_connected(self):
        """A triangle has exactly 2 strongly-connected orientations (both
        directed cycles, clockwise and counter-clockwise)."""
        orientations = list(generate_strongly_connected_orientations(_triangle()))
        assert len(orientations) == 2

    def test_all_results_are_digraphs(self):
        for dg in generate_strongly_connected_orientations(_triangle()):
            assert isinstance(dg, nx.DiGraph)

    def test_all_results_are_strongly_connected(self):
        for dg in generate_strongly_connected_orientations(_triangle()):
            assert nx.is_strongly_connected(dg)

    def test_path_graph_has_no_strongly_connected(self):
        """A simple path cannot be oriented to be strongly connected."""
        orientations = list(generate_strongly_connected_orientations(_path_graph()))
        assert len(orientations) == 0

    def test_preserves_all_nodes(self):
        g = _triangle()
        for dg in generate_strongly_connected_orientations(g):
            assert set(dg.nodes()) == set(g.nodes())

    def test_preserves_edge_count(self):
        g = _triangle()
        for dg in generate_strongly_connected_orientations(g):
            assert dg.number_of_edges() == g.number_of_edges()

    def test_empty_graph_yields_single_orientation(self):
        """A graph with no edges has exactly one (trivially empty) orientation.
        It is strongly connected only if it has 0 or 1 nodes."""
        g = nx.Graph()
        g.add_node(0)
        orientations = list(generate_strongly_connected_orientations(g))
        assert len(orientations) == 1
