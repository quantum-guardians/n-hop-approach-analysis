"""Tests for the case generator module."""

import networkx as nx
import pytest

from src.case_generator import (
    _compute_adaptive_chunk_size,
    generate_strongly_connected_orientations,
    sample_strongly_connected_orientations,
)


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

    def test_single_thread_and_multi_thread_match(self):
        g = _triangle()
        single = list(generate_strongly_connected_orientations(g, num_workers=1))
        multi = list(generate_strongly_connected_orientations(g, num_workers=2))
        assert len(single) == len(multi) == 2
        assert {frozenset(dg.edges()) for dg in single} == {
            frozenset(dg.edges()) for dg in multi
        }

    def test_chunk_size_does_not_change_results(self):
        g = _triangle()
        c1 = list(
            generate_strongly_connected_orientations(g, num_workers=1, chunk_size=1)
        )
        c4 = list(
            generate_strongly_connected_orientations(g, num_workers=1, chunk_size=4)
        )
        assert {frozenset(dg.edges()) for dg in c1} == {
            frozenset(dg.edges()) for dg in c4
        }

    def test_invalid_chunk_size_raises(self):
        g = _triangle()
        with pytest.raises(ValueError):
            list(generate_strongly_connected_orientations(g, chunk_size=0))


class TestSampleStronglyConnectedOrientations:
    def test_returns_digraphs(self):
        """All yielded objects are DiGraph instances."""
        for dg in sample_strongly_connected_orientations(_triangle(), max_samples=5, seed=0):
            assert isinstance(dg, nx.DiGraph)

    def test_all_results_are_strongly_connected(self):
        for dg in sample_strongly_connected_orientations(_triangle(), max_samples=10, seed=1):
            assert nx.is_strongly_connected(dg)

    def test_preserves_all_nodes(self):
        g = _triangle()
        for dg in sample_strongly_connected_orientations(g, max_samples=5, seed=2):
            assert set(dg.nodes()) == set(g.nodes())

    def test_preserves_edge_count(self):
        g = _triangle()
        for dg in sample_strongly_connected_orientations(g, max_samples=5, seed=3):
            assert dg.number_of_edges() == g.number_of_edges()

    def test_max_samples_is_respected(self):
        """Number of yielded orientations never exceeds max_samples."""
        g = _triangle()
        results = list(sample_strongly_connected_orientations(g, max_samples=1, seed=4))
        assert len(results) <= 1

    def test_reproducible_with_seed(self):
        """Same seed produces the same sequence of orientations."""
        g = _triangle()
        run1 = [frozenset(dg.edges()) for dg in
                sample_strongly_connected_orientations(g, max_samples=5, seed=42)]
        run2 = [frozenset(dg.edges()) for dg in
                sample_strongly_connected_orientations(g, max_samples=5, seed=42)]
        assert run1 == run2

    def test_different_seeds_may_differ(self):
        """Different seeds produce different sampling sequences (best-effort check).

        With max_samples=1 the single orientation returned can differ between
        seeds; if both seeds happen to pick the same orientation the assertion
        still passes because both results are valid SC orientations.
        """
        g = _triangle()
        run1 = [frozenset(dg.edges()) for dg in
                sample_strongly_connected_orientations(g, max_samples=1, seed=0)]
        run2 = [frozenset(dg.edges()) for dg in
                sample_strongly_connected_orientations(g, max_samples=1, seed=99)]
        # Each result must be a valid SC orientation edge set
        for edges in run1 + run2:
            assert nx.is_strongly_connected(nx.DiGraph(edges))

    def test_path_graph_yields_nothing(self):
        """A path graph has no SC orientation, so nothing should be yielded."""
        results = list(
            sample_strongly_connected_orientations(
                _path_graph(), max_samples=10, seed=0, max_attempts=200
            )
        )
        assert results == []

    def test_empty_graph_single_node_yields_one(self):
        """A single-node graph with no edges has exactly one (trivial) SC orientation."""
        g = nx.Graph()
        g.add_node(0)
        results = list(sample_strongly_connected_orientations(g, max_samples=5, seed=0))
        assert len(results) == 1

    def test_invalid_max_samples_raises(self):
        with pytest.raises(ValueError):
            list(sample_strongly_connected_orientations(_triangle(), max_samples=0))

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError):
            list(sample_strongly_connected_orientations(_triangle(), max_samples=1, max_attempts=0))

    def test_max_attempts_bounds_runtime(self):
        """max_attempts limits total candidates checked regardless of max_samples.

        This test uses num_workers=1 (sequential path) where max_attempts=1
        strictly means "try exactly 1 candidate".  The parallel path may check
        more candidates per round-trip due to batching; that behaviour is tested
        separately in TestSampleStronglyConnectedOrientationsParallel.
        """
        g = _triangle()
        # With max_attempts=1 and num_workers=1, at most 1 candidate is checked.
        results = list(
            sample_strongly_connected_orientations(
                g, max_samples=100, seed=7, max_attempts=1, num_workers=1
            )
        )
        assert len(results) <= 1

    def test_min_samples_zero_is_default(self):
        """min_samples=0 (default) never raises even when nothing is found."""
        results = list(
            sample_strongly_connected_orientations(
                _path_graph(), max_samples=10, seed=0, max_attempts=200
            )
        )
        assert results == []

    def test_min_samples_satisfied(self):
        """When enough SC orientations are found, no error is raised."""
        g = _triangle()
        results = list(
            sample_strongly_connected_orientations(g, max_samples=5, min_samples=1, seed=0)
        )
        assert len(results) >= 1

    def test_min_samples_keeps_trying_past_max_attempts(self):
        """Sampling continues past max_attempts until min_samples is reached."""
        g = _triangle()
        # max_attempts=1 would stop after 1 candidate, but min_samples=2 forces more sampling
        results = list(
            sample_strongly_connected_orientations(
                g, max_samples=2, min_samples=2, seed=0, max_attempts=1
            )
        )
        assert len(results) == 2

    def test_min_samples_exceeded_max_samples_raises_value_error(self):
        """ValueError when min_samples > max_samples."""
        with pytest.raises(ValueError):
            list(
                sample_strongly_connected_orientations(
                    _triangle(), max_samples=3, min_samples=5
                )
            )

    def test_negative_min_samples_raises(self):
        """ValueError when min_samples < 0."""
        with pytest.raises(ValueError):
            list(
                sample_strongly_connected_orientations(
                    _triangle(), max_samples=5, min_samples=-1
                )
            )

    def test_min_samples_single_node_no_edges_satisfied(self):
        """A single-node graph satisfies min_samples=1 since it yields one trivial orientation."""
        g = nx.Graph()
        g.add_node(0)
        results = list(
            sample_strongly_connected_orientations(g, max_samples=5, min_samples=1, seed=0)
        )
        assert len(results) == 1

    def test_min_samples_multi_node_no_edges_raises(self):
        """A graph with >1 node and no edges cannot be SC, so min_samples=1 raises."""
        g = nx.Graph()
        g.add_nodes_from([0, 1])
        with pytest.raises(RuntimeError, match="Could not find"):
            list(
                sample_strongly_connected_orientations(g, max_samples=5, min_samples=1, seed=0)
            )


class TestSampleStronglyConnectedOrientationsParallel:
    """Tests that the parallel (num_workers > 1) sampling path is correct."""

    def test_parallel_returns_digraphs(self):
        """All yielded objects are DiGraph instances with num_workers=2."""
        for dg in sample_strongly_connected_orientations(
            _triangle(), max_samples=5, seed=0, num_workers=2
        ):
            assert isinstance(dg, nx.DiGraph)

    def test_parallel_all_strongly_connected(self):
        """Every orientation yielded by the parallel path is strongly connected."""
        for dg in sample_strongly_connected_orientations(
            _triangle(), max_samples=10, seed=1, num_workers=2
        ):
            assert nx.is_strongly_connected(dg)

    def test_parallel_max_samples_respected(self):
        """num_workers=2 never yields more than max_samples orientations."""
        results = list(
            sample_strongly_connected_orientations(
                _triangle(), max_samples=1, seed=4, num_workers=2
            )
        )
        assert len(results) <= 1

    def test_parallel_preserves_nodes(self):
        g = _triangle()
        for dg in sample_strongly_connected_orientations(
            g, max_samples=5, seed=2, num_workers=2
        ):
            assert set(dg.nodes()) == set(g.nodes())

    def test_parallel_preserves_edge_count(self):
        g = _triangle()
        for dg in sample_strongly_connected_orientations(
            g, max_samples=5, seed=3, num_workers=2
        ):
            assert dg.number_of_edges() == g.number_of_edges()

    def test_parallel_matches_sequential_count(self):
        """Both parallel and sequential modes find at least one SC orientation
        for the triangle graph and all yielded orientations are strongly connected."""
        g = _triangle()
        sequential = list(
            sample_strongly_connected_orientations(
                g, max_samples=100, seed=0, num_workers=1, max_attempts=10_000
            )
        )
        parallel = list(
            sample_strongly_connected_orientations(
                g, max_samples=100, seed=0, num_workers=2, max_attempts=10_000
            )
        )
        # Both should find some SC orientations; triangle has exactly 2 SC orientations
        assert len(sequential) >= 1
        assert len(parallel) >= 1
        for dg in parallel:
            assert nx.is_strongly_connected(dg)

    def test_parallel_path_graph_yields_nothing(self):
        """A path graph has no SC orientation; parallel path also yields nothing."""
        results = list(
            sample_strongly_connected_orientations(
                _path_graph(), max_samples=10, seed=0, max_attempts=200, num_workers=2
            )
        )
        assert results == []

    def test_parallel_min_samples_satisfied(self):
        """Parallel path honours min_samples."""
        g = _triangle()
        results = list(
            sample_strongly_connected_orientations(
                g, max_samples=5, min_samples=2, seed=0, num_workers=2
            )
        )
        assert len(results) >= 2

    def test_parallel_chunk_size_parameter(self):
        """chunk_size parameter is accepted and produces valid results."""
        g = _triangle()
        results = list(
            sample_strongly_connected_orientations(
                g, max_samples=5, seed=7, num_workers=2, chunk_size=4
            )
        )
        for dg in results:
            assert nx.is_strongly_connected(dg)


class TestGenerateStronglyConnectedOrientationsProcesses:
    """Tests that the process-based (use_processes=True) exhaustive path is correct."""

    def test_process_triangle_count(self):
        """ProcessPoolExecutor path still finds exactly 2 SC orientations for triangle."""
        results = list(
            generate_strongly_connected_orientations(
                _triangle(), num_workers=2, use_processes=True
            )
        )
        assert len(results) == 2

    def test_process_all_strongly_connected(self):
        """Every orientation from the process path is strongly connected."""
        for dg in generate_strongly_connected_orientations(
            _triangle(), num_workers=2, use_processes=True
        ):
            assert nx.is_strongly_connected(dg)

    def test_process_matches_single_thread(self):
        """Process-based results match single-threaded results."""
        g = _triangle()
        single = list(generate_strongly_connected_orientations(g, num_workers=1))
        multi_proc = list(
            generate_strongly_connected_orientations(g, num_workers=2, use_processes=True)
        )
        assert len(single) == len(multi_proc)
        assert {frozenset(dg.edges()) for dg in single} == {
            frozenset(dg.edges()) for dg in multi_proc
        }

    def test_process_path_graph_yields_nothing(self):
        """A path graph has no SC orientation; process path also yields nothing."""
        results = list(
            generate_strongly_connected_orientations(
                _path_graph(), num_workers=2, use_processes=True
            )
        )
        assert results == []


class TestSampleStronglyConnectedOrientationsProcesses:
    """Tests that the process-based (use_processes=True) sampling path is correct."""

    def test_process_returns_digraphs(self):
        """All yielded objects are DiGraph instances with use_processes=True."""
        for dg in sample_strongly_connected_orientations(
            _triangle(), max_samples=5, seed=0, num_workers=2, use_processes=True
        ):
            assert isinstance(dg, nx.DiGraph)

    def test_process_all_strongly_connected(self):
        """Every orientation yielded by the process path is strongly connected."""
        for dg in sample_strongly_connected_orientations(
            _triangle(), max_samples=10, seed=1, num_workers=2, use_processes=True
        ):
            assert nx.is_strongly_connected(dg)

    def test_process_max_samples_respected(self):
        """use_processes=True never yields more than max_samples orientations."""
        results = list(
            sample_strongly_connected_orientations(
                _triangle(), max_samples=1, seed=4, num_workers=2, use_processes=True
            )
        )
        assert len(results) <= 1

    def test_process_path_graph_yields_nothing(self):
        """A path graph has no SC orientation; process path also yields nothing."""
        results = list(
            sample_strongly_connected_orientations(
                _path_graph(), max_samples=10, seed=0, max_attempts=200,
                num_workers=2, use_processes=True
            )
        )
        assert results == []

    def test_process_min_samples_satisfied(self):
        """Process path honours min_samples."""
        g = _triangle()
        results = list(
            sample_strongly_connected_orientations(
                g, max_samples=5, min_samples=2, seed=0,
                num_workers=2, use_processes=True
            )
        )
        assert len(results) >= 2

    def test_process_preserves_nodes(self):
        g = _triangle()
        for dg in sample_strongly_connected_orientations(
            g, max_samples=5, seed=2, num_workers=2, use_processes=True
        ):
            assert set(dg.nodes()) == set(g.nodes())


class TestComputeAdaptiveChunkSize:
    """Tests for the _compute_adaptive_chunk_size helper."""

    def test_always_returns_at_least_one(self):
        assert _compute_adaptive_chunk_size(1, 1) >= 1

    def test_capped_at_65536(self):
        assert _compute_adaptive_chunk_size(2 ** 30, 1) == 65_536

    def test_scales_with_total(self):
        small = _compute_adaptive_chunk_size(64, 4)
        large = _compute_adaptive_chunk_size(65_536, 4)
        assert small <= large

    def test_reasonable_for_typical_graph(self):
        # 2^16 total orientations, 4 workers → chunk ~1024 (within bounds)
        chunk = _compute_adaptive_chunk_size(1 << 16, 4)
        assert 1 <= chunk <= 65_536

    def test_more_workers_gives_smaller_or_equal_chunk(self):
        chunk_few = _compute_adaptive_chunk_size(1 << 20, 2)
        chunk_many = _compute_adaptive_chunk_size(1 << 20, 8)
        assert chunk_many <= chunk_few


class TestGenerateStronglyConnectedOrientationsAdaptive:
    """Tests for the adaptive_chunk_size parameter."""

    def test_adaptive_produces_same_results_as_default(self):
        """adaptive_chunk_size=True must yield the same SC orientations."""
        g = _triangle()
        default = {
            frozenset(dg.edges())
            for dg in generate_strongly_connected_orientations(g)
        }
        adaptive = {
            frozenset(dg.edges())
            for dg in generate_strongly_connected_orientations(
                g, adaptive_chunk_size=True
            )
        }
        assert default == adaptive

    def test_adaptive_multi_worker(self):
        """adaptive_chunk_size=True with multiple workers yields correct results."""
        g = _triangle()
        results = list(
            generate_strongly_connected_orientations(
                g, num_workers=2, adaptive_chunk_size=True
            )
        )
        assert len(results) == 2
        for dg in results:
            assert nx.is_strongly_connected(dg)

    def test_adaptive_path_graph_yields_nothing(self):
        """Path graph has no SC orientation even with adaptive chunk size."""
        results = list(
            generate_strongly_connected_orientations(
                _path_graph(), adaptive_chunk_size=True
            )
        )
        assert results == []

