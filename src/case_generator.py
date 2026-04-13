"""Case generator module.

Given an undirected graph, enumerates every possible orientation of its edges
(i.e. every assignment of a direction to each edge) and returns only those
orientations that yield a *strongly connected* directed graph — meaning every
vertex can reach every other vertex.

For a graph with *m* edges there are 2^m possible orientations in total.
For large graphs, :func:`sample_strongly_connected_orientations` can be used
to draw a bounded random sample instead of exhaustively checking all 2^m cases.
"""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import networkx as nx
import numpy as np


def _build_strongly_connected_orientations_for_range(
    nodes: list[object],
    edges: list[tuple[object, object]],
    start: int,
    stop: int,
) -> list[nx.DiGraph]:
    orientations: list[nx.DiGraph] = []
    edge_count = len(edges)

    if edge_count <= 63:
        indices = np.arange(start, stop, dtype=np.uint64)
        shifts = np.arange(edge_count, dtype=np.uint64)
        bits_matrix = (indices[:, None] >> shifts) & 1

        for row in bits_matrix:
            dg = nx.DiGraph()
            dg.add_nodes_from(nodes)
            for bit, (u, v) in zip(row, edges):
                if bit == 0:
                    dg.add_edge(u, v)
                else:
                    dg.add_edge(v, u)
            if nx.is_strongly_connected(dg):
                orientations.append(dg)
        return orientations

    for idx in range(start, stop):
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        for edge_idx, (u, v) in enumerate(edges):
            if (idx >> edge_idx) & 1:
                dg.add_edge(v, u)
            else:
                dg.add_edge(u, v)
        if nx.is_strongly_connected(dg):
            orientations.append(dg)
    return orientations


def _check_sample_batch(
    nodes: list[object],
    edges: list[tuple[object, object]],
    bits_matrix: np.ndarray,
) -> list[nx.DiGraph]:
    """Check a batch of random orientation bitmasks and return strongly-connected ones.

    Args:
        nodes: List of graph nodes.
        edges: List of undirected edges ``(u, v)``.
        bits_matrix: 2-D array of shape ``(batch_size, edge_count)`` with values
            in ``{0, 1}``.  Each row encodes one candidate orientation.

    Returns:
        List of strongly-connected :class:`networkx.DiGraph` instances.
    """
    orientations: list[nx.DiGraph] = []
    for bits in bits_matrix:
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        for bit, (u, v) in zip(bits, edges):
            if bit == 0:
                dg.add_edge(u, v)
            else:
                dg.add_edge(v, u)
        if nx.is_strongly_connected(dg):
            orientations.append(dg)
    return orientations


def generate_strongly_connected_orientations(
    graph: nx.Graph,
    num_workers: int | None = None,
    chunk_size: int = 2048,
) -> Iterator[nx.DiGraph]:
    """Yield all strongly-connected orientations of *graph*.

    Each orientation is a directed graph where every undirected edge ``(u, v)``
    becomes either ``(u, v)`` or ``(v, u)``.  Only orientations where every
    vertex can reach every other vertex are yielded.

    Args:
        graph: An undirected :class:`networkx.Graph`.
        num_workers: Number of worker threads for orientation checks. If
            ``None``, uses CPU core count.
        chunk_size: Number of orientation bitmasks processed per task.

    Yields:
        :class:`networkx.DiGraph` instances that are strongly connected.

    Note:
        The total number of orientations checked is ``2 ** len(graph.edges)``.
        This grows exponentially, so the function is practical only for small
        graphs (roughly up to ~20 edges).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    edges = list(graph.edges())
    nodes = list(graph.nodes())
    edge_count = len(edges)

    if edge_count == 0:
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        if len(nodes) <= 1 or nx.is_strongly_connected(dg):
            yield dg
        return

    total_orientations = 1 << edge_count
    starts = range(0, total_orientations, chunk_size)
    workers = os.cpu_count() if num_workers is None else num_workers
    workers = max(1, workers or 1)

    if workers == 1:
        for start in starts:
            stop = min(start + chunk_size, total_orientations)
            for dg in _build_strongly_connected_orientations_for_range(
                nodes, edges, start, stop
            ):
                yield dg
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(
            lambda start: _build_strongly_connected_orientations_for_range(
                nodes, edges, start, min(start + chunk_size, total_orientations)
            ),
            starts,
        )
        for result in futures:
            for dg in result:
                yield dg


def sample_strongly_connected_orientations(
    graph: nx.Graph,
    max_samples: int,
    min_samples: int = 0,
    seed: int | None = None,
    max_attempts: int | None = None,
    num_workers: int | None = None,
    chunk_size: int = 64,
) -> Iterator[nx.DiGraph]:
    """Yield up to *max_samples* strongly-connected orientations via random sampling.

    Instead of exhaustively enumerating all ``2 ** m`` orientations, this
    function randomly samples edge-direction assignments and yields those that
    produce a strongly-connected directed graph.  Because the number of
    candidates checked is bounded by *max_attempts* (independent of graph
    size), the runtime is effectively constant with respect to the number of
    vertices/edges.

    When *min_samples* > 0 the function keeps sampling without any attempt
    limit until at least *min_samples* orientations have been found.  Only
    after that does *max_attempts* take effect (to cap the search for
    additional samples up to *max_samples*).  If the graph has no
    strongly-connected orientation at all (e.g. more than one isolated node
    with no edges) and *min_samples* > 0 the function will loop indefinitely,
    so callers should only set *min_samples* > 0 when the graph is known to
    be strongly orientable.

    When *num_workers* > 1 (or ``None``, which defaults to the CPU core count)
    random-candidate batches are evaluated in parallel using a
    :class:`~concurrent.futures.ThreadPoolExecutor`, which can substantially
    reduce wall-clock time for large *max_samples* or *max_attempts* values.

    Args:
        graph: An undirected :class:`networkx.Graph`.
        max_samples: Maximum number of strongly-connected orientations to
            yield.  Must be >= 1.
        min_samples: Minimum number of strongly-connected orientations that
            must be found before *max_attempts* is enforced.  The function
            keeps sampling until this many are found regardless of
            *max_attempts*.  Must be >= 0 and <= *max_samples*.  Defaults to
            ``0`` (no minimum enforced; *max_attempts* always applies).
        seed: Random seed for reproducibility.  If ``None``, a random seed is
            used.
        max_attempts: Maximum number of random orientations to try *after*
            *min_samples* has been satisfied (to bound additional sampling up
            to *max_samples*).  Defaults to ``max(max_samples * 100, 1_000)``.
        num_workers: Number of worker threads for parallel candidate
            evaluation.  If ``None``, uses the CPU core count.  Set to ``1``
            to disable parallelism.
        chunk_size: Number of random orientations generated and evaluated per
            worker task.  Larger values reduce task-submission overhead but
            may cause the function to exceed *max_attempts* by up to
            ``num_workers * chunk_size`` candidates.  Defaults to ``64``.
            When called via the CLI the ``--chunk-size`` flag overrides this
            default (the CLI default is ``2048``).

    Yields:
        :class:`networkx.DiGraph` instances that are strongly connected.

    Raises:
        ValueError: If *max_samples* < 1, *max_attempts* < 1, *min_samples*
            < 0, or *min_samples* > *max_samples*.
        RuntimeError: If *min_samples* > 0 and the graph has no edges and
            more than one node (making a strongly-connected orientation
            impossible).
    """
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    if min_samples < 0:
        raise ValueError(f"min_samples must be >= 0, got {min_samples}")
    if min_samples > max_samples:
        raise ValueError(
            f"min_samples ({min_samples}) must be <= max_samples ({max_samples})"
        )
    if max_attempts is None:
        max_attempts = max(max_samples * 100, 1_000)
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    edges = list(graph.edges())
    nodes = list(graph.nodes())
    edge_count = len(edges)

    if edge_count == 0:
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        found = 0
        if len(nodes) <= 1 or nx.is_strongly_connected(dg):
            found = 1
            yield dg
        if found < min_samples:
            raise RuntimeError(
                f"Could not find {min_samples} strongly-connected orientation(s): "
                f"found {found} (graph has no edges; the graph cannot be "
                f"strongly connected with more than 1 node)."
            )
        return

    workers = os.cpu_count() if num_workers is None else num_workers
    workers = max(1, workers or 1)

    rng = np.random.default_rng(seed)
    found = 0
    attempt = 0

    if workers == 1:
        while True:
            if found >= max_samples:
                break
            # Enforce max_attempts only after min_samples has already been satisfied
            if found >= min_samples and attempt >= max_attempts:
                break

            bits = rng.integers(0, 2, size=edge_count)
            dg = nx.DiGraph()
            dg.add_nodes_from(nodes)
            for bit, (u, v) in zip(bits, edges):
                if bit == 0:
                    dg.add_edge(u, v)
                else:
                    dg.add_edge(v, u)

            if nx.is_strongly_connected(dg):
                found += 1
                yield dg

            attempt += 1
        return

    # --- Parallel sampling via ThreadPoolExecutor ---
    # We maintain a sliding window of in-flight futures.  After each future
    # completes we decide whether to submit another batch or simply drain the
    # remaining in-flight work and stop.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque = deque()

        def _submit_batch() -> None:
            batch = rng.integers(0, 2, size=(chunk_size, edge_count))
            pending.append(
                executor.submit(_check_sample_batch, nodes, edges, batch)
            )

        def _should_keep_going() -> bool:
            """True when we still need to submit more work."""
            if found >= max_samples:
                return False
            # While min_samples is not yet satisfied, always keep going
            if found < min_samples:
                return True
            return attempt < max_attempts

        # Pre-fill the pipeline with one chunk per worker
        for _ in range(workers):
            _submit_batch()

        while pending:
            results = pending.popleft().result()
            attempt += chunk_size

            for dg in results:
                if found < max_samples:
                    found += 1
                    yield dg

            if _should_keep_going():
                _submit_batch()
