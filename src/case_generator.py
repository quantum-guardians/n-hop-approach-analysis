"""Case generator module.

Given an undirected graph, enumerates every possible orientation of its edges
(i.e. every assignment of a direction to each edge) and returns only those
orientations that yield a *strongly connected* directed graph — meaning every
vertex can reach every other vertex.

For a graph with *m* edges there are 2^m possible orientations in total.
"""

import os
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
        bits_matrix = ((indices[:, None] >> shifts) & 1).astype(np.uint8, copy=False)

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
        if nx.is_strongly_connected(dg):
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
