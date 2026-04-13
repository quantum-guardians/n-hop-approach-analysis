"""Score calculator module.

Provides two families of metrics for a directed graph:

1. **APSP sum** – the sum of shortest-path lengths over all ordered pairs of
   distinct vertices.
2. **n-hop neighbour count** – for a given hop distance *n*, the total number
   of ordered pairs ``(source, target)`` where at least one *simple path* of
   exactly *n* edges exists from *source* to *target* (``source != target``),
   summed across all sources.  A simple path visits no vertex more than once.
"""

from typing import Any, Sequence

import networkx as nx
import numpy as np


def _collect_non_self_shortest_path_lengths(graph: nx.DiGraph) -> np.ndarray:
    return np.fromiter(
        (
            length
            for source, lengths in nx.all_pairs_shortest_path_length(graph)
            for target, length in lengths.items()
            if target != source
        ),
        dtype=np.int64,
    )


def _dfs_simple(
    adj: dict[Any, list],
    node: Any,
    depth: int,
    target_depth: int,
    visited: set,
    reached: set,
) -> None:
    """DFS helper: collect all nodes reachable from *node* via a simple path of
    exactly *target_depth* edges, given the current *visited* set."""
    if depth == target_depth:
        reached.add(node)
        return
    for nbr in adj.get(node, []):
        if nbr not in visited:
            visited.add(nbr)
            _dfs_simple(adj, nbr, depth + 1, target_depth, visited, reached)
            visited.remove(nbr)


def _count_simple_path_pairs_of_length(graph: nx.DiGraph, length: int) -> int:
    """Count ordered pairs ``(source, target)`` connected by a simple path of
    exactly *length* edges (no vertex repeated)."""
    adj: dict[Any, list] = {n: list(graph.successors(n)) for n in graph.nodes()}
    total = 0
    for source in graph.nodes():
        reached: set = set()
        visited: set = {source}
        _dfs_simple(adj, source, 0, length, visited, reached)
        total += len(reached)
    return total


def calculate_apsp_sum_and_nhop_neighbor_counts(
    graph: nx.DiGraph,
    hops: Sequence[int] = (2, 3, 4),
) -> tuple[float, dict[int, int]]:
    lengths = _collect_non_self_shortest_path_lengths(graph)
    apsp_sum = float(lengths.sum()) if lengths.size else 0.0

    if not hops:
        return apsp_sum, {}

    counts: dict[int, int] = {
        hop: _count_simple_path_pairs_of_length(graph, hop) for hop in hops
    }
    return apsp_sum, counts


def calculate_apsp_sum(graph: nx.DiGraph) -> float:
    """Return the sum of all-pairs shortest-path lengths.

    Only finite (reachable) distances are included in the sum.

    Args:
        graph: A directed graph.

    Returns:
        Sum of shortest-path lengths for all ordered pairs of distinct vertices.
    """
    return calculate_apsp_sum_and_nhop_neighbor_counts(graph, hops=())[0]


def calculate_nhop_neighbor_counts(
    graph: nx.DiGraph,
    hops: Sequence[int] = (2, 3, 4),
) -> dict[int, int]:
    """Return the total n-hop neighbour count for all vertices at each hop distance.

    For each hop value *n* in *hops*, count the total number of ordered pairs
    ``(source, target)`` where at least one *simple path* of exactly *n* edges
    exists from *source* to *target* (``source != target``), summed over all
    sources.  A simple path visits no vertex more than once.

    Args:
        graph: A directed graph.
        hops: Hop distances to compute.  Defaults to ``(2, 3, 4)``.

    Returns:
        A dictionary mapping each hop distance to its pair count.
    """
    return calculate_apsp_sum_and_nhop_neighbor_counts(graph, hops=hops)[1]
