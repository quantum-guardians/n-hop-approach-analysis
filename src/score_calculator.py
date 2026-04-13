"""Score calculator module.

Provides two families of metrics for a directed graph:

1. **APSP sum** – the sum of shortest-path lengths over all ordered pairs of
   distinct vertices.
2. **n-hop neighbour count** – for a given hop distance *n*, the total number
   of ordered pairs ``(source, target)`` where at least one path of exactly *n*
   edges exists from *source* to *target* (``source != target``), summed across
   all sources.  This is computed via the *n*-th power of the adjacency matrix:
   ``A^n[i][j] > 0`` iff such a path exists.
"""

from typing import Sequence

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


def calculate_apsp_sum_and_nhop_neighbor_counts(
    graph: nx.DiGraph,
    hops: Sequence[int] = (2, 3, 4),
) -> tuple[float, dict[int, int]]:
    lengths = _collect_non_self_shortest_path_lengths(graph)
    apsp_sum = float(lengths.sum()) if lengths.size else 0.0

    if not hops:
        return apsp_sum, {}

    if graph.number_of_nodes() == 0:
        return apsp_sum, {hop: 0 for hop in hops}

    A = nx.to_numpy_array(graph)

    A_bool = A != 0
    max_hop = max(hops)
    hops_set = set(hops)

    current = A_bool.copy()  # exactly-1-hop reachability
    counts: dict[int, int] = {}

    for k in range(1, max_hop + 1):
        if k > 1:
            current = (current @ A_bool) > 0
        if k in hops_set:
            mat = current.astype(np.int64)
            np.fill_diagonal(mat, 0)
            counts[k] = int(mat.sum())

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
    ``(source, target)`` where at least one path of exactly *n* edges exists
    from *source* to *target* (``source != target``), summed over all sources.
    This uses the *n*-th power of the adjacency matrix: a pair is counted when
    ``A^n[source][target] > 0``.

    Args:
        graph: A directed graph.
        hops: Hop distances to compute.  Defaults to ``(2, 3, 4)``.

    Returns:
        A dictionary mapping each hop distance to its pair count.
    """
    return calculate_apsp_sum_and_nhop_neighbor_counts(graph, hops=hops)[1]
