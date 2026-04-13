"""Score calculator module.

Provides two families of metrics for a directed graph:

1. **APSP sum** – the sum of shortest-path lengths over all ordered pairs of
   distinct vertices.
2. **n-hop neighbour count** – for a given hop distance *n*, the total number
   of (source, target) pairs whose shortest-path length equals exactly *n*,
   summed across all sources.
"""

from typing import Sequence

import networkx as nx


def calculate_apsp_sum(graph: nx.DiGraph) -> float:
    """Return the sum of all-pairs shortest-path lengths.

    Only finite (reachable) distances are included in the sum.

    Args:
        graph: A directed graph.

    Returns:
        Sum of shortest-path lengths for all ordered pairs of distinct vertices.
    """
    total = 0.0
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in lengths.items():
            if target != source:
                total += length
    return total


def calculate_nhop_neighbor_counts(
    graph: nx.DiGraph,
    hops: Sequence[int] = (2, 3, 4),
) -> dict[int, int]:
    """Return the number of node pairs at each specified hop distance.

    For each hop value *n* in *hops*, count the total number of ordered pairs
    ``(source, target)`` where the shortest path from *source* to *target* is
    exactly *n*.

    Args:
        graph: A directed graph.
        hops: Hop distances to compute.  Defaults to ``(2, 3, 4)``.

    Returns:
        A dictionary mapping each hop distance to its pair count.
    """
    hop_counts: dict[int, int] = {n: 0 for n in hops}
    hop_set = set(hops)

    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in lengths.items():
            if target != source and length in hop_set:
                hop_counts[length] += 1

    return hop_counts
