"""Case generator module.

Given an undirected graph, enumerates every possible orientation of its edges
(i.e. every assignment of a direction to each edge) and returns only those
orientations that yield a *strongly connected* directed graph — meaning every
vertex can reach every other vertex.

For a graph with *m* edges there are 2^m possible orientations in total.
"""

from itertools import product
from typing import Iterator

import networkx as nx


def generate_strongly_connected_orientations(
    graph: nx.Graph,
) -> Iterator[nx.DiGraph]:
    """Yield all strongly-connected orientations of *graph*.

    Each orientation is a directed graph where every undirected edge ``(u, v)``
    becomes either ``(u, v)`` or ``(v, u)``.  Only orientations where every
    vertex can reach every other vertex are yielded.

    Args:
        graph: An undirected :class:`networkx.Graph`.

    Yields:
        :class:`networkx.DiGraph` instances that are strongly connected.

    Note:
        The total number of orientations checked is ``2 ** len(graph.edges)``.
        This grows exponentially, so the function is practical only for small
        graphs (roughly up to ~20 edges).
    """
    edges = list(graph.edges())
    nodes = list(graph.nodes())

    for bits in product((0, 1), repeat=len(edges)):
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        for bit, (u, v) in zip(bits, edges):
            if bit == 0:
                dg.add_edge(u, v)
            else:
                dg.add_edge(v, u)

        if nx.is_strongly_connected(dg):
            yield dg
