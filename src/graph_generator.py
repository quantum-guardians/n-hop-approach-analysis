"""Graph generation module.

Provides utilities to create random undirected graphs parameterised by the
number of vertices and an optional edge-existence probability (connectivity
percentage).  When connectivity is not specified a Delaunay-triangulation-based
planar graph is generated instead.
"""

import random
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay


def generate_graph(
    num_vertices: int,
    connectivity: float | None = None,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a random undirected graph.

    When *connectivity* is given the graph is sampled from the Erdős–Rényi
    model (each pair of vertices is connected independently with probability
    *connectivity*).

    When *connectivity* is ``None`` (the default) a planar graph is produced
    by generating *num_vertices* random points in the unit square and
    connecting every pair of vertices that share an edge in the Delaunay
    triangulation of those points.

    Args:
        num_vertices: Number of vertices in the graph.
        connectivity: Probability that any pair of vertices is connected by an
            edge.  Must be in the range [0.0, 1.0].  Pass ``None`` (or omit)
            to use the Delaunay-based planar graph generator.
        seed: Optional random seed for reproducibility.

    Returns:
        A :class:`networkx.Graph` instance.

    Raises:
        ValueError: If *num_vertices* is less than 1 or *connectivity* is
            outside [0, 1].
    """
    if num_vertices < 1:
        raise ValueError(f"num_vertices must be >= 1, got {num_vertices}")

    if connectivity is None:
        return _generate_delaunay_graph(num_vertices, seed=seed)

    if not 0.0 <= connectivity <= 1.0:
        raise ValueError(
            f"connectivity must be in [0, 1], got {connectivity}"
        )

    return nx.erdos_renyi_graph(num_vertices, connectivity, seed=seed)


def _generate_delaunay_graph(
    num_vertices: int,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a planar graph via Delaunay triangulation.

    *num_vertices* random points are placed uniformly in the unit square.
    Edges are taken from the Delaunay triangulation of those points, yielding
    a connected planar graph (for ``num_vertices >= 3``).

    Args:
        num_vertices: Number of vertices (must be >= 1).
        seed: Optional random seed for reproducibility.

    Returns:
        A :class:`networkx.Graph` instance.
    """
    rng = np.random.default_rng(seed)
    points = rng.random((num_vertices, 2))

    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))

    if num_vertices < 3:
        # Delaunay triangulation requires at least 3 non-collinear points;
        # for tiny graphs simply connect all available vertices.
        graph.add_edges_from(
            (i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)
        )
        return graph

    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex[0], simplex[1], simplex[2]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))

    graph.add_edges_from(edges)
    return graph
