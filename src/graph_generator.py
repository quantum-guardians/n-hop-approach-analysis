"""Graph generation module.

Provides utilities to create random undirected graphs parameterised by the
number of vertices and an edge-existence probability (connectivity percentage).
"""

import random
import networkx as nx


def generate_graph(
    num_vertices: int,
    connectivity: float,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a random undirected graph (Erdős–Rényi model).

    Args:
        num_vertices: Number of vertices in the graph.
        connectivity: Probability that any pair of vertices is connected by an
            edge.  Must be in the range [0.0, 1.0].
        seed: Optional random seed for reproducibility.

    Returns:
        A :class:`networkx.Graph` instance.

    Raises:
        ValueError: If *num_vertices* is less than 1 or *connectivity* is
            outside [0, 1].
    """
    if num_vertices < 1:
        raise ValueError(f"num_vertices must be >= 1, got {num_vertices}")
    if not 0.0 <= connectivity <= 1.0:
        raise ValueError(
            f"connectivity must be in [0, 1], got {connectivity}"
        )

    return nx.erdos_renyi_graph(num_vertices, connectivity, seed=seed)
