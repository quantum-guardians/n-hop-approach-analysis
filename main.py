#!/usr/bin/env python3
"""Entry point for the n-hop approach analysis.

Usage examples
--------------
Run analysis on a single random graph::

    python main.py

Run with custom parameters::

    python main.py --vertices 5 --connectivity 0.7 --seed 42 --output out.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

from src.graph_generator import generate_graph
from src.case_generator import (
    generate_strongly_connected_orientations,
    sample_strongly_connected_orientations,
)
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts
from src.visualizer import plot_score_correlations


HOPS = (2, 3, 4)


def analyse(
    num_vertices: int,
    connectivity: float,
    seed: int | None,
    output: str | None,
    workers: int | None,
    chunk_size: int,
    max_samples: int | None = None,
) -> None:
    graph = generate_graph(num_vertices, connectivity, seed=seed)
    print(
        f"Graph: {num_vertices} vertices, {graph.number_of_edges()} edges "
        f"(connectivity={connectivity})"
    )

    apsp_sums: list[float] = []
    nhop_counts: dict[int, list[int]] = {n: [] for n in HOPS}

    n_orientations = 0
    if max_samples is not None:
        orientations_iter = sample_strongly_connected_orientations(
            graph, max_samples=max_samples, seed=seed
        )
        print(f"Sampling up to {max_samples} strongly-connected orientations …")
    else:
        orientations_iter = generate_strongly_connected_orientations(
            graph, num_workers=workers, chunk_size=chunk_size
        )

    for orientation in orientations_iter:
        n_orientations += 1
        apsp_sum, counts = calculate_apsp_sum_and_nhop_neighbor_counts(
            orientation, hops=HOPS
        )
        apsp_sums.append(apsp_sum)
        for hop in HOPS:
            nhop_counts[hop].append(counts[hop])

    print(f"Strongly-connected orientations found: {n_orientations}")

    if n_orientations == 0:
        print("No strongly-connected orientations found – nothing to plot.")
        return

    title = (
        f"n={num_vertices} vertices, p={connectivity} "
        f"({graph.number_of_edges()} edges, {n_orientations} orientations)"
    )
    save_path = output or f"result_v{num_vertices}_c{connectivity}.png"
    plot_score_correlations(
        apsp_sums,
        nhop_counts,
        title=title,
        save_path=save_path,
    )
    print(f"Plot saved to: {os.path.abspath(save_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse n-hop approach on random graph orientations."
    )
    parser.add_argument(
        "--vertices", type=int, default=5,
        help="Number of vertices (default: 5)"
    )
    parser.add_argument(
        "--connectivity", type=float, default=0.6,
        help="Edge probability 0–1 (default: 0.6)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="File path to save the plot (e.g. out.png). "
             "If omitted, defaults to result_v{vertices}_c{connectivity}.png."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Thread workers for orientation generation "
             "(default: CPU core count)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2048,
        help="Orientation chunk size processed per task (default: 2048)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Use random sampling instead of exhaustive search. "
             "Yield at most this many strongly-connected orientations "
             "(constant-time regardless of graph size)."
    )
    args = parser.parse_args()
    analyse(
        args.vertices,
        args.connectivity,
        args.seed,
        args.output,
        args.workers,
        args.chunk_size,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
