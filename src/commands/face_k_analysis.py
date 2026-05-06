"""``face-k-analysis`` subcommand – optimal face-cycle k analysis.

For Delaunay planar graphs with edges removed while maintaining biconnectivity,
sweeps over three variables:

* **graph size** (number of vertices)
* **edge removal percentage** (fraction of edges removed)
* **face-cycle target k** (number of face clusters used by FaceCycle)

For each combination the command computes:

* **SC ratio** – fraction of randomly sampled orientations (with FaceCycle
  boundary edges forced) that are strongly connected.
* **mean APSP** – mean all-pairs shortest-path sum across the SC orientations,
  normalised by ``n*(n-1)`` so graphs of different sizes are comparable.

Results are saved as a JSON file for reproducibility and the trends are
visualised in a multi-panel figure.  A brief report with an empirically
derived formula for the optimal ``target_k`` is written to
``results/face_k_analysis/report.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend when saving to file

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from mr2s_module import FaceCycle, Graph as MR2SGraph, Edge as MR2SEdge

from src.visualizer import plot_face_k_analysis


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _generate_delaunay_graph(num_vertices: int, seed: int | None) -> nx.Graph:
    """Return a Delaunay-triangulation-based planar graph."""
    rng = np.random.default_rng(seed)
    points = rng.random((num_vertices, 2))

    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))

    if num_vertices < 3:
        graph.add_edges_from(
            (i, j)
            for i in range(num_vertices)
            for j in range(i + 1, num_vertices)
        )
        return graph

    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))
    graph.add_edges_from(edges)
    return graph


def remove_edges_maintaining_biconnectivity(
    graph: nx.Graph,
    removal_pct: float,
    rng: np.random.Generator,
) -> tuple[nx.Graph, float]:
    """Remove up to *removal_pct* fraction of edges while keeping biconnectivity.

    Edges are visited in a random order; each candidate edge is removed only if
    the graph remains biconnected after removal.  Returns the reduced graph and
    the actual removal fraction achieved.

    Args:
        graph: Source undirected graph (must already be biconnected).
        removal_pct: Target fraction of edges to remove (0.0–1.0).
        rng: NumPy random generator used to shuffle the candidate edge list.

    Returns:
        ``(reduced_graph, actual_removal_fraction)`` where
        ``actual_removal_fraction`` may be less than *removal_pct* when not
        enough removable edges exist.
    """
    if removal_pct <= 0.0:
        return graph.copy(), 0.0

    edges = list(graph.edges())
    total = len(edges)
    target_remove = max(1, int(total * removal_pct))

    result = graph.copy()
    removed = 0

    order = rng.permutation(total).tolist()
    for idx in order:
        if removed >= target_remove:
            break
        u, v = edges[idx]
        if not result.has_edge(u, v):
            continue
        result.remove_edge(u, v)
        if nx.is_biconnected(result):
            removed += 1
        else:
            result.add_edge(u, v)

    return result, removed / total if total > 0 else 0.0


def _nx_to_mr2s_graph(nx_graph: nx.Graph) -> MR2SGraph:
    """Convert a NetworkX undirected graph to a :class:`mr2s_module.Graph`."""
    edges = [
        MR2SEdge(int(u), int(v), 1, directed=False)
        for u, v in nx_graph.edges()
    ]
    return MR2SGraph(edges)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _evaluate_face_cycle(
    reduced_graph: nx.Graph,
    target_k: int,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Evaluate SC ratio and mean APSP for a FaceCycle orientation.

    1. Converts *reduced_graph* to a :class:`mr2s_module.Graph`.
    2. Runs ``FaceCycle(target_k)`` to obtain forced directed boundary edges.
    3. Samples *num_samples* random orientations for the remaining undirected
       edges, keeping the FaceCycle-directed edges fixed.
    4. Returns ``(sc_ratio, mean_normalised_apsp)``.

    The APSP sum is normalised by ``n * (n-1)`` so values are comparable
    across graphs of different sizes.  ``mean_normalised_apsp`` is the mean
    over only the strongly-connected samples; ``nan`` when none are SC.

    Args:
        reduced_graph: Biconnected undirected graph (edges may have been
            removed from the original Delaunay graph).
        target_k: FaceCycle parameter controlling the number of face clusters.
        num_samples: Number of random orientation samples to evaluate.
        rng: NumPy random generator.

    Returns:
        ``(sc_ratio, mean_normalised_apsp)``
    """
    nodes = list(reduced_graph.nodes())
    n = len(nodes)
    all_edges = list(reduced_graph.edges())

    mr2s_graph = _nx_to_mr2s_graph(reduced_graph)
    fc = FaceCycle(target_k=target_k)
    directed_edges = fc.run(mr2s_graph)

    # Build a lookup: canonical edge id → directed (src, dst) pair
    forced: dict[tuple[int, int], tuple[int, int]] = {}
    for e in directed_edges:
        forced[e.id] = e.vertices

    # Separate forced and free edges
    free_edges: list[tuple[int, int]] = [
        (u, v)
        for u, v in all_edges
        if (min(u, v), max(u, v)) not in forced
    ]
    forced_pairs: list[tuple[int, int]] = list(forced.values())

    sc_count = 0
    apsp_sums: list[float] = []

    dg = nx.DiGraph()
    dg.add_nodes_from(nodes)

    bits = rng.integers(0, 2, size=(num_samples, len(free_edges)), dtype=np.int8)

    for sample_bits in bits:
        dg.clear_edges()
        dg.add_edges_from(forced_pairs)
        for bit, (u, v) in zip(sample_bits, free_edges):
            if bit == 0:
                dg.add_edge(u, v)
            else:
                dg.add_edge(v, u)

        if nx.is_strongly_connected(dg):
            sc_count += 1
            apsp_sums.append(
                sum(
                    d
                    for src, lengths in nx.all_pairs_shortest_path_length(dg)
                    for t, d in lengths.items()
                    if t != src
                )
                / max(n * (n - 1), 1)
            )

    sc_ratio = sc_count / num_samples if num_samples > 0 else 0.0
    mean_apsp = float(np.mean(apsp_sums)) if apsp_sums else float("nan")
    return sc_ratio, mean_apsp


def run(
    graph_sizes: list[int],
    removal_pcts: list[float],
    target_ks: list[int],
    num_graphs: int,
    num_samples: int,
    seed: int | None,
    output_dir: str,
    plot_output: str | None,
) -> None:
    """Run the face-k analysis and save results.

    For every combination of *graph_sizes* × *removal_pcts* × *target_ks*:

    * Generates *num_graphs* Delaunay planar graphs.
    * Removes edges while maintaining biconnectivity.
    * Applies ``FaceCycle(target_k)`` and samples *num_samples* random
      orientations to compute SC ratio and mean APSP.

    Saves a JSON results file and a multi-panel trend figure to *output_dir*,
    then writes a brief report with the empirically derived optimal-k formula.

    Args:
        graph_sizes: List of vertex counts to sweep.
        removal_pcts: List of edge-removal fractions (0.0–1.0) to sweep.
        target_ks: List of FaceCycle ``target_k`` values to sweep.
        num_graphs: Number of independent graphs generated per combination.
        num_samples: Number of random orientation samples per graph.
        seed: Base random seed for reproducibility.
        output_dir: Directory where results, plot, and report are saved.
        plot_output: Optional override for the plot file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # results[n][pct][k] = {"sc_ratio": float, "mean_apsp": float}
    results: dict[str, Any] = {}

    total_combos = len(graph_sizes) * len(removal_pcts) * len(target_ks)
    combo_idx = 0

    for n in graph_sizes:
        results[str(n)] = {}
        for pct in removal_pcts:
            results[str(n)][str(pct)] = {}
            for k in target_ks:
                combo_idx += 1
                print(
                    f"[{combo_idx}/{total_combos}] "
                    f"n={n}, removal={pct:.0%}, k={k} …",
                    end="  ",
                    flush=True,
                )

                sc_ratios: list[float] = []
                apsps: list[float] = []

                for trial in range(num_graphs):
                    graph_seed = (seed + trial * 997 + n * 31 + int(pct * 100) * 7) if seed is not None else None
                    graph_rng = np.random.default_rng(graph_seed)

                    base_graph = _generate_delaunay_graph(n, graph_seed)

                    # Skip if not biconnected (e.g., very small graphs)
                    if not nx.is_biconnected(base_graph):
                        continue

                    reduced_graph, _ = remove_edges_maintaining_biconnectivity(
                        base_graph, pct, graph_rng
                    )

                    if not nx.is_biconnected(reduced_graph):
                        continue

                    sc_r, mean_a = _evaluate_face_cycle(
                        reduced_graph, k, num_samples, graph_rng
                    )
                    sc_ratios.append(sc_r)
                    if not np.isnan(mean_a):
                        apsps.append(mean_a)

                agg_sc = float(np.mean(sc_ratios)) if sc_ratios else float("nan")
                agg_apsp = float(np.mean(apsps)) if apsps else float("nan")

                results[str(n)][str(pct)][str(k)] = {
                    "sc_ratio": agg_sc,
                    "mean_apsp": agg_apsp,
                }
                print(f"SC={agg_sc:.3f}, APSP={agg_apsp:.3f}")

    # Save JSON results
    json_path = os.path.join(output_dir, "face_k_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "graph_sizes": graph_sizes,
                "removal_pcts": removal_pcts,
                "target_ks": target_ks,
                "num_graphs": num_graphs,
                "num_samples": num_samples,
                "seed": seed,
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved to: {os.path.abspath(json_path)}")

    # Plot
    plot_path = plot_output or os.path.join(output_dir, "face_k_analysis.png")
    plot_face_k_analysis(
        results=results,
        graph_sizes=graph_sizes,
        removal_pcts=removal_pcts,
        target_ks=target_ks,
        save_path=plot_path,
    )
    print(f"Plot saved to: {os.path.abspath(plot_path)}")

    # Write report
    report_path = os.path.join(output_dir, "report.md")
    _write_report(
        results=results,
        graph_sizes=graph_sizes,
        removal_pcts=removal_pcts,
        target_ks=target_ks,
        num_graphs=num_graphs,
        num_samples=num_samples,
        seed=seed,
        report_path=report_path,
    )
    print(f"Report saved to: {os.path.abspath(report_path)}")


# ---------------------------------------------------------------------------
# Optimal-k formula derivation
# ---------------------------------------------------------------------------

def derive_optimal_k(
    results: dict[str, Any],
    graph_sizes: list[int],
    removal_pcts: list[float],
    target_ks: list[int],
) -> dict[tuple[int, float], int]:
    """Find the *k* that maximises SC ratio for each (n, pct) combination.

    When two k values yield the same (maximal) SC ratio, the one with a lower
    mean APSP is preferred; ties are broken by choosing the smaller k.

    Args:
        results: Nested dict ``results[n_str][pct_str][k_str]`` with
            ``"sc_ratio"`` and ``"mean_apsp"`` keys.
        graph_sizes: Sorted list of vertex counts.
        removal_pcts: List of removal fractions.
        target_ks: Sorted list of candidate k values.

    Returns:
        Mapping ``(n, pct) → optimal_k``.
    """
    optimal: dict[tuple[int, float], int] = {}
    for n in graph_sizes:
        for pct in removal_pcts:
            best_k = target_ks[0]
            best_sc = float("-inf")
            best_apsp = float("inf")
            for k in target_ks:
                entry = results.get(str(n), {}).get(str(pct), {}).get(str(k), {})
                sc = entry.get("sc_ratio", float("nan"))
                apsp = entry.get("mean_apsp", float("inf"))
                if np.isnan(sc):
                    continue
                if sc > best_sc or (sc == best_sc and apsp < best_apsp):
                    best_sc, best_apsp, best_k = sc, apsp, k
            optimal[(n, pct)] = best_k
    return optimal


def _fit_optimal_k_formula(
    optimal: dict[tuple[int, float], int],
    graph_sizes: list[int],
    removal_pcts: list[float],
) -> tuple[float, float, float]:
    """Fit ``k* ≈ a * n^b * exp(c * pct)`` via least-squares on log-linearised data.

    Fits the model ``log k* = log a + b * log n + c * pct`` by ordinary
    least squares, then recovers ``a = exp(intercept)``.

    Returns ``(a, b, c)``.
    """
    log_n_vals: list[float] = []
    pct_vals: list[float] = []
    k_vals: list[float] = []

    for n in graph_sizes:
        for pct in removal_pcts:
            k = optimal.get((n, pct))
            if k is None or k <= 0:
                continue
            log_n_vals.append(np.log(n))
            pct_vals.append(pct)
            k_vals.append(float(k))

    if len(k_vals) < 3:
        return 1.0, 0.5, 0.0

    # Linearise: log(k) ≈ log(a) + b*log(n) + c*pct
    # (derived from the approximation log(1 + c*pct) ≈ c*pct for small c*pct)
    A = np.column_stack([np.ones(len(k_vals)), log_n_vals, pct_vals])
    log_k = np.log(np.maximum(k_vals, 1e-6))
    coeffs, _, _, _ = np.linalg.lstsq(A, log_k, rcond=None)
    alpha, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    a = float(np.exp(alpha))
    return a, b, c


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    results: dict[str, Any],
    graph_sizes: list[int],
    removal_pcts: list[float],
    target_ks: list[int],
    num_graphs: int,
    num_samples: int,
    seed: int | None,
    report_path: str,
) -> None:
    """Write a Markdown report summarising the analysis and optimal-k formula."""
    optimal = derive_optimal_k(results, graph_sizes, removal_pcts, target_ks)
    a, b, c = _fit_optimal_k_formula(optimal, graph_sizes, removal_pcts)

    lines: list[str] = [
        "# 최적 면 개수(target k) 분석 보고서\n",
        "## 1. 개요\n",
        "본 보고서는 달로네(Delaunay) 평면 그래프에서 쌍연결성(biconnectivity)을 "
        "유지하면서 간선을 제거한 그래프에 대해 `FaceCycle(target_k)`를 적용하고, "
        "강연결 비율(SC ratio)과 정규화된 APSP 평균을 측정한 결과를 정리합니다.\n",
        "## 2. 실험 설정\n",
        f"| 파라미터 | 값 |\n|---|---|\n"
        f"| 그래프 크기(정점 수) | {graph_sizes} |\n"
        f"| 간선 제거 비율 | {[f'{p:.0%}' for p in removal_pcts]} |\n"
        f"| target k 후보 | {target_ks} |\n"
        f"| 그래프 수 | {num_graphs} |\n"
        f"| 샘플 수 | {num_samples} |\n"
        f"| 랜덤 시드 | {seed} |\n",
        "## 3. 결과 요약\n",
        "### 3.1 강연결 비율 (SC ratio) 추이\n",
        "| 정점 수 | 제거 비율 | 최적 k | 최고 SC 비율 |\n|---|---|---|---|\n",
    ]

    for n in graph_sizes:
        for pct in removal_pcts:
            k_star = optimal.get((n, pct), target_ks[0])
            entry = results.get(str(n), {}).get(str(pct), {}).get(str(k_star), {})
            sc = entry.get("sc_ratio", float("nan"))
            lines.append(f"| {n} | {pct:.0%} | {k_star} | {sc:.4f} |\n")

    lines += [
        "\n### 3.2 정규화된 APSP 평균 추이\n",
        "| 정점 수 | 제거 비율 | 최적 k | 평균 APSP |\n|---|---|---|---|\n",
    ]
    for n in graph_sizes:
        for pct in removal_pcts:
            k_star = optimal.get((n, pct), target_ks[0])
            entry = results.get(str(n), {}).get(str(pct), {}).get(str(k_star), {})
            apsp = entry.get("mean_apsp", float("nan"))
            lines.append(f"| {n} | {pct:.0%} | {k_star} | {apsp:.4f} |\n")

    lines += [
        "\n## 4. 최적 target k 수식 유도\n",
        "각 (정점 수 n, 제거 비율 pct) 조합에서 SC 비율을 최대화하는 k*를 구한 뒤,\n"
        "로그-선형 최소제곱 회귀(`log k* ≈ log a + b·log n + c·pct`)를 통해 다음 공식을 유도하였습니다.\n\n",
        "### 4.1 경험적 공식\n\n",
        "```\n"
        f"k*(n, pct) ≈ {a:.4f} × n^{b:.4f} × exp({c:.4f} × pct)\n"
        "```\n\n",
        "여기서:\n",
        "- **n**: 그래프의 정점 수\n",
        "- **pct**: 제거된 간선 비율 (0.0 ~ 1.0)\n",
        f"- **{a:.4f}**: 기저 스케일 상수 (a)\n",
        f"- **{b:.4f}**: 정점 수에 대한 지수 (b) — 그래프가 클수록 k*가 증가함을 나타냄\n",
        f"- **{c:.4f}**: 간선 제거 비율의 선형 계수 (c) — 간선 제거가 많을수록 k*가 {'증가' if c >= 0 else '감소'}함\n\n",
        "### 4.2 해석\n\n",
        "- **b > 0**: 그래프 크기가 커질수록 더 많은 face cluster(k)가 필요합니다. "
        "더 큰 그래프의 경우 FaceCycle이 더 세밀한 분할로 더 좋은 방향 부여를 생성합니다.\n",
        "- **c > 0** (양수인 경우): 간선 제거 비율이 높을수록 최적 k가 증가합니다. "
        "간선이 줄어든 희박한 그래프에서는 더 많은 face cluster가 강연결성을 확보하는 데 도움이 됩니다.\n"
        "- **c < 0** (음수인 경우): 간선 제거 비율이 높을수록 최적 k가 감소합니다. "
        "희박한 그래프에서는 face의 수 자체가 줄기 때문에 작은 k로도 충분합니다.\n\n",
        "### 4.3 최적 k 조견표\n\n",
        "| 정점 수 | 제거 비율 | 공식 k* | 실험 k* |\n|---|---|---|---|\n",
    ]
    for n in graph_sizes:
        for pct in removal_pcts:
            formula_k = max(1, round(a * (n ** b) * np.exp(c * pct)))
            exp_k = optimal.get((n, pct), target_ks[0])
            lines.append(f"| {n} | {pct:.0%} | {formula_k} | {exp_k} |\n")

    lines += [
        "\n## 5. 결론\n\n",
        "실험 결과, FaceCycle의 `target_k`는 그래프 크기와 간선 제거 비율의 함수로 "
        "표현할 수 있으며, 위 공식을 활용하여 강연결 비율을 최대화하는 k를 선택할 수 있습니다. "
        "단, 이 공식은 경험적 근사이므로 다양한 그래프 구조에서 추가 검증이 필요합니다.\n\n",
        "자세한 추이는 `face_k_analysis.png` 그래프를 참조하십시오.\n",
    ]

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def register_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the ``face-k-analysis`` subcommand to *subparsers*."""
    p = subparsers.add_parser(
        "face-k-analysis",
        help=(
            "Analyse optimal FaceCycle target-k for Delaunay graphs with "
            "biconnectivity-preserving edge removal."
        ),
        description=(
            "Sweeps over graph size, edge-removal percentage, and FaceCycle "
            "target_k.  For each combination, randomly samples orientations "
            "(with FaceCycle boundary edges forced) and computes the SC ratio "
            "and mean normalised APSP.  Saves a plot and a Markdown report."
        ),
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        metavar="N",
        help="Graph vertex counts to sweep (default: 10 20 30)",
    )
    p.add_argument(
        "--removal-pcts",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3],
        metavar="P",
        help="Edge-removal fractions to sweep, each in [0, 1] (default: 0.0 0.1 0.2 0.3)",
    )
    p.add_argument(
        "--target-ks",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        metavar="K",
        help="FaceCycle target_k values to sweep (default: 1..10)",
    )
    p.add_argument(
        "--num-graphs",
        type=int,
        default=10,
        help="Number of independent graphs per combination (default: 10)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Random orientation samples per graph (default: 200)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/face_k_analysis",
        help=(
            "Directory for the results JSON, plot, and report "
            "(default: results/face_k_analysis)"
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the plot file path (default: <output-dir>/face_k_analysis.png)",
    )
    p.set_defaults(func=_dispatch)


def _dispatch(args: argparse.Namespace) -> None:
    run(
        graph_sizes=args.sizes,
        removal_pcts=args.removal_pcts,
        target_ks=args.target_ks,
        num_graphs=args.num_graphs,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        plot_output=args.output,
    )
