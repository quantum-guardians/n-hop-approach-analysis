"""``poster-results`` subcommand – MR2S poster visualization data generation.

Four categories:
1. Raw SA (SAMR2SSolver)
2. Global (QuboMR2SSolver)
3. Clustered (DnCMr2sSolver)
4. Random baseline
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from typing import Any

import numpy as np
import mr2s_module.domain.graph

# Patch missing is_empty method in mr2s_module.domain.graph.Graph class directly
if not hasattr(mr2s_module.domain.graph.Graph, 'is_empty'):
    def is_empty(self):
        return len(self.edges) == 0
    mr2s_module.domain.graph.Graph.is_empty = is_empty

from mr2s_module import SAMR2SSolver, Evaluator, estimate_required_qubits, \
  QuboMR2SSolver, SAQuboSolver, ApspSumRanker, NHopPolyGenerator, \
  FlowPolyGenerator, SmallWorldSpec, NHop
from mr2s_module.solver.dnc_mr2s_solver import DnCMr2sSolver

from src.commands.face_k_analysis import _generate_delaunay_graph, _nx_to_mr2s_graph
from src.case_generator import sample_strongly_connected_orientations
from src.cache import SimpleCache, generate_cache_key
from src.score_calculator import calculate_apsp_sum_and_nhop_neighbor_counts
from src.visualizer import plot_apsp_reduction, plot_flow_stability, plot_preprocessing_scalability

spec = SmallWorldSpec([NHop(2, 1), NHop(3, 1)])
POSTER_CACHE_VERSION = 1

def _build_sa_solver(seed: int | None = None) -> SAMR2SSolver:
    return SAMR2SSolver(
        evaluator=Evaluator(),
        random_seed=seed
    )

def _build_qubo_solver() -> QuboMR2SSolver:
    n_hop_poly = NHopPolyGenerator()
    n_hop_poly.small_world_spec = spec
    return QuboMR2SSolver(
        qubo_solver=SAQuboSolver(ApspSumRanker()),
        evaluator=Evaluator(),
        poly_generators=[n_hop_poly, FlowPolyGenerator()],
    )

def _build_dnc_qubo_solver() -> DnCMr2sSolver:
    return DnCMr2sSolver(
      mr2s_solver=_build_qubo_solver()
    )

def _estimate_physical_qubits(bqm: Any) -> float:
    """Return physical qubits for embeddable BQMs; NaN when embedding fails."""
    try:
        return float(estimate_required_qubits(bqm).num_physical_qubits)
    except RuntimeError:
        return float("nan")

def _physical_qubit_stats(values: list[float]) -> dict[str, float]:
    finite_values = [value for value in values if not np.isnan(value)]
    if not finite_values:
        return {
            "total": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
        }
    return {
        "total": float(sum(finite_values)),
        "max": float(max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "min": float(min(finite_values)),
    }

def _run_trial(task: tuple[int, int, int | None]) -> tuple[int, int, dict[str, Any]]:
    """Run all poster-result solvers for one graph trial."""
    n, trial, seed = task
    trial_seed = (seed + trial * 100 + n) if seed is not None else None
    graph = _generate_delaunay_graph(n, trial_seed)
    timings: dict[str, float] = {}

    # 1. Raw SA
    start = time.monotonic()
    solver_rsa = _build_sa_solver(seed=trial_seed)
    solver_rsa.face_cycle = None
    sol_rsa = solver_rsa.run(_nx_to_mr2s_graph(graph))
    timings["raw_sa"] = time.monotonic() - start
    res_rsa = {
        "apsp": sol_rsa.score.apsp_sum / (n * (n - 1)),
        "flow": sol_rsa.score.flow_score,
    }

    # 2. Global
    start = time.monotonic()
    solver_glb = _build_qubo_solver()
    solver_glb.face_cycle = None
    sol_glb = solver_glb.run(_nx_to_mr2s_graph(graph))
    timings["global_solve"] = time.monotonic() - start

    start = time.monotonic()
    bqm_glb = solver_glb.build_bqm(_nx_to_mr2s_graph(graph))
    phys_glb = _estimate_physical_qubits(bqm_glb)
    timings["global_embed"] = time.monotonic() - start

    res_glb = {
        "apsp": sol_glb.score.apsp_sum / (n * (n - 1)),
        "flow": sol_glb.score.flow_score,
        "qvars": len(bqm_glb.variables),
        "sg": len(bqm_glb.variables),
        "pt": phys_glb,
    }

    # 3. Clustered
    start = time.monotonic()
    solver_cls = _build_dnc_qubo_solver()
    graph_cls = _nx_to_mr2s_graph(graph)
    sub_graphs_cls = solver_cls.divide_graph(graph_cls)
    if len(sub_graphs_cls) == 1 and sub_graphs_cls[0] is graph_cls:
        sol_cls = solver_cls.mr2s_solver.run(graph_cls)
    else:
        sub_solutions = solver_cls._solve_subgraphs(sub_graphs_cls)
        merged_solution = solver_cls.merge_solutions(
            solutions=sub_solutions,
            graph=graph_cls,
        )
        solver_cls._apply_merged_directions(graph_cls, merged_solution)
        sol_cls = solver_cls.mr2s_solver.run(graph_cls)
        sol_cls.score = solver_cls.score_merged_solution(sol_cls, sub_solutions)
    timings["clustered_solve"] = time.monotonic() - start

    start = time.monotonic()
    cluster_phys = []
    cluster_qvars = []
    for sg in sub_graphs_cls:
        if not sg.is_empty():
            sg_bqm = solver_cls.mr2s_solver.build_bqm(sg)
            cluster_qvars.append(len(sg_bqm.variables))
            cluster_phys.append(_estimate_physical_qubits(sg_bqm))

    if not cluster_qvars:
        cluster_qvars = [0]
    if not cluster_phys:
        cluster_phys = [0]

    timings["clustered_embed"] = time.monotonic() - start
    phys_cls = _physical_qubit_stats(cluster_phys)

    res_cls = {
        "apsp": sol_cls.score.apsp_sum / (n * (n - 1)),
        "flow": sol_cls.score.flow_score,
        "qvars": sum(cluster_qvars),
        "sg": max(cluster_qvars),
        "phys_total": phys_cls["total"],
        "phys_max": phys_cls["max"],
        "phys_mean": phys_cls["mean"],
        "phys_min": phys_cls["min"],
    }

    # 4. Random
    start = time.monotonic()
    random_samples = list(
        sample_strongly_connected_orientations(
            graph, max_samples=10, seed=trial_seed
        )
    )
    if random_samples:
        trial_apsp = []
        trial_flow = []
        for orient in random_samples:
            apsp, _ = calculate_apsp_sum_and_nhop_neighbor_counts(orient, hops=[])
            trial_apsp.append(apsp / (n * (n - 1)))
            imbalance = sum(
                (orient.in_degree(node) - orient.out_degree(node)) ** 2
                for node in orient.nodes()
            )
            trial_flow.append(imbalance)
        res_rnd = {"apsp": np.mean(trial_apsp), "flow": np.mean(trial_flow)}
    else:
        res_rnd = {"apsp": 0, "flow": 0}
    timings["random"] = time.monotonic() - start

    return n, trial, {
        "raw_sa": res_rsa,
        "global": res_glb,
        "mr2s": res_cls,
        "random": res_rnd,
        "timings": timings,
    }

def _poster_trial_cache_key(n: int, trial: int, seed: int | None) -> str:
    """Return the stable cache key for one poster-result graph trial."""
    return generate_cache_key(
        "poster-results-trial",
        version=POSTER_CACHE_VERSION,
        n=n,
        trial=trial,
        seed=seed,
    )

def _run_trial_with_cache(
    task: tuple[int, int, int | None, str | None]
) -> tuple[int, int, dict[str, Any]]:
    """Run one poster trial, reusing the on-disk cache when configured."""
    n, trial, seed, cache_dir = task
    if cache_dir is None:
        n, trial, result = _run_trial((n, trial, seed))
        result["cache_hit"] = False
        result.setdefault("timings", {})["cache_hit"] = False
        return n, trial, result

    cache = SimpleCache(cache_dir)
    cache_key = _poster_trial_cache_key(n, trial, seed)
    cached_result = cache.get(cache_key)
    if isinstance(cached_result, dict):
        cached_result["cache_hit"] = True
        cached_result.setdefault("timings", {})["cache_hit"] = True
        return n, trial, cached_result

    n, trial, result = _run_trial((n, trial, seed))
    result["cache_hit"] = False
    result.setdefault("timings", {})["cache_hit"] = False
    cache.set(cache_key, result)
    return n, trial, result

def run(
    sizes: list[int],
    num_graphs: int,
    seed: int | None,
    output_dir: str,
    num_workers: int | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None and use_cache:
        cache_dir = os.path.join(output_dir, "poster_trial_cache")
    elif not use_cache:
        cache_dir = None

    results = {
        "sizes": sizes,
        "mr2s": {
            "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
            "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": []
        },
        "global": {
            "apsp": [], "flow": [], "qubo_vars": [], "subgraph_size": [],
            "phys_total": [], "phys_max": [], "phys_mean": [], "phys_min": []
        },
        "raw_sa": {"apsp": [], "flow": []},
        "random": {"apsp": [], "flow": []},
    }

    tasks = [(n, trial, seed, cache_dir) for n in sizes for trial in range(num_graphs)]
    total_tasks = len(tasks)
    effective_workers = (
        num_workers
        if num_workers is not None
        else min(total_tasks, max((os.cpu_count() or 2) - 1, 1))
    )

    print(
        f"Starting poster results: {len(sizes)} size(s), "
        f"{num_graphs} graph(s) each, {effective_workers} worker process(es)."
    )
    if cache_dir is not None:
        print(f"Using poster trial cache: {cache_dir}")

    trial_results: dict[int, list[dict[str, Any]]] = {n: [] for n in sizes}

    if effective_workers == 0:
        for index, task in enumerate(tasks, start=1):
            n, trial, result = _run_trial_with_cache(task)
            _print_trial_progress(index, total_tasks, n, trial, result["timings"])
            trial_results[n].append(result)
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=effective_workers) as pool:
            for index, (n, trial, result) in enumerate(
                pool.imap_unordered(_run_trial_with_cache, tasks),
                start=1,
            ):
                _print_trial_progress(index, total_tasks, n, trial, result["timings"])
                trial_results[n].append(result)

    for n in sizes:
        a_rsa, f_rsa = [], []
        a_glb, f_glb, v_glb, s_glb, p_glb = [], [], [], [], []
        a_cls, f_cls, v_cls, s_cls, pt_cls, pmax_cls, pmean_cls, pmin_cls = [], [], [], [], [], [], [], []
        a_rnd, f_rnd = [], []

        print(f"\n>>> Aggregating size n={n}")

        for result in trial_results[n]:
            res_rsa = result["raw_sa"]
            res_glb = result["global"]
            res_cls = result["mr2s"]
            res_rnd = result["random"]
            # Accumulate
            a_rsa.append(res_rsa["apsp"]); f_rsa.append(res_rsa["flow"])
            a_glb.append(res_glb["apsp"]); f_glb.append(res_glb["flow"]); v_glb.append(res_glb["qvars"]); s_glb.append(res_glb["sg"]); p_glb.append(res_glb["pt"])
            a_cls.append(res_cls["apsp"]); f_cls.append(res_cls["flow"]); v_cls.append(res_cls["qvars"]); s_cls.append(res_cls["sg"])
            pt_cls.append(res_cls["phys_total"]); pmax_cls.append(res_cls["phys_max"]); pmean_cls.append(res_cls["phys_mean"]); pmin_cls.append(res_cls["phys_min"])
            a_rnd.append(res_rnd["apsp"]); f_rnd.append(res_rnd["flow"])

        # Store averages
        results["raw_sa"]["apsp"].append(float(np.mean(a_rsa)))
        results["raw_sa"]["flow"].append(float(np.mean(f_rsa)))
        results["global"]["apsp"].append(float(np.mean(a_glb)))
        results["global"]["flow"].append(float(np.mean(f_glb)))
        results["global"]["qubo_vars"].append(float(np.mean(v_glb)))
        results["global"]["subgraph_size"].append(float(np.mean(s_glb)))
        results["global"]["phys_total"].append(float(np.mean(p_glb)))
        results["mr2s"]["apsp"].append(float(np.mean(a_cls)))
        results["mr2s"]["flow"].append(float(np.mean(f_cls)))
        results["mr2s"]["qubo_vars"].append(float(np.mean(v_cls)))
        results["mr2s"]["subgraph_size"].append(float(np.mean(s_cls)))
        results["mr2s"]["phys_total"].append(float(np.mean(pt_cls)))
        results["mr2s"]["phys_max"].append(float(np.mean(pmax_cls)))
        results["mr2s"]["phys_mean"].append(float(np.mean(pmean_cls)))
        results["mr2s"]["phys_min"].append(float(np.mean(pmin_cls)))
        results["random"]["apsp"].append(float(np.mean(a_rnd)))
        results["random"]["flow"].append(float(np.mean(f_rnd)))

    with open(os.path.join(output_dir, "poster_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_results(results, output_dir)

def _print_trial_progress(
    index: int,
    total: int,
    n: int,
    trial: int,
    timings: dict[str, float],
) -> None:
    if timings.get("cache_hit"):
        print(f"[{index}/{total}] n={n}, trial={trial}: cache hit")
        return

    print(
        f"[{index}/{total}] n={n}, trial={trial}: "
        f"Raw SA {timings['raw_sa']:.2f}s, "
        f"Global {timings['global_solve']:.2f}s + {timings['global_embed']:.2f}s, "
        f"Clustered {timings['clustered_solve']:.2f}s + {timings['clustered_embed']:.2f}s, "
        f"Random {timings['random']:.2f}s"
    )

def _plot_results(results: dict, output_dir: str):
    sizes = results["sizes"]
    plot_apsp_reduction(sizes, results["random"]["apsp"], results["raw_sa"]["apsp"], results["global"]["apsp"], results["mr2s"]["apsp"], save_path=os.path.join(output_dir, "apsp_reduction.png"))
    plot_flow_stability(sizes, results["random"]["flow"], results["raw_sa"]["flow"], results["global"]["flow"], results["mr2s"]["flow"], save_path=os.path.join(output_dir, "flow_stability.png"))
    plot_preprocessing_scalability(sizes, results["global"]["qubo_vars"], results["mr2s"]["qubo_vars"], results["global"]["subgraph_size"], results["mr2s"]["subgraph_size"], global_physical=results["global"].get("phys_total"), clustered_physical_total=results["mr2s"].get("phys_total"), clustered_physical_max=results["mr2s"].get("phys_max"), clustered_physical_mean=results["mr2s"].get("phys_mean"), clustered_physical_min=results["mr2s"].get("phys_min"), save_path=os.path.join(output_dir, "scalability.png"))

def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("poster-results", help="Generate visualization data for MR2S poster.")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.add_argument("--num-graphs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/poster")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for per-trial cache files; defaults to OUTPUT_DIR/poster_trial_cache.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the per-trial cache.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes to use; omit for auto, 0 for sequential.",
    )
    p.set_defaults(func=_dispatch)

def _dispatch(args: argparse.Namespace) -> None:
    run(
        args.sizes,
        args.num_graphs,
        args.seed,
        args.output_dir,
        args.num_workers,
        args.cache_dir,
        not args.no_cache,
    )
