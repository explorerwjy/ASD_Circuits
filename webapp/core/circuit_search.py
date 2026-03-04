"""
webapp/core/circuit_search.py
==============================
Simulated annealing circuit search with progress tracking.

Wraps ``src/SA_optimized.py`` Numba-optimized SA classes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

# Add src/ to path so we can import SA modules
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _find_init_state(
    bias_df: pd.DataFrame,
    size: int,
    min_bias: float,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> np.ndarray:
    """Find a valid initial circuit state satisfying the bias constraint.

    Reimplements ``FindInitState`` from ``script_circuit_search.SI.py``.
    Samples structures with probability proportional to bias^power.
    """
    strs = bias_df.index.values
    biases = bias_df["EFFECT"].values.copy()
    # Replace NaN with 0 for probability calculation
    nan_mask = np.isnan(biases)
    biases[nan_mask] = 0.0
    min_b = np.min(biases)
    pseudo = biases - min_b + 1
    power = max(min_bias * 150 - 17, 0.1)
    pseudo = np.power(pseudo, power)
    # Zero out NaN entries so they're never sampled
    pseudo[nan_mask] = 0.0
    total = np.sum(pseudo)
    if total == 0:
        probs = np.ones(len(pseudo)) / len(pseudo)
    else:
        probs = pseudo / total
    # Fix rounding
    probs[-1] = 1.0 - np.sum(probs[:-1])

    for _ in range(max_attempts):
        chosen = rng.choice(len(strs), size=size, replace=False, p=probs)
        if bias_df.iloc[chosen]["EFFECT"].mean() >= min_bias:
            init_state = np.zeros(len(strs), dtype=np.float64)
            init_state[chosen] = 1.0
            return init_state

    # Fallback: top-N by bias
    init_state = np.zeros(len(strs), dtype=np.float64)
    init_state[:size] = 1.0
    return init_state


def _run_single_sa(
    init_state: np.ndarray,
    min_bias: float,
    bias_values: np.ndarray,
    candidate_nodes: np.ndarray,
    info_mat_np: np.ndarray,
    node_to_idx: dict,
    tmax: float = 1e-2,
    tmin: float = 5e-5,
    steps: int = 50000,
) -> tuple[float, float, list[str]]:
    """Run a single SA iteration.

    Returns
    -------
    (score, mean_bias, structures_list)
    """
    from SA_optimized import CircuitSearch_SA_InfoContent_Numba

    # Build a minimal BiasDF for the SA class
    bias_df = pd.DataFrame(
        {"EFFECT": bias_values},
        index=candidate_nodes,
    )

    # Reconstruct InfoMat DataFrame for SA class
    info_keys = list(node_to_idx.keys())
    info_df = pd.DataFrame(info_mat_np, index=info_keys, columns=info_keys)

    sa = CircuitSearch_SA_InfoContent_Numba(
        bias_df, init_state, None,
        info_df,
        candidate_nodes, min_bias,
    )
    sa.copy_strategy = "deepcopy"
    sa.Tmax = tmax
    sa.Tmin = tmin
    sa.steps = steps
    sa.updates = 0

    _, _, state, e = sa.anneal()
    score = -e
    result_nodes = candidate_nodes[np.where(state == 1)[0]]
    mean_bias = bias_df.loc[result_nodes, "EFFECT"].mean()
    return score, mean_bias, list(result_nodes)


def generate_bias_limits(bias_df: pd.DataFrame, circuit_size: int, n_points: int = 20) -> list[float]:
    """Generate evenly-spaced bias limits for Pareto front.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Structure bias results with EFFECT column, sorted descending.
    circuit_size : int
        Number of structures in each circuit.
    n_points : int
        Number of Pareto points to generate.

    Returns
    -------
    list[float]
        Bias limit values from low to high.
    """
    max_mean_bias = bias_df.head(circuit_size)["EFFECT"].mean()
    min_bias = 0.0
    limits = np.linspace(min_bias, max_mean_bias * 0.95, n_points)
    return [round(float(b), 4) for b in limits]


def run_pareto_search(
    bias_df: pd.DataFrame,
    info_mat: pd.DataFrame,
    adj_mat: pd.DataFrame,
    circuit_size: int = 46,
    n_points: int = 20,
    sa_runs: int = 5,
    sa_steps: int = 50000,
    n_workers: int = 10,
    seed: int = 42,
    progress_callback=None,
) -> pd.DataFrame:
    """Run SA circuit search across multiple bias limits to build a Pareto front.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Structure bias with EFFECT, Rank, REGION columns. Sorted by EFFECT descending.
    info_mat : pd.DataFrame
        213x213 Shannon information matrix.
    adj_mat : pd.DataFrame
        213x213 adjacency/weight matrix.
    circuit_size : int
        Number of structures per circuit.
    n_points : int
        Number of bias limit points for Pareto front.
    sa_runs : int
        Independent SA runs per bias limit.
    sa_steps : int
        SA annealing steps per run.
    n_workers : int
        Multiprocessing pool size (unused — sequential for Numba compatibility).
    seed : int
        Base random seed.
    progress_callback : callable, optional
        Called with (completed_count, total_count) for progress tracking.

    Returns
    -------
    pd.DataFrame
        Pareto front with columns:
        bias_limit, circuit_score, mean_bias, n_structures, structures, circuit_type
    """
    # Use structures that exist in both bias_df and info_mat
    common = bias_df.index.intersection(info_mat.index)
    candidate_df = bias_df.loc[common].dropna(subset=["EFFECT"])
    candidate_nodes = candidate_df.index.values

    bias_limits = generate_bias_limits(bias_df, circuit_size, n_points)

    # Pre-compute shared data
    bias_values = candidate_df["EFFECT"].values.astype(np.float64)
    info_mat_np = info_mat.values.astype(np.float64)
    node_to_idx = {node: i for i, node in enumerate(info_mat.index)}

    # Build job list: (init_state, min_bias) per run
    rng = np.random.default_rng(seed)
    jobs = []
    for bl_idx, bl in enumerate(bias_limits):
        for run_idx in range(sa_runs):
            init_state = _find_init_state(candidate_df, circuit_size, bl, rng)
            jobs.append((bl_idx, init_state, bl))

    total_jobs = len(jobs)
    results_by_limit: dict[int, list] = {i: [] for i in range(len(bias_limits))}
    completed = 0

    # Sequential execution (Numba JIT + fork = potential deadlocks)
    for bl_idx, init_state, min_bias in jobs:
        result = _run_single_sa(
            init_state=init_state,
            min_bias=min_bias,
            bias_values=bias_values,
            candidate_nodes=candidate_nodes,
            info_mat_np=info_mat_np,
            node_to_idx=node_to_idx,
            steps=sa_steps,
        )
        results_by_limit[bl_idx].append(result)
        completed += 1
        if progress_callback:
            progress_callback(completed, total_jobs)

    # Extract best circuit per bias limit
    pareto_rows = []
    for bl_idx, bl in enumerate(bias_limits):
        runs = results_by_limit[bl_idx]
        if not runs:
            continue
        best = max(runs, key=lambda r: r[0])  # highest score
        score, mean_bias, structures = best
        pareto_rows.append({
            "bias_limit": bl,
            "circuit_score": score,
            "mean_bias": mean_bias,
            "n_structures": len(structures),
            "structures": ",".join(structures),
            "circuit_type": "optimized",
        })

    # Add baseline circuit (top N by bias, no optimization)
    baseline_strs = bias_df.head(circuit_size).index.values
    baseline_strs_in_info = [s for s in baseline_strs if s in info_mat.index]
    if len(baseline_strs_in_info) > 0:
        sub = info_mat.loc[baseline_strs_in_info, baseline_strs_in_info].values
        n_events = np.count_nonzero(sub)
        baseline_score = np.sum(sub) / n_events if n_events > 0 else 0.0
        baseline_mean = bias_df.loc[baseline_strs, "EFFECT"].mean()
        pareto_rows.append({
            "bias_limit": None,
            "circuit_score": baseline_score,
            "mean_bias": baseline_mean,
            "n_structures": len(baseline_strs),
            "structures": ",".join(baseline_strs),
            "circuit_type": "baseline",
        })

    return pd.DataFrame(pareto_rows)
