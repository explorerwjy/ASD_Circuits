"""
Benchmark SA Optimization Performance

This script compares the performance of different SA implementations:
1. Original (pandas-based)
2. NumPy-optimized
3. Numba JIT-optimized

Usage:
    python scripts/benchmark_sa_optimization.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Import original and optimized versions
from ASD_Circuits import CircuitSearch_SA_InfoContent, ScoreCircuit_SI_Joint
from SA_optimized import (
    CircuitSearch_SA_InfoContent_Fast,
    CircuitSearch_SA_InfoContent_Numba,
    NUMBA_AVAILABLE
)


def load_test_data():
    """Load test data for benchmarking"""
    print("Loading test data...")

    # Load bias data
    bias_df = pd.read_csv("dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col=0)

    # Load connectivity matrices
    weight_mat = pd.read_csv("dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv", index_col=0)
    info_mat = pd.read_csv("dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv", index_col=0)

    print(f"Loaded bias data: {len(bias_df)} structures")
    print(f"Loaded info matrix: {info_mat.shape}")

    return bias_df, weight_mat, info_mat


def create_test_state(bias_df, topN=213, keepN=46, minbias=0.3):
    """Create initial state for testing"""
    print(f"\nCreating test state: topN={topN}, keepN={keepN}, minbias={minbias}")

    # Get candidate nodes
    tmp_bias_df = bias_df.head(topN)
    candidate_nodes = tmp_bias_df.index.values

    # Create initial state - start with top keepN nodes
    init_states = np.zeros(topN)
    init_states[:keepN] = 1

    # Shuffle a bit to make it more realistic
    np.random.seed(42)
    while True:
        np.random.shuffle(init_states)
        strs = candidate_nodes[init_states == 1]
        if tmp_bias_df.loc[strs, "EFFECT"].mean() >= minbias:
            break

    print(f"Initial state created: {int(init_states.sum())} nodes in circuit")
    print(f"Initial mean bias: {tmp_bias_df.loc[strs, 'EFFECT'].mean():.4f}")

    return init_states, candidate_nodes, tmp_bias_df


def benchmark_sa_version(sa_class, bias_df, weight_mat, info_mat, init_states,
                         candidate_nodes, minbias, steps=5000, name="Unknown"):
    """
    Benchmark a specific SA implementation.

    Parameters:
    -----------
    sa_class : class
        SA class to benchmark
    steps : int
        Number of SA steps to run
    name : str
        Name of the implementation for reporting

    Returns:
    --------
    dict : Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    # Create instance
    print("Creating SA instance...")
    start_init = time.time()

    ins = sa_class(
        BiasDF=bias_df,
        state=init_states.copy(),
        adjMat=weight_mat,
        InfoMat=info_mat,
        CandidateNodes=candidate_nodes,
        minbias=minbias
    )

    init_time = time.time() - start_init
    print(f"Initialization time: {init_time:.3f}s")

    # Configure SA parameters
    ins.copy_strategy = "method"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    ins.steps = steps
    ins.updates = 0  # Disable progress updates for clean benchmarking

    # Run SA
    print(f"Running SA for {steps} steps...")
    start_run = time.time()

    Tmps, Energys, state, energy = ins.anneal()

    run_time = time.time() - start_run

    # Calculate metrics
    final_strs = candidate_nodes[state == 1]
    final_bias = bias_df.loc[final_strs, "EFFECT"].mean()
    final_score = -energy

    print(f"\nResults:")
    print(f"  Total time: {run_time:.2f}s")
    print(f"  Time per step: {(run_time/steps)*1000:.3f}ms")
    print(f"  Final score: {final_score:.6f}")
    print(f"  Final bias: {final_bias:.4f}")
    print(f"  Circuit size: {int(state.sum())}")

    return {
        'name': name,
        'init_time': init_time,
        'run_time': run_time,
        'total_time': init_time + run_time,
        'time_per_step': (run_time / steps) * 1000,  # in ms
        'steps': steps,
        'final_score': final_score,
        'final_bias': final_bias,
        'circuit_size': int(state.sum())
    }


def run_benchmarks(steps=5000):
    """Run full benchmark suite"""
    print("="*60)
    print("SA OPTIMIZATION BENCHMARK")
    print("="*60)
    print(f"Steps per run: {steps}")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    # Load data
    bias_df, weight_mat, info_mat = load_test_data()

    # Create test state
    init_states, candidate_nodes, tmp_bias_df = create_test_state(bias_df)
    minbias = 0.3

    # Run benchmarks
    results = []

    # 1. Original version
    print("\n" + "="*60)
    print("TEST 1/3: Original Implementation")
    result_original = benchmark_sa_version(
        CircuitSearch_SA_InfoContent,
        tmp_bias_df, weight_mat, info_mat,
        init_states, candidate_nodes, minbias,
        steps=steps,
        name="Original (Pandas)"
    )
    results.append(result_original)

    # 2. NumPy-optimized version
    print("\n" + "="*60)
    print("TEST 2/3: NumPy-Optimized Implementation")
    result_fast = benchmark_sa_version(
        CircuitSearch_SA_InfoContent_Fast,
        tmp_bias_df, weight_mat, info_mat,
        init_states, candidate_nodes, minbias,
        steps=steps,
        name="NumPy-Optimized"
    )
    results.append(result_fast)

    # 3. Numba version (if available)
    if NUMBA_AVAILABLE:
        print("\n" + "="*60)
        print("TEST 3/3: Numba JIT-Optimized Implementation")
        result_numba = benchmark_sa_version(
            CircuitSearch_SA_InfoContent_Numba,
            tmp_bias_df, weight_mat, info_mat,
            init_states, candidate_nodes, minbias,
            steps=steps,
            name="Numba JIT-Optimized"
        )
        results.append(result_numba)
    else:
        print("\n" + "="*60)
        print("TEST 3/3: SKIPPED (Numba not available)")
        print("Install with: pip install numba")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    df_results = pd.DataFrame(results)

    print("\n" + df_results.to_string(index=False))

    # Calculate speedups
    print("\n" + "="*60)
    print("SPEEDUP COMPARISON")
    print("="*60)

    baseline_time = results[0]['run_time']

    for result in results:
        speedup = baseline_time / result['run_time']
        time_saved = baseline_time - result['run_time']
        print(f"\n{result['name']}:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time: {result['run_time']:.2f}s (saved {time_saved:.2f}s)")
        print(f"  Time per step: {result['time_per_step']:.3f}ms")

    # Extrapolate to full run
    print("\n" + "="*60)
    print("PROJECTED TIME FOR FULL RUN (50,000 steps)")
    print("="*60)

    for result in results:
        full_run_time = (result['run_time'] / steps) * 50000
        print(f"{result['name']}: {full_run_time:.1f}s ({full_run_time/60:.1f} min)")

    return df_results


if __name__ == "__main__":
    # Run with different step counts
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark SA optimizations')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of SA steps per test (default: 5000)')
    parser.add_argument('--full', action='store_true',
                        help='Run full 50,000 step benchmark (takes ~5-15 min)')

    args = parser.parse_args()

    if args.full:
        print("Running FULL benchmark with 50,000 steps...")
        print("This will take 5-15 minutes depending on your system.")
        steps = 50000
    else:
        steps = args.steps

    results = run_benchmarks(steps=steps)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
