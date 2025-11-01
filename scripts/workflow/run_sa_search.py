"""
Run Simulated Annealing circuit search.

This script runs SA optimization to find optimal circuits for a specific
bias limit.

Snakemake inputs:
    input.weight_mat: Connectivity weight matrix
    input.info_mat: Information content matrix
    input.biaslim: Bias limits file

Snakemake outputs:
    output.result: SA search results file

Snakemake params:
    params.topn: Number of top structures to consider
    params.size: Circuit size
    params.bias: Minimum bias constraint
    params.dataset_name: Dataset name
    params.runtimes: Number of SA runs
    params.sa_steps: Number of SA steps per run
    params.measure: Circuit scoring measure ("SI" or "Connectivity")
    params.input_str_bias: Dataset configurations
    params.verbose: Print detailed output (default False)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

# Try to import optimized version (6-15x faster!)
try:
    from SA_optimized import (
        CircuitSearch_SA_InfoContent_Optimized as CircuitSearch_SA_InfoContent,
        CircuitSearch_SA_Connectivity_Optimized as CircuitSearch_SA_Connectivity
    )
    print("Using optimized SA implementation (6-15x faster)")
except ImportError:
    # Fallback to original version
    from ASD_Circuits import CircuitSearch_SA_InfoContent, CircuitSearch_SA_Connectivity
    print("Warning: Using original SA implementation. Install numba for 15x speedup: pip install numba")

# Get Snakemake variables
dataset_name = snakemake.params.dataset_name
topN = int(snakemake.params.topn)
keepN = int(snakemake.params.size)
minbias = float(snakemake.params.bias)
runtimes = snakemake.params.runtimes
sa_steps = snakemake.params.sa_steps
measure = snakemake.params.measure
INPUT_STR_BIAS = snakemake.params.input_str_bias
verbose = snakemake.params.verbose

# Find dataset configuration
dataset_key = None
for key, config_data in INPUT_STR_BIAS.items():
    if config_data['name'] == dataset_name:
        dataset_key = key
        break

if dataset_key is None:
    raise ValueError(f"Dataset name '{dataset_name}' not found in config")

# Get bias file path
bias_df_path = INPUT_STR_BIAS[dataset_key]['bias_df']


# Helper functions
def FindInitState(tmpDF, size, BiasLim):
    """Find initial state satisfying bias constraint"""
    STRs = tmpDF.index.values
    Biases = tmpDF["EFFECT"].values
    minBias = np.min(Biases)
    PsudoBiases = Biases - minBias + 1
    PsudoBiases = np.power(PsudoBiases, BiasLim*150-17)
    Probs = PsudoBiases/np.sum(PsudoBiases)
    Probs[-1] = 1 - np.sum(Probs[0:-1])

    rand_count = 0
    while True:
        init_STRs = np.random.choice(STRs, size=size, replace=False, p=Probs)
        avg_bias_init_STRs = tmpDF.loc[init_STRs, "EFFECT"].mean()
        if avg_bias_init_STRs >= BiasLim:
            return init_STRs, rand_count
        else:
            rand_count += 1
            if rand_count > 10000:  # Safety break
                raise ValueError(f"Cannot find initial state with bias >= {BiasLim}")


def run_CircuitOpt(BiasDF, adj_mat, InfoMat, topN, keepN, minbias, measure, sa_steps, verbose=False):
    """Run circuit optimization with SA"""
    tmpBiasDF = BiasDF.head(topN)
    CandidateNodes = tmpBiasDF.index.values
    Init_States = np.zeros(topN)
    init_STRs, rand_count = FindInitState(tmpBiasDF, keepN, minbias)

    for i, _STR in enumerate(CandidateNodes):
        if _STR in init_STRs:
            Init_States[i] = 1

    if measure == "SI":
        ins = CircuitSearch_SA_InfoContent(tmpBiasDF, Init_States, adj_mat,
                                           InfoMat, CandidateNodes, minbias=minbias)
    elif measure == "Connectivity":
        ins = CircuitSearch_SA_Connectivity(BiasDF, Init_States, adj_mat,
                                            InfoMat, CandidateNodes, minbias=minbias)

    # Use method copy for faster state copying (numpy arrays have .copy())
    ins.copy_strategy = "method"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    ins.steps = sa_steps  # User-configurable speed vs quality tradeoff

    # Control verbosity: 0 = no output, 100 = show progress updates
    ins.updates = 100 if verbose else 0

    Tmps, Energys, state, e = ins.anneal()

    res = CandidateNodes[np.where(state == 1)[0]]
    score = -e
    return res, score


# Load data
print(f"[{dataset_name}] Loading bias data...")
BiasDF = pd.read_csv(bias_df_path, index_col=0)

print(f"[{dataset_name}] Loading connectivity matrices...")
adj_mat = pd.read_csv(snakemake.input.weight_mat, index_col=0)
InfoMat = pd.read_csv(snakemake.input.info_mat, index_col=0)

print(f"[{dataset_name}] Running SA search: topN={topN}, keepN={keepN}, minbias={minbias}, measure={measure}")

# Create output directory
os.makedirs(os.path.dirname(snakemake.output.result), exist_ok=True)

# Run SA optimization multiple times
with open(snakemake.output.result, 'w') as fout:
    for i in range(runtimes):
        try:
            res, score = run_CircuitOpt(BiasDF, adj_mat, InfoMat,
                                       topN=topN, keepN=keepN,
                                       minbias=minbias, measure=measure,
                                       sa_steps=sa_steps, verbose=verbose)
            meanbias = BiasDF.loc[res, "EFFECT"].mean()
            fout.write(f"{score}\t{meanbias}\t" + ",".join(res) + "\n")
            print(f"[{dataset_name}] SA run {i+1}/{runtimes}: score={score:.6f}, bias={meanbias:.4f}")
        except Exception as e:
            print(f"[{dataset_name}] Warning: SA run {i+1} failed: {e}")
            continue

print(f"[{dataset_name}] Completed {runtimes} SA runs for size={keepN}, bias={minbias}")
