"""
Create consolidated Pareto front CSV file.

This script combines all best circuits into a single CSV file that's
easy to load and analyze. Also includes the baseline naive circuit
(top N structures by bias without optimization).

Snakemake inputs:
    input.best_circuits: File with best circuits
    input.biaslim: Bias limits file
    input.weight_mat: Connectivity weight matrix
    input.info_mat: Information content matrix

Snakemake outputs:
    output.pareto_csv: Pareto front CSV file

Snakemake params:
    params.size: Circuit size
    params.dataset_name: Dataset name
    params.dataset_config: Dictionary with single dataset configuration
    params.topn: Number of top structures to consider
    params.measure: Circuit scoring measure
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from ASD_Circuits import ScoreCircuit_SI_Joint, ScoreCircuit_NEdges

# Get Snakemake variables
dataset_name = snakemake.params.dataset_name
size = int(snakemake.params.size)
dataset_config = snakemake.params.dataset_config
topn = int(snakemake.params.topn)
measure = snakemake.params.measure

print(f"[{dataset_name}] Creating Pareto front CSV for size {size}...")

# Get the dataset configuration (should be a single-entry dict)
if not dataset_config:
    raise ValueError(f"No dataset config found for dataset_name '{dataset_name}'")

# Extract the config (first and only entry)
config_data = list(dataset_config.values())[0]

# Parse best circuits file
circuits_data = []
current_bias_limit = None

with open(snakemake.input.best_circuits, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("# Bias limit:"):
            current_bias_limit = float(line.split(":")[1].strip())
        elif line and not line.startswith("#"):
            parts = line.split('\t')
            if len(parts) >= 3:
                score = float(parts[0])
                mean_bias = float(parts[1])
                structures = parts[2]
                n_structures = len(structures.split(','))

                circuits_data.append({
                    'bias_limit': current_bias_limit,
                    'circuit_score': score,
                    'mean_bias': mean_bias,
                    'n_structures': n_structures,
                    'structures': structures
                })

# Load data for baseline circuit calculation
print(f"[{dataset_name}] Calculating baseline circuit (top {size} structures by bias)...")

bias_df_path = config_data['bias_df']
BiasDF = pd.read_csv(bias_df_path, index_col=0)
adj_mat = pd.read_csv(snakemake.input.weight_mat, index_col=0)
InfoMat = pd.read_csv(snakemake.input.info_mat, index_col=0)

# Calculate baseline circuit: top N structures by bias (no optimization)
baseline_structures = BiasDF.head(size).index.values
baseline_mean_bias = BiasDF.head(size)["EFFECT"].mean()

# Calculate score for baseline circuit
if measure == "SI":
    baseline_score = ScoreCircuit_SI_Joint(baseline_structures, InfoMat)
elif measure == "Connectivity":
    baseline_score = ScoreCircuit_NEdges(baseline_structures, adj_mat)
else:
    raise ValueError(f"Unknown measure: {measure}")

# Add baseline circuit to the data
baseline_entry = {
    'bias_limit': None,  # No bias limit constraint (just top N)
    'circuit_score': baseline_score,
    'mean_bias': baseline_mean_bias,
    'n_structures': len(baseline_structures),
    'structures': ','.join(baseline_structures),
    'circuit_type': 'baseline'  # Mark as baseline
}

# Mark all SA-optimized circuits
for entry in circuits_data:
    entry['circuit_type'] = 'optimized'

# Combine baseline and optimized circuits
all_circuits = [baseline_entry] + circuits_data

# Create DataFrame and save
df_pareto = pd.DataFrame(all_circuits)

# Sort optimized circuits by bias_limit, but keep baseline first
df_baseline = df_pareto[df_pareto['circuit_type'] == 'baseline']
df_optimized = df_pareto[df_pareto['circuit_type'] == 'optimized'].sort_values('bias_limit').reset_index(drop=True)
df_pareto = pd.concat([df_baseline, df_optimized], ignore_index=True)

os.makedirs(os.path.dirname(snakemake.output.pareto_csv), exist_ok=True)
df_pareto.to_csv(snakemake.output.pareto_csv, index=False)

print(f"[{dataset_name}] Baseline circuit score: {baseline_score:.6f}, bias: {baseline_mean_bias:.4f}")
print(f"[{dataset_name}] Created Pareto front CSV with {len(df_pareto)} circuits (1 baseline + {len(circuits_data)} optimized)")
print(f"[{dataset_name}] Output: {snakemake.output.pareto_csv}")
