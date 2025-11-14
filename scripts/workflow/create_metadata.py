"""
Create analysis metadata YAML file.

This script documents all analysis parameters for reproducibility.

Snakemake inputs:
    input.pareto_fronts: List of Pareto front CSV files

Snakemake outputs:
    output.metadata: Metadata YAML file

Snakemake params:
    params.dataset_name: Dataset name
    params.dataset_config: Dictionary with single dataset configuration
    params.weight_mat: Weight matrix path
    params.info_mat: Info matrix path
    params.circuit_sizes: List of circuit sizes
    params.top_n: Top N parameter
    params.min_bias_rank: Min bias rank
    params.sa_runtimes: SA runtimes
    params.sa_steps: SA steps
    params.measure: Measure type
"""

import os
import yaml
from datetime import datetime

# Get Snakemake variables
dataset_name = snakemake.params.dataset_name
dataset_config = snakemake.params.dataset_config
WEIGHT_MAT = snakemake.params.weight_mat
INFO_MAT = snakemake.params.info_mat
CIRCUIT_SIZES = snakemake.params.circuit_sizes
TOP_N = snakemake.params.top_n
MIN_BIAS_RANK = snakemake.params.min_bias_rank
SA_RUNTIMES = snakemake.params.sa_runtimes
SA_STEPS = snakemake.params.sa_steps
MEASURE = snakemake.params.measure

print(f"[{dataset_name}] Creating metadata file...")

# Get the dataset configuration (should be a single-entry dict)
if not dataset_config:
    raise ValueError(f"No dataset config found for dataset_name '{dataset_name}'")

# Extract the config (first and only entry)
dataset_key = list(dataset_config.keys())[0]
dataset_info = list(dataset_config.values())[0]

metadata = {
    'dataset_key': dataset_key,
    'dataset_name': dataset_name,
    'description': dataset_info.get('description', ''),
    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'input_files': {
        'bias_df': dataset_info['bias_df'],
        'weight_mat': WEIGHT_MAT,
        'info_mat': INFO_MAT
    },
    'parameters': {
        'circuit_sizes': CIRCUIT_SIZES,
        'top_n': TOP_N,
        'min_bias_rank': MIN_BIAS_RANK,
        'sa_runtimes': SA_RUNTIMES,
        'sa_steps': SA_STEPS,
        'measure': MEASURE
    },
    'output_files': {
        'pareto_fronts': [f"{dataset_name}_size_{size}_pareto_front.csv"
                          for size in CIRCUIT_SIZES],
        'best_circuits': [f"size_{size}_best_circuits.txt"
                          for size in CIRCUIT_SIZES]
    }
}

os.makedirs(os.path.dirname(snakemake.output.metadata), exist_ok=True)
with open(snakemake.output.metadata, 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"[{dataset_name}] Created metadata file: {snakemake.output.metadata}")
