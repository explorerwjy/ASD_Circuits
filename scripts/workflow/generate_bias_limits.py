"""
Generate bias limits for circuit search.

This script generates bias limits for different circuit sizes with adaptive
step sizes based on bias magnitude.

Snakemake inputs:
    None (uses params.dataset_name to look up bias file)

Snakemake outputs:
    output.raw_biaslim: Full list of bias limits
    output.filtered_biaslim: Filtered bias limits (>= threshold)

Snakemake params:
    params.size: Circuit size
    params.dataset_name: Dataset name
    params.min_bias_rank: Rank to use for minimum bias threshold
    params.dataset_config: Dictionary with single dataset configuration
"""

import sys
import os
import pandas as pd

sys.path.insert(0, 'src')
from ASD_Circuits import BiasLim

# Get Snakemake variables
dataset_name = snakemake.params.dataset_name
size = int(snakemake.params.size)
min_bias_rank = snakemake.params.min_bias_rank
dataset_config = snakemake.params.dataset_config

# Get the dataset configuration (should be a single-entry dict)
if not dataset_config:
    raise ValueError(f"No dataset config found for dataset_name '{dataset_name}'")

# Extract the config (first and only entry)
config_data = list(dataset_config.values())[0]

# Get bias file path
bias_df_path = config_data['bias_df']

# Load bias dataframe
print(f"[{dataset_name}] Loading bias data from: {bias_df_path}")
BiasDF = pd.read_csv(bias_df_path, index_col=0)

# Generate bias limits
print(f"[{dataset_name}] Generating bias limits for size {size}...")
lims = BiasLim(BiasDF, size)

# Save raw bias limits
os.makedirs(os.path.dirname(snakemake.output.raw_biaslim), exist_ok=True)
with open(snakemake.output.raw_biaslim, 'w') as fout:
    fout.write("size,bias\n")
    for s, bias in lims:
        fout.write(f"{s},{bias}\n")

# Filter bias limits to reduce computation
# Use the bias value at min_bias_rank as threshold
min_bias_threshold = BiasDF.iloc[min_bias_rank - 1]["EFFECT"]

# Read raw limits and filter
df = pd.read_csv(snakemake.output.raw_biaslim)
df_filtered = df[df["bias"] >= min_bias_threshold].reset_index(drop=True)
df_filtered.to_csv(snakemake.output.filtered_biaslim, index=False)

print(f"[{dataset_name}] Generated {len(lims)} bias limits for size {size}")
print(f"[{dataset_name}] Filtered to {len(df_filtered)} bias limits (>= {min_bias_threshold:.3f})")
