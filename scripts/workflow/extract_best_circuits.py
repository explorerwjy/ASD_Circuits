"""
Extract best circuit from each SA result file.

This script reads all SA runs for each bias limit and keeps only the
circuit with the highest score.

Snakemake inputs:
    input.biaslim: Bias limits file
    input.sa_results: List of SA result files

Snakemake outputs:
    output.best_circuits: File with best circuit per bias limit

Snakemake params:
    params.size: Circuit size
    params.dataset_name: Dataset name
    params.output_dir: Output directory
    params.topn: Number of top structures considered
"""

import os
import pandas as pd

# Get Snakemake variables
dataset_name = snakemake.params.dataset_name
size = int(snakemake.params.size)
output_dir = snakemake.params.output_dir
topn = snakemake.params.topn

print(f"[{dataset_name}] Extracting best circuits for size {size}...")

# Read bias limits
df_biaslim = pd.read_csv(snakemake.input.biaslim)

best_results = []
for _, row in df_biaslim.iterrows():
    bias_limit = row["bias"]
    sa_file = os.path.join(
        output_dir, dataset_name, "SA_results",
        f"size_{size}",
        f"SA..topN_{topn}-keepN_{size}-minbias_{bias_limit}.txt"
    )

    # Read all SA runs and find the best one
    best_score = -float('inf')
    best_line = None

    try:
        with open(sa_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    score = float(parts[1])
                    if score > best_score:
                        best_score = score
                        best_line = line.strip()
    except FileNotFoundError:
        print(f"[{dataset_name}] Warning: SA file not found: {sa_file}")
        continue

    if best_line:
        best_results.append((bias_limit, best_line))

# Write best results
os.makedirs(os.path.dirname(snakemake.output.best_circuits), exist_ok=True)
with open(snakemake.output.best_circuits, 'w') as f:
    f.write("# Best circuit for each bias limit\n")
    f.write("# Format: score<tab>mean_bias<tab>structure1,structure2,...\n")
    for bias_limit, line in best_results:
        f.write(f"# Bias limit: {bias_limit}\n")
        f.write(line + "\n")

print(f"[{dataset_name}] Extracted {len(best_results)} best circuits for size {size}")
