"""
Aggregate a batch of sibling circuit search results into NPZ.

Reads pareto front CSVs for siblings [start, end), recalculates real
sibling bias, and saves NPZ with the same format as the full aggregation.

Usage:
    python scripts/aggregate_sibling_batch.py --start 0 --end 1000 \
        --output-dir results/CircuitSearch_Sibling_Summary/Mutability/size_46/batch_0
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts/workflow')
from sibling_utils import get_real_sibling_bias

# Defaults matching circuit_config_sibling_mutability.yaml
RESULT_DIR = "results/CircuitSearch_Sibling_Mutability"
PARQUET = "results/STR_ISH/null_bias/ASD_All_null_bias_sibling.parquet"
PREFIX = "ASD_Sib_Mutability_"
SIZE = 46


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    print(f"Aggregating siblings {args.start}-{args.end - 1}")

    # Load all pareto fronts for this batch
    sibling_data = []
    for sim_id in range(args.start, args.end):
        dataset_name = f"{PREFIX}{sim_id}"
        pf_path = os.path.join(
            RESULT_DIR, dataset_name, "pareto_fronts",
            f"{dataset_name}_size_{SIZE}_pareto_front.csv")

        if not os.path.exists(pf_path):
            continue

        df = pd.read_csv(pf_path)
        df['sim_id'] = sim_id

        # Recalculate real sibling bias
        real_biases = []
        for _, row in df.iterrows():
            if pd.notna(row.get('structures', np.nan)):
                structs = row['structures'].split(',')
                real_bias = get_real_sibling_bias(PARQUET, sim_id, structs)
            else:
                real_bias = np.nan
            real_biases.append(real_bias)

        df['transferred_bias'] = df['mean_bias']
        df['mean_bias'] = real_biases
        sibling_data.append(df)

    print(f"Loaded {len(sibling_data)} sibling pareto fronts")

    if len(sibling_data) == 0:
        print("WARNING: No data found for this batch!")
        return

    df_all = pd.concat(sibling_data, ignore_index=True)
    optimized = df_all[df_all['circuit_type'] == 'optimized']
    bias_limits = sorted(optimized['bias_limit'].unique())

    n_bias_limits = len(bias_limits)
    n_loaded = len(sibling_data)

    # Build profile arrays
    all_profiles = np.full((n_loaded, 2, n_bias_limits), np.nan)

    sim_ids_loaded = sorted(set(df_all['sim_id']))
    for idx, sim_id in enumerate(sim_ids_loaded):
        sib_opt = optimized[optimized['sim_id'] == sim_id]
        for j, bl in enumerate(bias_limits):
            row = sib_opt[sib_opt['bias_limit'] == bl]
            if len(row) > 0:
                all_profiles[idx, 0, j] = row['mean_bias'].iloc[0]  # real bias
                all_profiles[idx, 1, j] = row['circuit_score'].iloc[0]

    meanbias = np.nanmean(all_profiles[:, 0, :], axis=0)
    meanSI = np.nanmean(all_profiles[:, 1, :], axis=0)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "sibling_profiles.npz")
    np.savez_compressed(
        npz_path,
        meanbias=meanbias,
        meanSI=meanSI,
        topbias_sub=all_profiles,
        bias_limits=np.array(bias_limits),
    )
    print(f"Saved NPZ: {npz_path}")
    print(f"  Shape: {all_profiles.shape} ({n_loaded} siblings x 2 x {n_bias_limits} bias limits)")
    print(f"  Real meanbias range: [{meanbias.min():.4f}, {meanbias.max():.4f}]")
    print(f"  Circuit score range: [{meanSI.min():.4f}, {meanSI.max():.4f}]")

    # Also save summary CSV
    summary_path = os.path.join(args.output_dir, "sibling_summary.csv")
    summary = pd.DataFrame({
        'bias_limit': bias_limits,
        'real_bias_mean': meanbias,
        'score_mean': meanSI,
    })
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()
