#!/usr/bin/env python3
"""
Build MERFISH structure-level UMI expression matrices.

Aggregates cluster-level expression to brain structure level using MERFISH cell
annotations. Produces 4 matrices per dataset (Allen, Zhuang):
  - STR_Cell_Mean_DF.UMI.csv  — Mean expression per cell in structure
  - STR_Vol_Mean_DF.UMI.csv   — Volume-weighted mean (cell sums / voxel count)
  - STR_NEU_Mean_DF.UMI.csv   — Neuron-only mean (excludes glia classes)
  - STR_NEU_Vol_Mean_DF.UMI.csv — Volume-weighted neuron-only mean

Input:
  - cluster_MeanLogUMI.csv — Cluster-level log2(UMI+1) expression (17938 genes × 5312 clusters)
  - MERFISH.ISH_Annot.csv  — Cell metadata with ISH_STR column
  - CCF_V3_ISH_MERFISH.csv — Structure voxel counts for volume normalization

Usage:
  python scripts/build_merfish_umi_matrices.py \\
      --output-dir dat/MERFISH \\
      [--annotation dat/MERFISH/MERFISH.ISH_Annot.csv]

  For Zhuang dataset:
  python scripts/build_merfish_umi_matrices.py \\
      --output-dir dat/MERFISH_Zhuang \\
      --annotation dat/MERFISH_Zhuang/MERFISH_Zhuang.ISH_Annot.csv
"""

import argparse
import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
import yaml

NON_NEURON_CLASS = [
    "30 Astro-Epen", "31 OPC-Oligo", "32 OEC", "33 Vascular", "34 Immune"
]


def str_sc_exp(str_name, cluster_exp, ccf_df, merfish_annot):
    """Compute structure-level expression aggregates for a single ISH structure.

    Parameters
    ----------
    str_name : str
        ISH structure name
    cluster_exp : DataFrame
        Cluster expression matrix (genes × clusters)
    ccf_df : DataFrame
        CCF structure info, indexed by CleanName, with voxel counts
    merfish_annot : DataFrame
        MERFISH cell annotations with ISH_STR, cluster, class columns

    Returns
    -------
    tuple : (str_name, cell_avg, vol_avg, neuron_avg, neuron_vol_avg)
    """
    merfish_str = merfish_annot[merfish_annot["ISH_STR"] == str_name]
    n_genes = cluster_exp.shape[0]
    str_volumes = ccf_df.loc[str_name, "total_voxel_counts (10 um)"]

    cell_sum = np.zeros(n_genes)
    neuron_sum = np.zeros(n_genes)
    neuron_count = 0

    cluster_cols = set(cluster_exp.columns.values)
    for _, row in merfish_str.iterrows():
        cluster = row["cluster"]
        cell_class = row["class"]
        if cluster in cluster_cols:
            cell_exp = np.nan_to_num(cluster_exp[cluster].values, nan=0)
            cell_sum += cell_exp
            if cell_class not in NON_NEURON_CLASS:
                neuron_sum += cell_exp
                neuron_count += 1

    n_cells = merfish_str.shape[0]
    cell_avg = cell_sum / n_cells if n_cells > 0 else cell_sum
    neuron_avg = neuron_sum / neuron_count if neuron_count > 0 else neuron_sum
    vol_avg = cell_sum / str_volumes
    neuron_vol_avg = neuron_sum / str_volumes

    return (str_name, cell_avg, vol_avg, neuron_avg, neuron_vol_avg)


def build_matrices(results, genes):
    """Assemble structure-level results into DataFrames.

    Returns
    -------
    dict : {name: DataFrame} for cell_mean, vol_mean, neu_mean, neu_vol_mean
    """
    strs, cell_avgs, vol_avgs, neu_avgs, neu_vol_avgs = [], [], [], [], []
    for (s, ca, va, na, nva) in results:
        strs.append(s)
        cell_avgs.append(ca)
        vol_avgs.append(va)
        neu_avgs.append(na)
        neu_vol_avgs.append(nva)

    return {
        "STR_Cell_Mean_DF.UMI.csv": pd.DataFrame(cell_avgs, index=strs, columns=genes).T,
        "STR_Vol_Mean_DF.UMI.csv": pd.DataFrame(vol_avgs, index=strs, columns=genes).T,
        "STR_NEU_Mean_DF.UMI.csv": pd.DataFrame(neu_avgs, index=strs, columns=genes).T,
        "STR_NEU_Vol_Mean_DF.UMI.csv": pd.DataFrame(neu_vol_avgs, index=strs, columns=genes).T,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build MERFISH structure-level UMI expression matrices"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory (e.g., dat/MERFISH or dat/MERFISH_Zhuang)"
    )
    parser.add_argument(
        "--annotation", default=None,
        help="MERFISH annotation CSV (default: dat/MERFISH/MERFISH.ISH_Annot.csv)"
    )
    parser.add_argument(
        "--n-processes", type=int, default=10,
        help="Number of parallel processes (default: 10)"
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Project root directory (default: auto-detect)"
    )
    args = parser.parse_args()

    # Resolve project directory
    proj_dir = args.project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load config
    with open(os.path.join(proj_dir, "config/config.yaml")) as f:
        config = yaml.safe_load(f)

    # Load inputs
    ccf_path = os.path.join(proj_dir, config["data_files"]["merfish_ccf_ontology"])
    ccf_df = pd.read_csv(ccf_path, index_col="CleanName")
    strs = ccf_df.index.values
    print(f"Structures: {len(strs)}")

    cluster_exp_path = os.path.join(proj_dir, config["data_files"]["cluster_mean_log_umi_csv"])
    cluster_exp = pd.read_csv(cluster_exp_path, index_col=0)
    print(f"Cluster expression: {cluster_exp.shape}")

    annot_path = args.annotation
    if annot_path is None:
        annot_path = os.path.join(proj_dir, config["data_files"]["merfish_annotation"])
    elif not os.path.isabs(annot_path):
        annot_path = os.path.join(proj_dir, annot_path)

    # Try parquet first (faster)
    parquet_path = annot_path.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        merfish = pd.read_parquet(parquet_path)
        print(f"Loaded annotation from parquet: {merfish.shape}")
    else:
        merfish = pd.read_csv(annot_path)
        print(f"Loaded annotation from CSV: {merfish.shape}")

    # Run in parallel
    print(f"\nAggregating expression for {len(strs)} structures "
          f"with {args.n_processes} processes...")
    with multiprocessing.Pool(processes=args.n_processes) as pool:
        results = pool.starmap(
            str_sc_exp,
            [(s, cluster_exp, ccf_df, merfish) for s in strs]
        )

    # Build and save matrices
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(proj_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    matrices = build_matrices(results, cluster_exp.index.values)
    for name, df in matrices.items():
        outpath = os.path.join(output_dir, name)
        df.to_csv(outpath)
        print(f"Saved: {outpath} ({df.shape}, {os.path.getsize(outpath)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
