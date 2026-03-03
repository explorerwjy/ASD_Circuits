#!/usr/bin/env python3
"""
Build cluster_MeanLogUMI.csv from Allen Brain Cell Atlas h5ad files.

Computes mean(log2(UMI+1)) per gene per cluster directly from raw h5ad files,
using file-centric chunked streaming to handle large clusters without OOM.

This script provides an end-to-end pipeline from raw h5ad → cluster_MeanLogUMI,
bypassing the intermediate GeneXCell CSV extraction step. For exact reproduction
of the legacy MeanLogUMI (from pre-extracted GeneXCell CSVs), use instead:
    python scripts/build_celltype_z2_matrix.py --stage1

Validation: r=0.999 vs legacy MeanLogUMI (small differences due to v3-only cell
selection vs legacy's barcode-collision behavior).

Pipeline:
  h5ad files (WMB-10Xv3) + cell metadata + ENSMUSG→Entrez mapping
    → cluster_MeanLogUMI.csv (genes × clusters)

Input:
  - WMB-10Xv3 raw h5ad files (13 files, ~90 GB total)
  - cell_metadata_with_cluster_annotation.csv (cluster assignments)
  - ENSMUSG2HumanEntrez.json (ENSMUSG → human Entrez ID mapping)
  - Cluster annotation Excel (41586_2023_6812_MOESM8_ESM.xlsx)

Output:
  - cluster_MeanLogUMI.csv (17,938 genes × 5,312 clusters)
  - cluster_MeanLogUMI.parquet (same, compressed)
  - ABC_LogUMI.Match.10xV3.parquet (expression quantiles for matching)

Usage:
  python scripts/build_cluster_mean_log_umi.py \\
      --h5ad-dir /mnt/data0/AllenMouseSC/abc_download_root/expression_matrices/WMB-10Xv3/20230630 \\
      --cell-meta /path/to/cell_metadata_with_cluster_annotation.csv \\
      --gene-map /path/to/ENSMUSG2HumanEntrez.json \\
      --output-dir dat/SC_UMI_Mats \\
      [--chunk-size 5000] [--n-workers 10] [--test N]
"""

import argparse
import json
import os

import sys
import time
import numpy as np
import pandas as pd
import anndata
from concurrent.futures import ThreadPoolExecutor


def build_ensmusg_to_entrez(gene_map_path):
    """Load ENSMUSG → human Entrez ID mapping.

    Returns a dict mapping each ENSMUSG to a list of Entrez IDs.
    Multi-mapped genes (one ENSMUSG → multiple Entrez) are preserved
    to match the legacy pipeline behavior (17,938 output genes).
    """
    with open(gene_map_path) as f:
        mapping = json.load(f)
    return {k: v for k, v in mapping.items() if v}


def _process_chunk(log2_chunk, chunk_clusters, cluster_sums_local, cluster_counts_local):
    """Process one chunk: distribute log2 values to cluster accumulators."""
    i = 0
    while i < len(chunk_clusters):
        c = chunk_clusters[i]
        j = i + 1
        while j < len(chunk_clusters) and chunk_clusters[j] == c:
            j += 1
        if c not in cluster_sums_local:
            cluster_sums_local[c] = log2_chunk[i:j].sum(axis=0)
            cluster_counts_local[c] = j - i
        else:
            cluster_sums_local[c] += log2_chunk[i:j].sum(axis=0)
            cluster_counts_local[c] += j - i
        i = j


def process_one_h5ad(h5ad_path, feature_name, cluster_barcode_map, clusters_set,
                     n_genes, chunk_size=5000):
    """Process a single h5ad file: find matching cells and compute partial sums.

    Opens the h5ad file, finds cells belonging to any cluster, reads them in
    sorted-index chunks, and returns per-cluster sum(log2(UMI+1)) and counts.

    Returns
    -------
    (cluster_sums, cluster_counts, n_cells_found)
    """
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    barcodes = adata.obs["cell_barcode"].values
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}

    # Find cells belonging to any cluster
    idx_cluster_pairs = []
    for cluster in clusters_set:
        if cluster not in cluster_barcode_map:
            continue
        feature_barcodes = cluster_barcode_map[cluster].get(feature_name, [])
        for bc in feature_barcodes:
            if bc in barcode_to_idx:
                idx_cluster_pairs.append((barcode_to_idx[bc], cluster))

    if not idx_cluster_pairs:
        adata.file.close()
        return {}, {}, 0

    # Sort by row index for sequential disk access
    idx_cluster_pairs.sort(key=lambda x: x[0])
    all_indices = [p[0] for p in idx_cluster_pairs]
    all_clusters_for_idx = [p[1] for p in idx_cluster_pairs]

    cluster_sums = {}
    cluster_counts = {}

    for start in range(0, len(all_indices), chunk_size):
        end = min(start + chunk_size, len(all_indices))
        chunk_indices = all_indices[start:end]
        chunk_clusters = all_clusters_for_idx[start:end]

        chunk = adata.X[chunk_indices, :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()

        log2_chunk = np.log2(chunk.astype(np.float64) + 1)
        _process_chunk(log2_chunk, chunk_clusters, cluster_sums, cluster_counts)

    n_cells = len(all_indices)
    adata.file.close()
    return cluster_sums, cluster_counts, n_cells


def main():
    parser = argparse.ArgumentParser(
        description="Build cluster_MeanLogUMI from Allen h5ad files"
    )
    parser.add_argument(
        "--h5ad-dir",
        default="/mnt/data0/AllenMouseSC/abc_download_root/expression_matrices/WMB-10Xv3/20230630",
        help="Directory with WMB-10Xv3 raw h5ad files",
    )
    parser.add_argument(
        "--cell-meta",
        default=None,
        help="Cell metadata CSV with cluster assignments",
    )
    parser.add_argument(
        "--gene-map",
        default=None,
        help="ENSMUSG2HumanEntrez.json mapping file",
    )
    parser.add_argument(
        "--cluster-ann",
        default=None,
        help="Cluster annotation Excel (MOESM8)",
    )
    parser.add_argument(
        "--output-dir",
        default="dat/SC_UMI_Mats",
        help="Output directory (default: dat/SC_UMI_Mats)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Cells per chunk for streaming (default: 5000)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Number of parallel h5ad file workers (default: 4, limited by disk I/O)",
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--test", type=int, default=0,
        help="Only process first N clusters (for testing, 0=all)",
    )
    args = parser.parse_args()

    proj_dir = args.project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths for metadata
    cell_meta_path = args.cell_meta or os.path.join(
        proj_dir, "dat/Allen_Mouse_Brain_Cell_Atlas/cell_metadata_with_cluster_annotation.csv")
    if not os.path.exists(cell_meta_path):
        cell_meta_path = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/cell_metadata_with_cluster_annotation.csv"

    gene_map_path = args.gene_map or os.path.join(
        proj_dir, "dat/Allen_Mouse_Brain_Cell_Atlas/ENSMUSG2HumanEntrez.json")
    if not os.path.exists(gene_map_path):
        gene_map_path = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/ENSMUSG2HumanEntrez.json"

    cluster_ann_path = args.cluster_ann or "/mnt/data0/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx"

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(proj_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load metadata ---
    print("Loading cell metadata...")
    t_start = time.time()
    t0 = time.time()
    cell_meta = pd.read_csv(cell_meta_path,
                            usecols=["cell_barcode", "cluster", "feature_matrix_label"])
    # Keep only v3 cells (proper cell selection from metadata)
    cell_meta = cell_meta[cell_meta["feature_matrix_label"].str.contains("v3", na=False)]
    print(f"  {len(cell_meta):,} v3 cells, {cell_meta['cluster'].nunique()} clusters "
          f"({time.time()-t0:.1f}s)")

    print("Loading gene ID mapping...")
    ensmusg2entrez = build_ensmusg_to_entrez(gene_map_path)
    print(f"  {len(ensmusg2entrez)} ENSMUSG → Entrez mappings")

    print("Loading cluster annotations...")
    cluster_ann = pd.read_excel(cluster_ann_path, sheet_name="cluster_annotation",
                                index_col="cluster_id_label")
    all_clusters = cluster_ann.index.values
    print(f"  {len(all_clusters)} clusters in annotation")

    # --- Find h5ad files ---
    print("\nScanning h5ad directory...")
    h5ad_files = {}
    for fname in sorted(os.listdir(args.h5ad_dir)):
        if fname.endswith("-raw.h5ad") and "10Xv3" in fname:
            feature = fname.replace("-raw.h5ad", "")
            h5ad_files[feature] = os.path.join(args.h5ad_dir, fname)
    print(f"  Found {len(h5ad_files)} v3 h5ad files")

    # Get n_genes from first file
    first_adata = anndata.read_h5ad(next(iter(h5ad_files.values())), backed="r")
    n_genes = first_adata.shape[1]
    ensmusg_ids = first_adata.var.index.values
    first_adata.file.close()
    print(f"  {n_genes} ENSMUSG genes per file")

    # --- Build cluster → {feature: [barcodes]} index ---
    print("\nBuilding cluster-barcode index...")
    t0 = time.time()
    cluster_barcode_map = {}
    for (cluster, feature), grp in cell_meta.groupby(["cluster", "feature_matrix_label"]):
        if cluster not in cluster_barcode_map:
            cluster_barcode_map[cluster] = {}
        cluster_barcode_map[cluster][feature] = grp["cell_barcode"].tolist()
    print(f"  Indexed {len(cluster_barcode_map)} clusters ({time.time()-t0:.1f}s)")

    # --- Determine clusters to process ---
    clusters_to_process = list(cluster_barcode_map.keys())
    if args.test > 0:
        clusters_to_process = clusters_to_process[:args.test]
        print(f"\n*** TEST MODE: processing {args.test} clusters ***")

    clusters_set = set(clusters_to_process)
    total_cells = sum(
        sum(len(v) for v in cluster_barcode_map[c].values())
        for c in clusters_to_process
    )
    print(f"\nProcessing {len(clusters_to_process)} clusters ({total_cells:,} cells total)")
    print(f"Using {args.n_workers} parallel workers\n")

    # --- Process h5ad files in parallel ---
    t0 = time.time()
    # Global accumulators
    global_sums = {c: np.zeros(n_genes, dtype=np.float64) for c in clusters_set}
    global_counts = {c: 0 for c in clusters_set}

    def process_and_report(feature_name, h5ad_path):
        t_f = time.time()
        sums, counts, n_cells = process_one_h5ad(
            h5ad_path, feature_name, cluster_barcode_map, clusters_set,
            n_genes, chunk_size=args.chunk_size
        )
        elapsed = time.time() - t_f
        return feature_name, sums, counts, n_cells, elapsed

    # Use ThreadPoolExecutor (h5ad reading releases GIL during I/O)
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        for feature, path in h5ad_files.items():
            futures.append(executor.submit(process_and_report, feature, path))

        for future in futures:
            feature_name, sums, counts, n_cells, elapsed = future.result()
            if n_cells == 0:
                print(f"  {feature_name}: 0 cells (skip)")
            else:
                n_clusters = len(counts)
                print(f"  {feature_name}: {n_cells:,} cells from "
                      f"{n_clusters} clusters ({elapsed:.1f}s)")
                # Merge into global accumulators
                for c, s in sums.items():
                    global_sums[c] += s
                for c, n in counts.items():
                    global_counts[c] += n

    print(f"\nAll files processed in {time.time()-t0:.0f}s")

    # --- Build ENSMUSG → Entrez mapping ---
    entrez_ids = []
    ensmusg_col_indices = []
    seen_entrez = set()
    for i, ensmusg in enumerate(ensmusg_ids):
        if ensmusg in ensmusg2entrez:
            for entrez in ensmusg2entrez[ensmusg]:
                if entrez not in seen_entrez:
                    entrez_ids.append(entrez)
                    ensmusg_col_indices.append(i)
                    seen_entrez.add(entrez)
    col_indices = np.array(ensmusg_col_indices)
    print(f"  {len(entrez_ids)} Entrez output genes")

    # --- Compute means and build result ---
    results = {}
    for cluster in clusters_to_process:
        count = global_counts[cluster]
        if count == 0:
            continue
        mean_log2 = global_sums[cluster][col_indices] / count
        results[cluster] = pd.Series(mean_log2, index=entrez_ids, dtype=np.float64)

    # --- Assemble and save ---
    print(f"\nAssembling matrix from {len(results)} clusters...")
    df = pd.DataFrame(results)
    df.index.name = "Genes"
    print(f"  Shape: {df.shape}")

    csv_path = os.path.join(output_dir, "cluster_MeanLogUMI.csv")
    parquet_path = os.path.join(output_dir, "cluster_MeanLogUMI.parquet")
    df.to_csv(csv_path)
    df.to_parquet(parquet_path)
    print(f"  Saved: {csv_path} ({os.path.getsize(csv_path)/1e6:.0f} MB)")
    print(f"  Saved: {parquet_path} ({os.path.getsize(parquet_path)/1e6:.0f} MB)")

    # --- Compute expression quantiles ---
    print("\nComputing expression quantiles...")
    mean_exp = df.mean(axis=1)
    quantiles = pd.qcut(mean_exp, q=20, labels=False, duplicates="drop")
    quantile_df = pd.DataFrame({
        "MeanExp": mean_exp,
        "Quantile": quantiles,
    })
    quantile_path = os.path.join(output_dir, "ABC_LogUMI.Match.10xV3.parquet")
    quantile_df.to_parquet(quantile_path)
    print(f"  Saved: {quantile_path}")

    total_time = time.time() - t_start
    print(f"\nDone! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
