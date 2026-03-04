#!/usr/bin/env python3
"""
Build Subclass V2/V3 Expression Matrices from raw h5ad files.

Computes mean(log2(UMI+1)) per gene per subclass directly from h5ad files,
separately for V2 and V3 chemistry. Used for V2-V3 cross-platform correlation
to derive denoised (DN) gene weights.

Pipeline:
  h5ad files (WMB-10Xv2 + WMB-10Xv3) + cell metadata + ENSMUSG→Entrez mapping
    → Subclass_V2_ExpMat_UMI.csv, Subclass_V3_ExpMat_UMI.csv

Usage:
  python scripts/build_subclass_expmat.py \
      --h5ad-base /mnt/data0/AllenMouseSC/abc_download_root/expression_matrices \
      --cell-meta /path/to/cell_metadata_with_cluster_annotation.csv \
      --gene-map /path/to/ENSMUSG2HumanEntrez.json \
      --output-dir dat/ExpressionMats \
      [--chunk-size 5000] [--n-workers 4]
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
    """Load ENSMUSG → human Entrez ID mapping."""
    with open(gene_map_path) as f:
        mapping = json.load(f)
    return {k: v for k, v in mapping.items() if v}


def _process_chunk(log2_chunk, chunk_subclasses, sums, counts):
    """Process one chunk: accumulate log2 values per subclass."""
    i = 0
    while i < len(chunk_subclasses):
        sc = chunk_subclasses[i]
        j = i + 1
        while j < len(chunk_subclasses) and chunk_subclasses[j] == sc:
            j += 1
        if sc not in sums:
            sums[sc] = log2_chunk[i:j].sum(axis=0)
            counts[sc] = j - i
        else:
            sums[sc] += log2_chunk[i:j].sum(axis=0)
            counts[sc] += j - i
        i = j


def process_one_h5ad(h5ad_path, feature_name, subclass_barcode_map,
                     subclasses_set, n_genes, chunk_size=5000):
    """Process a single h5ad file for subclass-level aggregation.

    Returns (subclass_sums, subclass_counts, n_cells_found)
    """
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    barcodes = adata.obs["cell_barcode"].values
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}

    # Find cells belonging to any subclass
    idx_subclass_pairs = []
    for subclass in subclasses_set:
        if subclass not in subclass_barcode_map:
            continue
        feature_barcodes = subclass_barcode_map[subclass].get(feature_name, [])
        for bc in feature_barcodes:
            if bc in barcode_to_idx:
                idx_subclass_pairs.append((barcode_to_idx[bc], subclass))

    if not idx_subclass_pairs:
        adata.file.close()
        return {}, {}, 0

    # Sort by row index for sequential disk access
    idx_subclass_pairs.sort(key=lambda x: x[0])
    all_indices = [p[0] for p in idx_subclass_pairs]
    all_subclasses = [p[1] for p in idx_subclass_pairs]

    sums = {}
    counts = {}

    for start in range(0, len(all_indices), chunk_size):
        end = min(start + chunk_size, len(all_indices))
        chunk_indices = all_indices[start:end]
        chunk_subclasses = all_subclasses[start:end]

        chunk = adata.X[chunk_indices, :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()

        log2_chunk = np.log2(chunk.astype(np.float64) + 1)
        _process_chunk(log2_chunk, chunk_subclasses, sums, counts)

    n_cells = len(all_indices)
    adata.file.close()
    return sums, counts, n_cells


def build_subclass_barcode_map(cell_meta, version_prefix):
    """Build subclass → {feature: [barcodes]} index for one chemistry version.

    Parameters
    ----------
    cell_meta : DataFrame with columns [cell_barcode, subclass, feature_matrix_label]
    version_prefix : str, e.g. 'WMB-10Xv2' or 'WMB-10Xv3'

    Returns
    -------
    dict: subclass → {feature_name: [barcodes]}
    """
    # Filter to this version
    mask = cell_meta["feature_matrix_label"].str.startswith(version_prefix)
    version_meta = cell_meta[mask]

    subclass_map = {}
    for (subclass, feature), grp in version_meta.groupby(["subclass", "feature_matrix_label"]):
        if subclass not in subclass_map:
            subclass_map[subclass] = {}
        subclass_map[subclass][feature] = grp["cell_barcode"].tolist()
    return subclass_map


def process_version(version, h5ad_dir, subclass_barcode_map, n_genes,
                    chunk_size, n_workers):
    """Process all h5ad files for one chemistry version."""
    # Find h5ad files
    h5ad_files = {}
    for fname in sorted(os.listdir(h5ad_dir)):
        if fname.endswith("-raw.h5ad"):
            feature = fname.replace("-raw.h5ad", "")
            h5ad_files[feature] = os.path.join(h5ad_dir, fname)
    print(f"  Found {len(h5ad_files)} {version} raw h5ad files")

    subclasses_set = set(subclass_barcode_map.keys())
    total_cells = sum(
        sum(len(v) for v in subclass_barcode_map[sc].values())
        for sc in subclasses_set
    )
    print(f"  {len(subclasses_set)} subclasses, {total_cells:,} cells total")

    # Global accumulators
    global_sums = {sc: np.zeros(n_genes, dtype=np.float64) for sc in subclasses_set}
    global_counts = {sc: 0 for sc in subclasses_set}

    def process_and_report(feature_name, h5ad_path):
        t_f = time.time()
        sums, counts, n_cells = process_one_h5ad(
            h5ad_path, feature_name, subclass_barcode_map, subclasses_set,
            n_genes, chunk_size=chunk_size
        )
        elapsed = time.time() - t_f
        return feature_name, sums, counts, n_cells, elapsed

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_and_report, feature, path)
            for feature, path in h5ad_files.items()
        ]
        for future in futures:
            feature_name, sums, counts, n_cells, elapsed = future.result()
            if n_cells == 0:
                print(f"    {feature_name}: 0 cells (skip)")
            else:
                n_sc = len(counts)
                print(f"    {feature_name}: {n_cells:,} cells from {n_sc} subclasses ({elapsed:.1f}s)")
                for sc, s in sums.items():
                    global_sums[sc] += s
                for sc, n in counts.items():
                    global_counts[sc] += n

    print(f"  {version} processed in {time.time()-t0:.0f}s")
    return global_sums, global_counts


def main():
    parser = argparse.ArgumentParser(
        description="Build Subclass V2/V3 expression matrices from h5ad files"
    )
    parser.add_argument(
        "--h5ad-base",
        default="/mnt/data0/AllenMouseSC/abc_download_root/expression_matrices",
        help="Base directory containing WMB-10Xv2/ and WMB-10Xv3/ subdirs",
    )
    parser.add_argument(
        "--cell-meta", default=None,
        help="Cell metadata CSV with subclass assignments",
    )
    parser.add_argument(
        "--gene-map", default=None,
        help="ENSMUSG2HumanEntrez.json mapping file",
    )
    parser.add_argument(
        "--output-dir", default="dat/ExpressionMats",
        help="Output directory (default: dat/ExpressionMats)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Cells per chunk for streaming (default: 5000)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Number of parallel h5ad file workers (default: 4)",
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Project root directory (default: auto-detect)",
    )
    args = parser.parse_args()

    proj_dir = args.project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths
    cell_meta_path = args.cell_meta or os.path.join(
        proj_dir, "dat/Allen_Mouse_Brain_Cell_Atlas/cell_metadata_with_cluster_annotation.csv")
    if not os.path.exists(cell_meta_path):
        cell_meta_path = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/cell_metadata_with_cluster_annotation.csv"

    gene_map_path = args.gene_map or os.path.join(
        proj_dir, "dat/Allen_Mouse_Brain_Cell_Atlas/ENSMUSG2HumanEntrez.json")
    if not os.path.exists(gene_map_path):
        gene_map_path = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/ENSMUSG2HumanEntrez.json"

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(proj_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    # --- Load metadata ---
    print("Loading cell metadata...")
    t0 = time.time()
    cell_meta = pd.read_csv(
        cell_meta_path,
        usecols=["cell_barcode", "subclass", "feature_matrix_label"]
    )
    print(f"  {len(cell_meta):,} total cells, {cell_meta['subclass'].nunique()} subclasses "
          f"({time.time()-t0:.1f}s)")

    print("Loading gene ID mapping...")
    ensmusg2entrez = build_ensmusg_to_entrez(gene_map_path)
    print(f"  {len(ensmusg2entrez)} ENSMUSG → Entrez mappings")

    # --- Get gene info from a V3 file ---
    v3_dir = os.path.join(args.h5ad_base, "WMB-10Xv3/20230630")
    v2_dir = os.path.join(args.h5ad_base, "WMB-10Xv2/20230630")

    first_h5ad = None
    for fname in sorted(os.listdir(v3_dir)):
        if fname.endswith("-raw.h5ad"):
            first_h5ad = os.path.join(v3_dir, fname)
            break
    adata = anndata.read_h5ad(first_h5ad, backed="r")
    n_genes = adata.shape[1]
    ensmusg_ids = adata.var.index.values
    adata.file.close()
    print(f"  {n_genes} ENSMUSG genes per file")

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

    # --- Process V3 ---
    print("\n=== Processing V3 ===")
    v3_barcode_map = build_subclass_barcode_map(cell_meta, "WMB-10Xv3")
    v3_sums, v3_counts = process_version(
        "V3", v3_dir, v3_barcode_map, n_genes,
        args.chunk_size, args.n_workers
    )

    # --- Process V2 ---
    print("\n=== Processing V2 ===")
    v2_barcode_map = build_subclass_barcode_map(cell_meta, "WMB-10Xv2")
    v2_sums, v2_counts = process_version(
        "V2", v2_dir, v2_barcode_map, n_genes,
        args.chunk_size, args.n_workers
    )

    # --- Build result DataFrames ---
    for version, sums, counts, label in [
        ("V3", v3_sums, v3_counts, "Subclass_V3_ExpMat_UMI"),
        ("V2", v2_sums, v2_counts, "Subclass_V2_ExpMat_UMI"),
    ]:
        print(f"\nAssembling {version} matrix...")
        results = {}
        for subclass in sorted(sums.keys()):
            count = counts[subclass]
            if count == 0:
                continue
            mean_log2 = sums[subclass][col_indices] / count
            results[subclass] = pd.Series(mean_log2, index=entrez_ids, dtype=np.float64)

        df = pd.DataFrame(results)
        df.index.name = "Genes"
        print(f"  Shape: {df.shape}")

        csv_path = os.path.join(output_dir, f"{label}.csv")
        parquet_path = os.path.join(output_dir, f"{label}.parquet")
        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        print(f"  Saved: {csv_path} ({os.path.getsize(csv_path)/1e6:.0f} MB)")
        print(f"  Saved: {parquet_path} ({os.path.getsize(parquet_path)/1e6:.0f} MB)")

    total_time = time.time() - t_start
    print(f"\nDone! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
