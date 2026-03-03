#!/usr/bin/env python
"""Build cell-type Z2 expression specificity matrix from Allen Brain Cell Atlas data.

Pipeline:
  Stage 1: Aggregate per-cluster GeneXCell UMI → cluster_MeanLogUMI (genes × clusters)
  Stage 2: Z1 normalize (z-score per gene across clusters) and clip to ±3
  Stage 3: Z2 calculation using ISH expression-matched gene sets (parallelized)

Each stage saves intermediate results as parquet files.
Final output: Cluster_Z2Mat_ISHMatch.z1clip3.parquet

Usage:
  # Run full pipeline
  python scripts/build_celltype_z2_matrix.py --all

  # Run individual stages
  python scripts/build_celltype_z2_matrix.py --stage1
  python scripts/build_celltype_z2_matrix.py --stage2
  python scripts/build_celltype_z2_matrix.py --stage3

  # Compare with legacy results
  python scripts/build_celltype_z2_matrix.py --validate
"""

import argparse
import csv
import gzip
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
GENEXCELL_DIR = "/mnt/data0/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI"
CLUSTER_ANNOTATION = "/mnt/data0/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx"
ISH_MATCH_DIR = None  # Set below after parsing ProjDIR
LEGACY_Z2_CSV = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv.gz"

# Output directory (relative to project root)
OUT_DIR = None  # Set after parsing ProjDIR


def get_project_root():
    """Infer project root from script location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Stage 1: Aggregate per-cluster GeneXCell UMI to cluster-level mean log2 UMI
# ---------------------------------------------------------------------------
def _process_one_cluster(cluster_id, genexcell_dir):
    """Read one cluster's GeneXCell CSV with streaming csv.reader.

    Reads one gene (row) at a time to avoid loading the full matrix into memory.
    Computes mean(log2(UMI+1)) and sum(log2(UMI+1)) per gene, plus cell depths.
    """
    cluster_clean = re.sub(r'\W+', '_', cluster_id)
    # Try .csv.gz first, then .csv
    for ext, opener in [(".GeneXCell.csv.gz", lambda p: gzip.open(p, 'rt')),
                        (".GeneXCell.csv", lambda p: open(p, 'r'))]:
        fpath = os.path.join(genexcell_dir, cluster_clean + ext)
        if os.path.exists(fpath):
            gene_indices = []
            gene_means = []
            gene_totals = []
            cell_depth = None
            n_cells = 0
            with opener(fpath) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    gene_id = int(row[0])
                    values = np.array([float(x) for x in row[1:]])
                    log2vals = np.log2(values + 1)
                    gene_indices.append(gene_id)
                    gene_means.append(log2vals.mean())
                    gene_totals.append(log2vals.sum())
                    if cell_depth is None:
                        n_cells = len(values)
                        cell_depth = log2vals.copy()
                    else:
                        cell_depth += log2vals
            mean_ser = pd.Series(data=gene_means, index=gene_indices,
                                 name=cluster_id)
            total_ser = pd.Series(data=gene_totals, index=gene_indices)
            depth_ser = pd.Series(data=cell_depth)
            return mean_ser, total_ser, depth_ser, n_cells
    print(f"  WARNING: No GeneXCell file for {cluster_id}")
    return None, None, None, 0


def stage1_mean_log_umi(genexcell_dir, cluster_ann_file, out_dir, n_jobs=10):
    """Aggregate per-cluster GeneXCell UMI to mean log2(UMI+1) matrix."""
    print("=== Stage 1: Compute cluster-level mean log2 UMI ===")
    out_path = os.path.join(out_dir, "cluster_MeanLogUMI.parquet")
    if os.path.exists(out_path):
        print(f"  Output exists: {out_path}")
        print("  Delete it to recompute. Skipping.")
        return pd.read_parquet(out_path)

    t0 = time.time()
    # Load cluster annotations to get cluster IDs
    cluster_ann = pd.read_excel(
        cluster_ann_file, sheet_name="cluster_annotation",
        index_col="cluster_id_label"
    )
    clusters = cluster_ann.index.values
    print(f"  {len(clusters)} clusters from annotation")

    # Process clusters in parallel
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one_cluster)(c, genexcell_dir) for c in clusters
    )

    # Collect results
    means = []
    total_exp = None
    total_cells = 0
    for cluster_id, (mean_vec, total_umi, cell_depth, n_cells) in zip(clusters, results):
        if mean_vec is None:
            continue
        means.append(mean_vec)
        if total_exp is None:
            total_exp = total_umi.copy()
        else:
            total_exp += total_umi
        total_cells += n_cells

    mean_log_umi = pd.concat(means, axis=1)
    mean_log_umi.index = mean_log_umi.index.astype(int)
    print(f"  Result: {mean_log_umi.shape[0]} genes × {mean_log_umi.shape[1]} clusters")

    # Save
    mean_log_umi.to_parquet(out_path)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    # Also save expression quantile table (for expression matching reference)
    if total_exp is not None:
        per_cell_exp = total_exp / total_cells
        per_cell_exp.name = "PerCellExp"
        quant_df = pd.DataFrame(per_cell_exp).sort_values("PerCellExp")
        quant_df["Rank"] = range(1, len(quant_df) + 1)
        quant_df["quantile"] = quant_df["Rank"] / len(quant_df)
        quant_path = os.path.join(out_dir, "ABC_LogUMI.Match.10xV3.parquet")
        quant_df.to_parquet(quant_path)
        print(f"  Saved expression quantiles: {quant_path}")

    print(f"  Stage 1 done in {time.time()-t0:.1f}s")
    return mean_log_umi


# ---------------------------------------------------------------------------
# Stage 2: Z1 normalization (z-score per gene across clusters) + clip
# ---------------------------------------------------------------------------
def _zscore_row(values):
    """Z-score a single row, handling NaN."""
    real = values[~np.isnan(values)]
    if len(real) == 0 or np.std(real) == 0:
        return np.zeros_like(values)
    mean = np.mean(real)
    std = np.std(real)
    return (values - mean) / std


def stage2_z1_normalize(out_dir, clip=3.0):
    """Z1-normalize the mean log UMI matrix and clip."""
    print(f"=== Stage 2: Z1 normalization (clip ±{clip}) ===")
    out_path = os.path.join(out_dir, f"cluster_Z1Mat.clip{int(clip)}.parquet")
    if os.path.exists(out_path):
        print(f"  Output exists: {out_path}")
        print("  Delete it to recompute. Skipping.")
        return pd.read_parquet(out_path)

    t0 = time.time()
    mean_path = os.path.join(out_dir, "cluster_MeanLogUMI.parquet")
    mean_log_umi = pd.read_parquet(mean_path)
    print(f"  Loaded: {mean_log_umi.shape}")

    # Z-score each gene (row) across clusters (columns)
    z1_values = np.array([_zscore_row(row) for row in mean_log_umi.values])
    z1_df = pd.DataFrame(z1_values, index=mean_log_umi.index,
                         columns=mean_log_umi.columns)

    # Clip
    z1_clipped = z1_df.clip(lower=-clip, upper=clip)

    z1_clipped.to_parquet(out_path)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  Stage 2 done in {time.time()-t0:.1f}s")
    return z1_clipped


# ---------------------------------------------------------------------------
# Stage 3: Z2 calculation using ISH expression-matched gene sets
# ---------------------------------------------------------------------------
def _load_match_genes(entrez_id, match_dir, max_genes=1000):
    """Load ISH expression-matched genes for a given gene."""
    fpath = os.path.join(match_dir, f"{entrez_id}.csv")
    if not os.path.exists(fpath):
        return None
    # Format: single column with header = first gene, remaining genes as rows
    try:
        df = pd.read_csv(fpath)
        genes = [int(df.columns[0])] + [int(x) for x in df.iloc[:, 0].values]
        return genes[:max_genes]
    except Exception:
        return None


def _z2_gene_chunk(gene_indices, z1_mat_values, z1_index, z1_columns,
                   match_dir, index_to_entrez, entrez_to_row):
    """Compute Z2 for a chunk of genes. Returns (indices, data_rows)."""
    result_indices = []
    result_rows = []
    for idx in gene_indices:
        if idx not in index_to_entrez:
            continue
        entrez = index_to_entrez[idx]
        match_genes = _load_match_genes(entrez, match_dir)
        if match_genes is None:
            continue
        # Filter to genes present in z1 matrix
        match_rows = [entrez_to_row[g] for g in match_genes
                      if g in entrez_to_row]
        if len(match_rows) < 2:
            continue

        gene_row = z1_mat_values[entrez_to_row[entrez]]
        match_vals = z1_mat_values[match_rows]  # (n_match, n_clusters)

        match_mean = np.nanmean(match_vals, axis=0)
        match_std = np.nanstd(match_vals, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            z2_row = (gene_row - match_mean) / match_std

        # NaN where gene itself was NaN or std was 0
        z2_row[~np.isfinite(z2_row)] = np.nan
        result_indices.append(entrez)
        result_rows.append(z2_row)

    return result_indices, result_rows


def stage3_z2_calculate(out_dir, match_dir, n_jobs=10, chunk_size=500):
    """Compute Z2 using ISH expression-matched gene sets."""
    print("=== Stage 3: Z2 calculation (ISH expression matching) ===")
    out_path = os.path.join(out_dir, "Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    if os.path.exists(out_path):
        print(f"  Output exists: {out_path}")
        print("  Delete it to recompute. Skipping.")
        return pd.read_parquet(out_path)

    t0 = time.time()
    z1_path = os.path.join(out_dir, "cluster_Z1Mat.clip3.parquet")
    z1_mat = pd.read_parquet(z1_path)
    print(f"  Loaded Z1 matrix: {z1_mat.shape}")

    z1_values = z1_mat.values
    z1_index = z1_mat.index.values
    z1_columns = z1_mat.columns.values

    # Build lookup tables
    index_to_entrez = {i: int(z1_index[i]) for i in range(len(z1_index))}
    entrez_to_row = {int(z1_index[i]): i for i in range(len(z1_index))}

    # Split into chunks
    n_genes = len(z1_index)
    chunks = [range(i, min(i + chunk_size, n_genes))
              for i in range(0, n_genes, chunk_size)]
    print(f"  Processing {n_genes} genes in {len(chunks)} chunks "
          f"({n_jobs} parallel jobs)...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_z2_gene_chunk)(
            list(chunk), z1_values, z1_index, z1_columns,
            match_dir, index_to_entrez, entrez_to_row
        )
        for chunk in chunks
    )

    # Combine results
    all_indices = []
    all_rows = []
    for indices, rows in results:
        all_indices.extend(indices)
        all_rows.extend(rows)

    z2_df = pd.DataFrame(
        data=np.array(all_rows),
        index=np.array(all_indices),
        columns=z1_columns
    )
    z2_df.index.name = None
    print(f"  Z2 matrix: {z2_df.shape}")

    z2_df.to_parquet(out_path)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  Stage 3 done in {time.time()-t0:.1f}s")
    return z2_df


# ---------------------------------------------------------------------------
# Validation: compare with legacy results
# ---------------------------------------------------------------------------
def validate(out_dir, legacy_csv):
    """Compare newly computed Z2 matrix with legacy CSV."""
    print("=== Validation: comparing with legacy Z2 matrix ===")
    new_path = os.path.join(out_dir, "Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    if not os.path.exists(new_path):
        print(f"  ERROR: {new_path} not found. Run --stage3 first.")
        return

    print(f"  Loading new parquet: {new_path}")
    new_df = pd.read_parquet(new_path)

    print(f"  Loading legacy CSV: {legacy_csv}")
    legacy_df = pd.read_csv(legacy_csv, index_col=0)
    legacy_df.index = legacy_df.index.astype(int)

    print(f"  New shape:    {new_df.shape}")
    print(f"  Legacy shape: {legacy_df.shape}")

    # Align on shared genes and clusters
    shared_genes = new_df.index.intersection(legacy_df.index)
    shared_cols = new_df.columns.intersection(legacy_df.columns)
    print(f"  Shared genes: {len(shared_genes)}, Shared clusters: {len(shared_cols)}")

    new_aligned = new_df.loc[shared_genes, shared_cols]
    legacy_aligned = legacy_df.loc[shared_genes, shared_cols]

    diff = (new_aligned - legacy_aligned).abs()
    max_diff = diff.max().max()
    mean_diff = diff.mean().mean()
    nan_new = new_aligned.isna().sum().sum()
    nan_legacy = legacy_aligned.isna().sum().sum()

    print(f"\n  Results:")
    print(f"    Max absolute difference: {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")
    print(f"    NaN in new: {nan_new}, NaN in legacy: {nan_legacy}")

    # Correlation
    mask = new_aligned.notna() & legacy_aligned.notna()
    new_flat = new_aligned.values[mask.values]
    legacy_flat = legacy_aligned.values[mask.values]
    corr = np.corrcoef(new_flat, legacy_flat)[0, 1]
    print(f"    Pearson correlation: {corr:.10f}")

    if max_diff < 1e-4:
        print("\n  PASS: Results match legacy within floating-point tolerance.")
    elif max_diff < 0.01:
        print("\n  WARN: Small differences detected (likely rounding).")
    else:
        print("\n  FAIL: Significant differences from legacy results.")

    # Also validate intermediate stages
    legacy_mean = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_MeanLogUMI.csv"
    new_mean = os.path.join(out_dir, "cluster_MeanLogUMI.parquet")
    if os.path.exists(new_mean) and os.path.exists(legacy_mean):
        print("\n  Validating Stage 1 (MeanLogUMI)...")
        new_m = pd.read_parquet(new_mean)
        legacy_m = pd.read_csv(legacy_mean, index_col=0)
        legacy_m.index = legacy_m.index.astype(int)
        sg = new_m.index.intersection(legacy_m.index)
        sc = new_m.columns.intersection(legacy_m.columns)
        d = (new_m.loc[sg, sc] - legacy_m.loc[sg, sc]).abs()
        print(f"    Max diff: {d.max().max():.6e}, Mean diff: {d.mean().mean():.6e}")


# ---------------------------------------------------------------------------
# Copy final parquet to project dat/BiasMatrices/
# ---------------------------------------------------------------------------
def deploy(out_dir, proj_root):
    """Copy final Z2 parquet to project BiasMatrices directory."""
    src = os.path.join(out_dir, "Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    dst = os.path.join(proj_root, "dat", "BiasMatrices",
                       "Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    if not os.path.exists(src):
        print(f"ERROR: {src} not found. Run the pipeline first.")
        return
    import shutil
    shutil.copy2(src, dst)
    print(f"Deployed: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    proj_root = get_project_root()
    default_out = os.path.join(proj_root, "results", "CellType_Z2_pipeline")
    default_match = os.path.join(
        proj_root, "dat", "genes", "ExpMatch_RootExp_uniform_kernal"
    )
    # Fallback to old project path
    if not os.path.exists(default_match):
        default_match = "/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal"

    parser = argparse.ArgumentParser(
        description="Build cell-type Z2 expression matrix from Allen Brain Cell Atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--stage1", action="store_true",
                        help="Run Stage 1: cluster mean log2 UMI")
    parser.add_argument("--stage2", action="store_true",
                        help="Run Stage 2: Z1 normalization + clip")
    parser.add_argument("--stage3", action="store_true",
                        help="Run Stage 3: Z2 calculation")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages")
    parser.add_argument("--validate", action="store_true",
                        help="Compare with legacy results")
    parser.add_argument("--deploy", action="store_true",
                        help="Copy final parquet to dat/BiasMatrices/")
    parser.add_argument("--genexcell-dir", default=GENEXCELL_DIR,
                        help=f"GeneXCell UMI directory (default: {GENEXCELL_DIR})")
    parser.add_argument("--cluster-ann", default=CLUSTER_ANNOTATION,
                        help="Cluster annotation Excel file")
    parser.add_argument("--match-dir", default=default_match,
                        help="ISH expression match directory")
    parser.add_argument("--out-dir", default=default_out,
                        help=f"Output directory (default: {default_out})")
    parser.add_argument("--legacy-csv", default=LEGACY_Z2_CSV,
                        help="Legacy Z2 CSV for validation")
    parser.add_argument("--n-jobs", type=int, default=10,
                        help="Number of parallel jobs (default: 10)")
    parser.add_argument("--clip", type=float, default=3.0,
                        help="Z1 clip value (default: 3.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all or args.stage1:
        stage1_mean_log_umi(args.genexcell_dir, args.cluster_ann,
                            args.out_dir, n_jobs=args.n_jobs)
    if args.all or args.stage2:
        stage2_z1_normalize(args.out_dir, clip=args.clip)
    if args.all or args.stage3:
        stage3_z2_calculate(args.out_dir, args.match_dir,
                            n_jobs=args.n_jobs)
    if args.validate:
        validate(args.out_dir, args.legacy_csv)
    if args.deploy:
        deploy(args.out_dir, proj_root)

    if not any([args.all, args.stage1, args.stage2, args.stage3,
                args.validate, args.deploy]):
        parser.print_help()


if __name__ == "__main__":
    main()
