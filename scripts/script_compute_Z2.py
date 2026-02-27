#!/usr/bin/env python3
"""
Compute Z2 expression-matched z-score matrix from Z1 (per-gene z-score) matrix.

Z2 controls for the relationship between baseline expression level and z-score
magnitude by comparing each gene's Z1 to expression-matched genes.

For each gene g and structure s:
    Z2(g, s) = (Z1(g, s) - mean(Z1(matched, s))) / std(Z1(matched, s))

Expression matching: for each gene, sample genes within ±5% quantile of its
root expression level (uniform kernel, with replacement).

Usage:
    python script_compute_Z2.py \\
        --z1 dat/BiasMatrices/AllenMouseBrain_Z1.parquet \\
        --exp-features dat/allen-mouse-exp/ExpMatchFeatures.csv \\
        --output dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet \\
        --seed 42 --n-jobs 10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import time


def compute_z2_chunk(args):
    """Compute Z2 for a chunk of genes. Used by multiprocessing Pool."""
    (chunk_indices, z1_array, quantiles, gene_ids, gene_to_idx,
     sample_size, interval, max_match, seed_offset, match_dir) = args

    rng = np.random.default_rng(seed_offset)
    n_structures = z1_array.shape[1]
    results = {}

    for i in chunk_indices:
        gene = gene_ids[i]

        # Get matched gene indices
        if match_dir is not None:
            # Load pre-computed match file
            match_path = Path(match_dir) / f"{gene}.csv"
            if not match_path.exists():
                results[i] = np.full(n_structures, np.nan)
                continue
            with open(match_path) as f:
                match_gene_ids = [int(line.strip()) for line in f if line.strip()]
            # Map to indices in Z1 array, keeping ALL entries (with duplicates)
            # to preserve frequency weighting from the sampling procedure.
            # This matches the legacy Z2 computation which used all 10K entries.
            all_indices = [gene_to_idx[mg] for mg in match_gene_ids
                           if mg in gene_to_idx]
        else:
            # Generate matches on the fly from quantile window
            q = quantiles[i]
            q_min = max(0, q - interval)
            q_max = min(1, q + interval)

            mask = (quantiles >= q_min) & (quantiles <= q_max)
            mask[i] = False
            interval_indices = np.where(mask)[0]

            if len(interval_indices) == 0:
                results[i] = np.full(n_structures, np.nan)
                continue

            n_sample = min(sample_size, len(interval_indices) * 10)
            sampled = rng.choice(interval_indices, size=n_sample, replace=True)
            unique_indices = list(dict.fromkeys(sampled))[:max_match]

        match_indices = all_indices if match_dir is not None else unique_indices
        if len(match_indices) < 2:
            results[i] = np.full(n_structures, np.nan)
            continue

        # Compute Z2 for all structures (vectorized)
        z1_gene = z1_array[i]
        z1_matches = z1_array[match_indices]  # (N_matched, N_structures)
        match_mean = np.nanmean(z1_matches, axis=0)
        match_std = np.nanstd(z1_matches, axis=0, ddof=0)

        z2_row = np.full(n_structures, np.nan)
        valid = match_std > 0
        z2_row[valid] = (z1_gene[valid] - match_mean[valid]) / match_std[valid]
        results[i] = z2_row

    return results


def compute_z2_parallel(z1_mat, exp_features, seed=42, sample_size=10000,
                        interval=0.05, max_match=1000, n_jobs=10,
                        match_dir=None):
    """
    Compute Z2 matrix with parallelized expression matching.

    Parameters
    ----------
    z1_mat : pd.DataFrame
        Z1 matrix (genes x structures)
    exp_features : pd.DataFrame
        Expression features with 'quantile' column, indexed by gene ID
    seed : int
        Random seed for reproducibility (ignored if match_dir is provided)
    sample_size : int
        Number of genes to sample from expression window (with replacement)
    interval : float
        Quantile interval for expression matching (±interval)
    max_match : int
        Maximum number of unique matched genes to use
    n_jobs : int
        Number of parallel processes
    match_dir : str or None
        Directory containing pre-computed match files ({entrez}.csv).
        If provided, uses these instead of generating matches on the fly.

    Returns
    -------
    pd.DataFrame
        Z2 matrix (genes x structures)
    """
    # Align genes
    common_genes = z1_mat.index[z1_mat.index.isin(exp_features.index)]
    z1_aligned = z1_mat.loc[common_genes]
    exp_aligned = exp_features.loc[common_genes]

    z1_array = z1_aligned.values
    quantiles = exp_aligned["quantile"].values
    gene_ids = z1_aligned.index.values
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    n_genes = len(gene_ids)

    if match_dir:
        print(f"Computing Z2 for {n_genes} genes x {z1_array.shape[1]} structures")
        print(f"  Using pre-computed match files from: {match_dir}")
        print(f"  max_match={max_match}, n_jobs={n_jobs}")
    else:
        print(f"Computing Z2 for {n_genes} genes x {z1_array.shape[1]} structures")
        print(f"  seed={seed}, sample_size={sample_size}, interval=±{interval}, "
              f"max_match={max_match}, n_jobs={n_jobs}")

    # Split genes into chunks for parallel processing
    chunk_size = max(1, n_genes // n_jobs)
    chunks = []
    base_rng = np.random.default_rng(seed)
    chunk_seeds = base_rng.integers(0, 2**31, size=n_jobs + 1)

    for j in range(n_jobs):
        start = j * chunk_size
        end = min(start + chunk_size, n_genes)
        if j == n_jobs - 1:
            end = n_genes
        if start >= n_genes:
            break
        chunk_indices = list(range(start, end))
        chunks.append((
            chunk_indices, z1_array, quantiles, gene_ids, gene_to_idx,
            sample_size, interval, max_match, int(chunk_seeds[j]),
            match_dir
        ))

    # Run in parallel
    t0 = time.time()
    with Pool(n_jobs) as pool:
        chunk_results = pool.map(compute_z2_chunk, chunks)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({n_genes / elapsed:.0f} genes/sec)")

    # Combine results
    z2_array = np.full_like(z1_array, np.nan)
    for result_dict in chunk_results:
        for i, z2_row in result_dict.items():
            z2_array[i] = z2_row

    Z2Mat = pd.DataFrame(z2_array, index=gene_ids, columns=z1_aligned.columns)
    Z2Mat.index.name = None

    # Add back genes from Z1 that weren't in exp_features (all NaN)
    missing_genes = z1_mat.index.difference(Z2Mat.index)
    if len(missing_genes) > 0:
        missing_df = pd.DataFrame(np.nan, index=missing_genes, columns=Z2Mat.columns)
        Z2Mat = pd.concat([Z2Mat, missing_df])
        Z2Mat = Z2Mat.loc[z1_mat.index.intersection(Z2Mat.index)]

    return Z2Mat


def main():
    parser = argparse.ArgumentParser(description="Compute Z2 expression-matched z-score matrix")
    parser.add_argument("--z1", required=True, help="Z1 matrix (parquet or csv.gz)")
    parser.add_argument("--exp-features", required=True, help="Expression features CSV")
    parser.add_argument("--output", required=True, help="Output Z2 matrix path (parquet)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Expression match sample size (default: 10000)")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="Quantile interval (default: 0.05)")
    parser.add_argument("--max-match", type=int, default=1000,
                        help="Max unique matched genes (default: 1000)")
    parser.add_argument("--n-jobs", type=int, default=10,
                        help="Number of parallel processes (default: 10)")
    parser.add_argument("--also-csv", action="store_true",
                        help="Also save as csv.gz alongside parquet")
    parser.add_argument("--match-dir", default=None,
                        help="Directory with pre-computed match files ({entrez}.csv)")
    args = parser.parse_args()

    # Load Z1
    if args.z1.endswith(".parquet"):
        Z1Mat = pd.read_parquet(args.z1)
    else:
        Z1Mat = pd.read_csv(args.z1, index_col=0)
    print(f"Z1 matrix: {Z1Mat.shape}")

    # Load expression features
    exp_features = pd.read_csv(args.exp_features, index_col="Genes")
    print(f"Expression features: {exp_features.shape[0]} genes")

    # Compute Z2
    Z2Mat = compute_z2_parallel(
        Z1Mat, exp_features,
        seed=args.seed,
        sample_size=args.sample_size,
        interval=args.interval,
        max_match=args.max_match,
        n_jobs=args.n_jobs,
        match_dir=args.match_dir
    )

    # Save
    Z2Mat.to_parquet(args.output)
    print(f"Saved Z2: {Z2Mat.shape} -> {args.output}")

    if args.also_csv:
        csv_path = args.output.replace(".parquet", ".csv.gz")
        Z2Mat.to_csv(csv_path, compression="gzip")
        print(f"Also saved: {csv_path}")

    # Stats
    n_nan = Z2Mat.isna().sum().sum()
    print(f"Z2 NaN count: {n_nan}")
    print(f"Z2 range: [{Z2Mat.min().min():.4f}, {Z2Mat.max().max():.4f}]")


if __name__ == "__main__":
    main()
