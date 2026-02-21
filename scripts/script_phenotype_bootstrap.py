#!/usr/bin/env python
"""Bootstrap resampling of mutations for phenotype-stratified structural bias.

Resamples mutations with replacement, computes gene weights via PPV+pLI,
and calculates structural bias via MouseSTR_AvgZ_Weighted.

The main function `bootstrap_phenotype_bias` is importable from notebooks.
Cache strategy (checked in order):
  1. Parquet in cache_dir → instant load
  2. Legacy CSVs in legacy_csv_dir → load, create parquet cache
  3. Neither → compute from scratch, save parquet cache
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Ensure src is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ASD_Circuits import (
    LoadGeneINFO, MouseSTR_AvgZ_Weighted, CountMut, Mut2GeneDF,
)


def _single_bootstrap(muts_df, exp_mat, gene_symbol_to_entrez, mut_type, seed):
    """Run a single bootstrap iteration and return EFFECT column."""
    rng = np.random.RandomState(seed)
    bootstrapped = muts_df.sample(frac=1, replace=True, random_state=rng)

    lgd = mut_type in ("ALL", "LGD")
    dmis = mut_type in ("ALL", "Dmis")
    gw = Mut2GeneDF(bootstrapped, LGD=lgd, Dmis=dmis,
                    gene_symbol_to_entrez=gene_symbol_to_entrez)
    bias_df = MouseSTR_AvgZ_Weighted(exp_mat, gw)
    return bias_df["EFFECT"]


def _load_legacy_csvs(legacy_csv_dir, mut_type, n_boot):
    """Load legacy individual CSV files into a single DataFrame."""
    effects = []
    for i in range(1, n_boot + 1):
        path = os.path.join(legacy_csv_dir, f"boot.bias.{mut_type}.{i}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Legacy CSV not found: {path} (expected {n_boot} files)"
            )
        df = pd.read_csv(path, index_col="STR")
        effects.append(df["EFFECT"].rename(i - 1))
    return pd.concat(effects, axis=1)


def bootstrap_phenotype_bias(muts_df, exp_mat, gene_symbol_to_entrez=None,
                              n_boot=1000, mut_types=("ALL",),
                              n_jobs=10, seed=42, cache_dir=None,
                              group_name=None, legacy_csv_dir=None):
    """Bootstrap resampling of mutations for structural bias estimation.

    Parameters
    ----------
    muts_df : DataFrame
        Mutation-level data with columns: HGNC, GeneEff, REVEL, ExACpLI.
    exp_mat : DataFrame
        Expression z-score matrix (genes × structures).
    gene_symbol_to_entrez : dict or None
        Gene symbol to Entrez ID mapping. Loaded automatically if None.
    n_boot : int
        Number of bootstrap iterations.
    mut_types : tuple of str
        Mutation types to compute: 'ALL', 'LGD', 'Dmis'.
    n_jobs : int
        Number of parallel workers.
    seed : int
        Random seed for reproducibility.
    cache_dir : str or None
        Directory for parquet cache files.
    group_name : str or None
        Group identifier for cache filenames (e.g., 'ASD.HIQ').
    legacy_csv_dir : str or None
        Directory with legacy individual CSV files for migration.

    Returns
    -------
    dict : {mut_type: DataFrame(index=structures, columns=0..n_boot-1)}
    """
    if gene_symbol_to_entrez is None:
        _, _, gene_symbol_to_entrez, _ = LoadGeneINFO()

    results = {}
    for mut_type in mut_types:
        cache_path = None
        if cache_dir and group_name:
            cache_path = os.path.join(cache_dir, f"{group_name}.{mut_type}.parquet")

        # Strategy 1: Load from parquet cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached {group_name}.{mut_type} from parquet...")
            results[mut_type] = pd.read_parquet(cache_path)
            continue

        # Strategy 2: Migrate from legacy CSVs
        if legacy_csv_dir and os.path.exists(legacy_csv_dir):
            legacy_file = os.path.join(legacy_csv_dir, f"boot.bias.{mut_type}.1.csv")
            if os.path.exists(legacy_file):
                print(f"  Migrating {group_name}.{mut_type} from {n_boot} legacy CSVs...")
                boot_df = _load_legacy_csvs(legacy_csv_dir, mut_type, n_boot)
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    boot_df.to_parquet(cache_path)
                    print(f"  Saved parquet cache: {cache_path}")
                results[mut_type] = boot_df
                continue

        # Strategy 3: Compute from scratch
        print(f"  Computing {group_name or ''}.{mut_type} bootstrap ({n_boot} iterations, {n_jobs} jobs)...")
        effects = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_single_bootstrap)(
                muts_df, exp_mat, gene_symbol_to_entrez, mut_type, seed + i
            )
            for i in range(n_boot)
        )
        boot_df = pd.concat(effects, axis=1)
        boot_df.columns = range(n_boot)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            boot_df.to_parquet(cache_path)
            print(f"  Saved parquet cache: {cache_path}")

        results[mut_type] = boot_df

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap resampling of mutations for structural bias."
    )
    parser.add_argument("--mutations", type=str, required=True,
                        help="CSV file with mutation data")
    parser.add_argument("--group", type=str, required=True,
                        help="Group name for cache files (e.g., ASD.HIQ)")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--mut_types", nargs="+", default=["ALL"],
                        help="Mutation types: ALL, LGD, Dmis (default: ALL)")
    parser.add_argument("--n_jobs", type=int, default=10,
                        help="Number of parallel jobs (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cache_dir", type=str,
                        default="results/Phenotype_bootstrap",
                        help="Output directory for parquet caches")
    parser.add_argument("--legacy_csv_dir", type=str, default=None,
                        help="Legacy CSV directory for migration")
    parser.add_argument("--expr_matrix", type=str, default=None,
                        help="Expression matrix path (default: from config)")
    args = parser.parse_args()

    # Load expression matrix
    if args.expr_matrix:
        exp_mat = pd.read_parquet(args.expr_matrix)
    else:
        import yaml
        config_path = os.path.join(os.path.dirname(_SCRIPT_DIR), "config", "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        proj_dir = config["ProjDIR"]
        expr_path = os.path.join(proj_dir, config["analysis_types"]["STR_ISH"]["expr_matrix"])
        exp_mat = pd.read_parquet(expr_path)

    _, _, gene_symbol_to_entrez, _ = LoadGeneINFO()
    muts_df = pd.read_csv(args.mutations)

    results = bootstrap_phenotype_bias(
        muts_df, exp_mat, gene_symbol_to_entrez,
        n_boot=args.n_boot, mut_types=tuple(args.mut_types),
        n_jobs=args.n_jobs, seed=args.seed,
        cache_dir=args.cache_dir, group_name=args.group,
        legacy_csv_dir=args.legacy_csv_dir,
    )

    for mt, df in results.items():
        print(f"\n{args.group}.{mt}: {df.shape[0]} structures × {df.shape[1]} bootstrap replicates")
        print(f"  Mean bias range: [{df.mean(axis=1).min():.4f}, {df.mean(axis=1).max():.4f}]")


if __name__ == "__main__":
    main()
