#!/usr/bin/env python
"""Permutation testing for phenotype-stratified structural bias differences.

Shuffles phenotype labels (IQ, Sex, etc.) across mutations, recomputes
structural bias per group, and calculates per-structure p-values.

The main function `permutation_test_phenotype` is importable from notebooks.
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
    LoadGeneINFO, MouseSTR_AvgZ_Weighted, Mut2GeneDF, STR2Region,
)


def _single_permutation(all_muts_df, phenotype_col, threshold, exp_mat,
                          gene_symbol_to_entrez, gene_col, seed):
    """Run a single permutation: shuffle labels, compute bias diff."""
    rng = np.random.RandomState(seed)
    perm_df = all_muts_df.copy()
    perm_df[phenotype_col] = rng.permutation(perm_df[phenotype_col].values)

    if isinstance(threshold, (int, float)):
        group1 = perm_df[perm_df[phenotype_col] > threshold]
        group2 = perm_df[perm_df[phenotype_col] <= threshold]
    else:
        group1 = perm_df[perm_df[phenotype_col] == threshold]
        group2 = perm_df[perm_df[phenotype_col] != threshold]

    gw1 = Mut2GeneDF(group1, gene_col=gene_col,
                     gene_symbol_to_entrez=gene_symbol_to_entrez)
    gw2 = Mut2GeneDF(group2, gene_col=gene_col,
                     gene_symbol_to_entrez=gene_symbol_to_entrez)

    bias1 = MouseSTR_AvgZ_Weighted(exp_mat, gw1)["EFFECT"]
    bias2 = MouseSTR_AvgZ_Weighted(exp_mat, gw2)["EFFECT"]
    return bias2 - bias1


def _load_legacy_permutations(legacy_dir, group1_prefix, group2_prefix, n_perm):
    """Load legacy individual permutation CSV files."""
    diffs = []
    for i in range(1, n_perm + 1):
        g1_path = os.path.join(legacy_dir, f"{group1_prefix}.bias.perm.{i}.csv")
        g2_path = os.path.join(legacy_dir, f"{group2_prefix}.bias.perm.{i}.csv")
        if not os.path.exists(g1_path) or not os.path.exists(g2_path):
            raise FileNotFoundError(f"Legacy permutation CSVs not found at index {i}")
        g1 = pd.read_csv(g1_path, index_col="STR")
        g2 = pd.read_csv(g2_path, index_col="STR")
        diffs.append((g2["EFFECT"] - g1["EFFECT"]).rename(i - 1))
    return pd.concat(diffs, axis=1)


def permutation_test_phenotype(all_muts_df, phenotype_col, threshold,
                                exp_mat, gene_symbol_to_entrez=None,
                                n_perm=10000, n_jobs=10, seed=42,
                                cache_path=None, gene_col="HGNC",
                                legacy_perm_dir=None,
                                legacy_group1_prefix=None,
                                legacy_group2_prefix=None):
    """Permutation test for phenotype-stratified structural bias differences.

    Parameters
    ----------
    all_muts_df : DataFrame
        Mutation-level data including the phenotype column.
    phenotype_col : str
        Column name for the phenotype to test (e.g., 'IQ', 'Sex').
    threshold : int, float, or str
        Split criterion. Numeric: group1 = > threshold, group2 = <= threshold.
        String: group1 = == threshold, group2 = != threshold.
    exp_mat : DataFrame
        Expression z-score matrix (genes × structures).
    gene_symbol_to_entrez : dict or None
        Gene symbol to Entrez ID mapping.
    n_perm : int
        Number of permutations.
    n_jobs : int
        Number of parallel workers.
    seed : int
        Random seed.
    cache_path : str or None
        Path to save/load results CSV.
    gene_col : str
        Column for gene IDs ('HGNC' or 'Entrez').
    legacy_perm_dir : str or None
        Directory with legacy per-iteration CSV files.
    legacy_group1_prefix : str or None
        Prefix for group1 legacy files (e.g., 'HighIQ').
    legacy_group2_prefix : str or None
        Prefix for group2 legacy files (e.g., 'LowIQ').

    Returns
    -------
    DataFrame : index=STR, columns=[HighGroup_Bias, LowGroup_Bias, Bias_Diff, Rank_Diff, Pvalue]
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached permutation results from {cache_path}")
        return pd.read_csv(cache_path, index_col="STR")

    if gene_symbol_to_entrez is None:
        _, _, gene_symbol_to_entrez, _ = LoadGeneINFO()

    # Compute observed bias
    if isinstance(threshold, (int, float)):
        group1_df = all_muts_df[all_muts_df[phenotype_col] > threshold]
        group2_df = all_muts_df[all_muts_df[phenotype_col] <= threshold]
    else:
        group1_df = all_muts_df[all_muts_df[phenotype_col] == threshold]
        group2_df = all_muts_df[all_muts_df[phenotype_col] != threshold]

    gw1 = Mut2GeneDF(group1_df, gene_col=gene_col,
                     gene_symbol_to_entrez=gene_symbol_to_entrez)
    gw2 = Mut2GeneDF(group2_df, gene_col=gene_col,
                     gene_symbol_to_entrez=gene_symbol_to_entrez)
    obs_bias1 = MouseSTR_AvgZ_Weighted(exp_mat, gw1)
    obs_bias2 = MouseSTR_AvgZ_Weighted(exp_mat, gw2)

    obs_diff = obs_bias2["EFFECT"] - obs_bias1["EFFECT"]
    str2reg = STR2Region()

    # Try legacy permutation loading
    if (legacy_perm_dir and os.path.exists(legacy_perm_dir)
            and legacy_group1_prefix and legacy_group2_prefix):
        legacy_file = os.path.join(
            legacy_perm_dir,
            f"{legacy_group1_prefix}.bias.perm.1.csv"
        )
        if os.path.exists(legacy_file):
            print(f"  Loading {n_perm} legacy permutation files from {legacy_perm_dir}...")
            perm_diffs = _load_legacy_permutations(
                legacy_perm_dir, legacy_group1_prefix, legacy_group2_prefix, n_perm
            )
            # Compute p-values from legacy data
            perm_abs = perm_diffs.abs()
            obs_abs = obs_diff.abs()
            pvalues = ((perm_abs.T >= obs_abs).sum(axis=0) + 1) / (n_perm + 1)

            result = pd.DataFrame({
                "REG": [str2reg.get(s, "") for s in obs_diff.index],
                "HighGroup_Bias": obs_bias1["EFFECT"],
                "LowGroup_Bias": obs_bias2["EFFECT"],
                "HighGroup_Rank": obs_bias1["Rank"],
                "LowGroup_Rank": obs_bias2["Rank"],
                "Bias_Diff": obs_diff,
                "Rank_Diff": obs_bias2["Rank"] - obs_bias1["Rank"],
                "Pvalue": pvalues,
            }, index=obs_diff.index)
            result.index.name = "STR"

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                result.to_csv(cache_path)
                print(f"  Saved results to {cache_path}")
            return result

    # Compute from scratch
    print(f"  Running {n_perm} permutations ({n_jobs} jobs)...")
    perm_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_single_permutation)(
            all_muts_df, phenotype_col, threshold, exp_mat,
            gene_symbol_to_entrez, gene_col, seed + i
        )
        for i in range(n_perm)
    )
    perm_diffs = pd.concat(perm_results, axis=1)
    perm_diffs.columns = range(n_perm)

    # P-values: fraction where |perm_diff| >= |obs_diff| (two-sided)
    perm_abs = perm_diffs.abs()
    obs_abs = obs_diff.abs()
    pvalues = ((perm_abs.T >= obs_abs).sum(axis=0) + 1) / (n_perm + 1)

    result = pd.DataFrame({
        "REG": [str2reg.get(s, "") for s in obs_diff.index],
        "HighGroup_Bias": obs_bias1["EFFECT"],
        "LowGroup_Bias": obs_bias2["EFFECT"],
        "HighGroup_Rank": obs_bias1["Rank"],
        "LowGroup_Rank": obs_bias2["Rank"],
        "Bias_Diff": obs_diff,
        "Rank_Diff": obs_bias2["Rank"] - obs_bias1["Rank"],
        "Pvalue": pvalues,
    }, index=obs_diff.index)
    result.index.name = "STR"

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        result.to_csv(cache_path)
        print(f"  Saved results to {cache_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Permutation test for phenotype-stratified structural bias."
    )
    parser.add_argument("--mutations", type=str, required=True,
                        help="CSV with mutations + phenotype column")
    parser.add_argument("--phenotype_col", type=str, required=True,
                        help="Column name for phenotype (e.g., IQ, Sex)")
    parser.add_argument("--threshold", type=str, required=True,
                        help="Split threshold (numeric or string)")
    parser.add_argument("--n_perm", type=int, default=10000,
                        help="Number of permutations (default: 10000)")
    parser.add_argument("--n_jobs", type=int, default=10,
                        help="Number of parallel jobs (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path for results")
    parser.add_argument("--gene_col", type=str, default="HGNC",
                        help="Gene ID column (default: HGNC)")
    parser.add_argument("--legacy_perm_dir", type=str, default=None,
                        help="Legacy permutation CSV directory for migration")
    parser.add_argument("--legacy_group1_prefix", type=str, default=None,
                        help="Legacy group1 file prefix (e.g., HighIQ)")
    parser.add_argument("--legacy_group2_prefix", type=str, default=None,
                        help="Legacy group2 file prefix (e.g., LowIQ)")
    parser.add_argument("--expr_matrix", type=str, default=None,
                        help="Expression matrix path (default: from config)")
    args = parser.parse_args()

    # Parse threshold
    try:
        threshold = float(args.threshold)
    except ValueError:
        threshold = args.threshold

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

    result = permutation_test_phenotype(
        muts_df, args.phenotype_col, threshold,
        exp_mat, gene_symbol_to_entrez,
        n_perm=args.n_perm, n_jobs=args.n_jobs, seed=args.seed,
        cache_path=args.output, gene_col=args.gene_col,
        legacy_perm_dir=args.legacy_perm_dir,
        legacy_group1_prefix=args.legacy_group1_prefix,
        legacy_group2_prefix=args.legacy_group2_prefix,
    )

    n_sig = (result["Pvalue"] < 0.05).sum()
    print(f"\nResults: {len(result)} structures, {n_sig} significant at p<0.05")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
