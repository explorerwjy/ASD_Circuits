# Author: jywang explorerwjy@gmail.com
# ========================================================================================================
# script_enhanced_validation.py
# Comprehensive validation framework for the enhanced permutation analysis
# Addresses Reviewer 3, Point 1: Enhanced null model validation
# ========================================================================================================

import argparse
import sys
import os
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config to get project directory
def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
ProjDIR = config["ProjDIR"]
sys.path.insert(1, f'{ProjDIR}/src/')

# Also add the local src directory for worktree usage
script_dir = os.path.dirname(os.path.abspath(__file__))
local_src = os.path.join(os.path.dirname(script_dir), 'src')
if local_src not in sys.path:
    sys.path.insert(0, local_src)

from ASD_Circuits import (
    MouseSTR_AvgZ_Weighted,
    MouseCT_AvgZ_Weighted,
    Fil2Dict,
    load_expression_matrix_cached,
    GetPermutationP_vectorized,
    Aggregate_Gene_Weights_NDD_BGMR,
    SPARK_Gene_Weights_Extended
)
from gene_matching import (
    GenePropertyMatcher,
    MatchingConfig,
    load_gene_annotations,
    run_sensitivity_analysis
)


def run_enhanced_permutation_test(
    observed_bias: pd.DataFrame,
    null_bias_dfs: Dict[str, pd.DataFrame],
    output_prefix: str
) -> pd.DataFrame:
    """
    Run permutation tests using enhanced null distributions.

    Parameters
    ----------
    observed_bias : pd.DataFrame
        Observed bias values (from real gene weights)
    null_bias_dfs : dict
        Dictionary mapping stringency level to null bias DataFrames
    output_prefix : str
        Prefix for output files

    Returns
    -------
    pd.DataFrame
        Summary of results across stringency levels
    """
    from statsmodels.stats.multitest import multipletests

    results = {}

    for stringency, null_df in null_bias_dfs.items():
        logger.info(f"Processing {stringency} stringency null distribution")

        # Get observed values
        observed_vals = observed_bias['EFFECT'].values
        cell_type_ids = observed_bias.index.tolist()

        # Build null matrix (n_permutations x n_cell_types)
        null_cols = [c for c in null_df.columns if c != 'GeneWeight']
        null_matrix = null_df.loc[cell_type_ids, null_cols].values.T

        # Calculate p-values and z-scores
        z_scores, p_values, obs_adjs = GetPermutationP_vectorized(
            null_matrix, observed_vals, greater_than=True
        )

        # Calculate FDR-corrected q-values
        _, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[0:2]

        # Create results DataFrame
        result_df = observed_bias.copy()
        result_df['P-value'] = p_values
        result_df['Z-score'] = z_scores
        result_df['q-value'] = q_values
        result_df['-logP'] = -np.log10(p_values)
        result_df['EFFECT_adj'] = obs_adjs
        result_df['stringency'] = stringency

        # Save individual results
        result_path = f"{output_prefix}_{stringency}.csv"
        result_df.to_csv(result_path)
        logger.info(f"Saved {stringency} results to {result_path}")

        # Collect summary statistics
        n_sig_005 = (q_values < 0.05).sum()
        n_sig_010 = (q_values < 0.10).sum()

        results[stringency] = {
            'stringency': stringency,
            'n_significant_q005': n_sig_005,
            'n_significant_q010': n_sig_010,
            'mean_zscore': np.mean(z_scores),
            'median_zscore': np.median(z_scores),
            'max_zscore': np.max(z_scores),
            'mean_effect_top10': result_df.head(10)['EFFECT'].mean()
        }

    return pd.DataFrame(results).T


def compare_null_methods(
    observed_bias: pd.DataFrame,
    sibling_null: pd.DataFrame,
    random_null: pd.DataFrame,
    matched_null: pd.DataFrame,
    output_prefix: str
) -> pd.DataFrame:
    """
    Compare results across different null generation methods.

    This directly addresses the reviewer concern about null model robustness.
    """
    from statsmodels.stats.multitest import multipletests

    methods = {
        'sibling': sibling_null,
        'random': random_null,
        'matched': matched_null
    }

    all_results = {}

    for method_name, null_df in methods.items():
        if null_df is None:
            continue

        logger.info(f"Processing {method_name} null distribution")

        observed_vals = observed_bias['EFFECT'].values
        cell_type_ids = observed_bias.index.tolist()

        null_cols = [c for c in null_df.columns if c != 'GeneWeight']
        null_matrix = null_df.loc[cell_type_ids, null_cols].values.T

        z_scores, p_values, _ = GetPermutationP_vectorized(
            null_matrix, observed_vals, greater_than=True
        )
        _, q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[0:2]

        result_df = observed_bias.copy()
        result_df[f'P-value_{method_name}'] = p_values
        result_df[f'Z-score_{method_name}'] = z_scores
        result_df[f'q-value_{method_name}'] = q_values

        all_results[method_name] = result_df[[
            f'P-value_{method_name}',
            f'Z-score_{method_name}',
            f'q-value_{method_name}'
        ]]

    # Merge all results
    comparison_df = observed_bias.copy()
    for method_name, result_df in all_results.items():
        comparison_df = comparison_df.join(result_df)

    # Save comparison
    comparison_path = f"{output_prefix}_method_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"Saved method comparison to {comparison_path}")

    return comparison_df


def generate_supplementary_tables(
    results_dir: str,
    gene_sets: List[str],
    expression_matrix_path: str,
    output_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Generate supplementary tables for the manuscript.

    Creates tables showing:
    1. FDR-adjusted results with enhanced null
    2. Comparison across matching stringencies
    3. Method comparison (sibling vs random vs matched)
    """
    os.makedirs(output_dir, exist_ok=True)

    all_tables = {}

    for gene_set in gene_sets:
        logger.info(f"Processing {gene_set}")

        # Load observed bias
        observed_path = os.path.join(results_dir, gene_set, f"{gene_set}_bias.csv")
        if not os.path.exists(observed_path):
            logger.warning(f"Observed bias not found: {observed_path}")
            continue

        observed_bias = pd.read_csv(observed_path, index_col=0)

        # Try to load different null distributions
        null_files = {
            'sibling': os.path.join(results_dir, gene_set, f"{gene_set}_null_sibling.parquet"),
            'random': os.path.join(results_dir, gene_set, f"{gene_set}_null_random.parquet"),
            'matched_loose': os.path.join(results_dir, gene_set, f"{gene_set}_null_matched_loose.parquet"),
            'matched_medium': os.path.join(results_dir, gene_set, f"{gene_set}_null_matched_medium.parquet"),
            'matched_tight': os.path.join(results_dir, gene_set, f"{gene_set}_null_matched_tight.parquet"),
        }

        null_dfs = {}
        for null_name, null_path in null_files.items():
            if os.path.exists(null_path):
                null_dfs[null_name] = pd.read_parquet(null_path)

        if not null_dfs:
            logger.warning(f"No null distributions found for {gene_set}")
            continue

        # Generate comparison table
        output_prefix = os.path.join(output_dir, gene_set)
        summary = run_enhanced_permutation_test(
            observed_bias, null_dfs, output_prefix
        )

        all_tables[gene_set] = summary

    return all_tables


def create_summary_report(
    all_tables: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Create a summary report for all gene sets and stringency levels.
    """
    with open(output_path, 'w') as f:
        f.write("# Enhanced Permutation Framework Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write("This report summarizes the validation of mutation bias results\n")
        f.write("using the enhanced permutation framework that matches on:\n")
        f.write("- Gene length (CDS length)\n")
        f.write("- Conservation (phastCons)\n")
        f.write("- Expression level\n\n")

        f.write("## Results by Gene Set\n\n")

        for gene_set, summary in all_tables.items():
            f.write(f"### {gene_set}\n\n")
            f.write(summary.to_string())
            f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("- n_significant_q005: Number of regions/cell types with q < 0.05\n")
        f.write("- n_significant_q010: Number of regions/cell types with q < 0.10\n")
        f.write("- mean_zscore: Average z-score across all tests\n")
        f.write("- Robust results should show consistent significance across stringency levels\n")

    logger.info(f"Created summary report at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run enhanced permutation framework validation"
    )
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing bias results')
    parser.add_argument('--expr_matrix', required=True,
                        help='Path to expression matrix')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for validation results')
    parser.add_argument('--gene_sets', nargs='+', required=True,
                        help='Gene sets to analyze')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting enhanced permutation framework validation")

    # Generate supplementary tables
    all_tables = generate_supplementary_tables(
        results_dir=args.results_dir,
        gene_sets=args.gene_sets,
        expression_matrix_path=args.expr_matrix,
        output_dir=args.output_dir
    )

    # Create summary report
    report_path = os.path.join(args.output_dir, 'validation_report.txt')
    create_summary_report(all_tables, report_path)

    logger.info("Validation complete")


if __name__ == '__main__':
    main()
