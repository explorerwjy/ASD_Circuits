# Author: jywang explorerwjy@gmail.com
# ========================================================================================================
# script_specificity_sensitivity.py
# Sensitivity analysis for specificity score capping
# Addresses Reviewer 3's minor point about specificity score cap at 2
# ========================================================================================================

import argparse
import sys
import os
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple

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
    GetPermutationP_vectorized
)


def apply_specificity_cap(expression_matrix: pd.DataFrame, cap_value: float) -> pd.DataFrame:
    """
    Apply a cap to specificity (z-score) values.

    This clips the expression z-scores at the specified cap value,
    which controls how much influence extreme expression values have
    on the bias calculation.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        Expression z-score matrix (genes x cell_types/structures)
    cap_value : float
        Maximum absolute value for z-scores (e.g., 2.0 or 3.0)

    Returns
    -------
    pd.DataFrame
        Clipped expression matrix
    """
    return expression_matrix.clip(lower=-cap_value, upper=cap_value)


def run_cap_sensitivity_analysis(
    expression_matrix_path: str,
    gene_weights_path: str,
    cap_values: List[float] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    mode: str = 'mouse_str_bias',
    output_dir: str = None
) -> pd.DataFrame:
    """
    Run sensitivity analysis across different specificity score caps.

    This addresses Reviewer 3's concern about the impact of the specificity
    score cap on results.

    Parameters
    ----------
    expression_matrix_path : str
        Path to expression z-score matrix
    gene_weights_path : str
        Path to gene weights file
    cap_values : list of float
        List of cap values to test
    mode : str
        Analysis mode ('mouse_str_bias', 'mouse_ct_bias')
    output_dir : str, optional
        Directory to save results

    Returns
    -------
    pd.DataFrame
        Summary of results across cap values
    """
    logger.info(f"Running specificity cap sensitivity analysis")
    logger.info(f"Testing cap values: {cap_values}")

    # Load expression matrix
    expr_mat = load_expression_matrix_cached(expression_matrix_path)
    logger.info(f"Loaded expression matrix: {expr_mat.shape}")

    # Load gene weights
    gene_weights = Fil2Dict(gene_weights_path)
    logger.info(f"Loaded {len(gene_weights)} gene weights")

    results = {}
    all_bias_dfs = {}

    for cap in cap_values:
        logger.info(f"Processing cap = {cap}")

        # Apply cap to expression matrix
        capped_expr = apply_specificity_cap(expr_mat, cap)

        # Calculate bias
        if mode == 'mouse_str_bias':
            bias_df = MouseSTR_AvgZ_Weighted(capped_expr, gene_weights)
        elif mode == 'mouse_ct_bias':
            bias_df = MouseCT_AvgZ_Weighted(capped_expr, gene_weights)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_bias_dfs[cap] = bias_df

        # Store key statistics
        results[cap] = {
            'cap_value': cap,
            'mean_effect': bias_df['EFFECT'].mean(),
            'std_effect': bias_df['EFFECT'].std(),
            'max_effect': bias_df['EFFECT'].max(),
            'min_effect': bias_df['EFFECT'].min(),
            'top5_mean': bias_df.head(5)['EFFECT'].mean(),
            'top10_mean': bias_df.head(10)['EFFECT'].mean(),
            'top20_mean': bias_df.head(20)['EFFECT'].mean(),
        }

        # Get top 10 structures/cell types for comparison
        top10_labels = list(bias_df.head(10).index)
        results[cap]['top10_labels'] = '; '.join(str(x) for x in top10_labels)

    # Create summary DataFrame
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.reset_index(drop=True)

    # Calculate correlations between different cap values
    logger.info("Calculating cross-cap correlations...")
    correlation_results = []

    cap_list = sorted(all_bias_dfs.keys())
    reference_cap = 3.0  # Current default

    for cap in cap_list:
        if reference_cap in all_bias_dfs:
            ref_df = all_bias_dfs[reference_cap]
            test_df = all_bias_dfs[cap]

            # Align indices
            common_idx = ref_df.index.intersection(test_df.index)
            ref_effects = ref_df.loc[common_idx, 'EFFECT']
            test_effects = test_df.loc[common_idx, 'EFFECT']

            # Calculate Pearson correlation
            corr = np.corrcoef(ref_effects, test_effects)[0, 1]

            # Calculate Spearman rank correlation
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(ref_effects, test_effects)

            # Calculate overlap in top N
            ref_top20 = set(ref_df.head(20).index)
            test_top20 = set(test_df.head(20).index)
            overlap_top20 = len(ref_top20.intersection(test_top20))

            ref_top50 = set(ref_df.head(50).index)
            test_top50 = set(test_df.head(50).index)
            overlap_top50 = len(ref_top50.intersection(test_top50))

            correlation_results.append({
                'cap_value': cap,
                'reference_cap': reference_cap,
                'pearson_r': corr,
                'spearman_rho': spearman_corr,
                'top20_overlap': overlap_top20,
                'top50_overlap': overlap_top50
            })

    corr_df = pd.DataFrame(correlation_results)

    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, 'specificity_cap_sensitivity_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")

        corr_path = os.path.join(output_dir, 'specificity_cap_correlations.csv')
        corr_df.to_csv(corr_path, index=False)
        logger.info(f"Saved correlations to {corr_path}")

        # Save individual bias results for each cap value
        for cap, bias_df in all_bias_dfs.items():
            cap_str = str(cap).replace('.', '_')
            bias_path = os.path.join(output_dir, f'bias_cap{cap_str}.csv')
            bias_df.to_csv(bias_path)

    return summary_df, corr_df, all_bias_dfs


def compare_cap_to_reference(
    all_bias_dfs: Dict[float, pd.DataFrame],
    reference_cap: float = 3.0,
    test_cap: float = 2.0
) -> pd.DataFrame:
    """
    Detailed comparison between two cap values.

    Parameters
    ----------
    all_bias_dfs : dict
        Dictionary mapping cap values to bias DataFrames
    reference_cap : float
        Reference cap value (current default)
    test_cap : float
        Test cap value to compare against reference

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with effects from both caps
    """
    if reference_cap not in all_bias_dfs or test_cap not in all_bias_dfs:
        raise ValueError("Both cap values must be in all_bias_dfs")

    ref_df = all_bias_dfs[reference_cap].copy()
    test_df = all_bias_dfs[test_cap].copy()

    # Merge on index
    comparison = ref_df[['EFFECT', 'Rank']].copy()
    comparison.columns = [f'EFFECT_cap{reference_cap}', f'Rank_cap{reference_cap}']

    comparison[f'EFFECT_cap{test_cap}'] = test_df.loc[comparison.index, 'EFFECT']
    comparison[f'Rank_cap{test_cap}'] = test_df.loc[comparison.index, 'Rank']

    # Calculate differences
    comparison['EFFECT_diff'] = (
        comparison[f'EFFECT_cap{test_cap}'] - comparison[f'EFFECT_cap{reference_cap}']
    )
    comparison['Rank_diff'] = (
        comparison[f'Rank_cap{test_cap}'] - comparison[f'Rank_cap{reference_cap}']
    )

    # Sort by reference effect
    comparison = comparison.sort_values(f'EFFECT_cap{reference_cap}', ascending=False)

    return comparison


def create_sensitivity_report(
    summary_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: str
):
    """
    Create a formatted report for the sensitivity analysis.

    This can be included as a supplementary table for reviewer response.
    """
    with open(output_path, 'w') as f:
        f.write("# Specificity Score Cap Sensitivity Analysis\n\n")
        f.write("## Summary\n\n")
        f.write("This analysis examines the robustness of mutation bias results\n")
        f.write("to different specificity score (z-score) cap values.\n\n")

        f.write("## Effect Size Summary by Cap Value\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("## Correlation with Reference (cap=3.0)\n\n")
        f.write(corr_df.to_string(index=False))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("- Pearson r: Correlation of effect sizes\n")
        f.write("- Spearman rho: Rank correlation (preserved ordering)\n")
        f.write("- Top20/Top50 overlap: Number of shared top-ranked structures/cell types\n")
        f.write("\n")

        # Key findings
        if len(corr_df) > 0:
            cap2_row = corr_df[corr_df['cap_value'] == 2.0]
            if len(cap2_row) > 0:
                pearson_r = cap2_row['pearson_r'].values[0]
                spearman_rho = cap2_row['spearman_rho'].values[0]
                top20_overlap = cap2_row['top20_overlap'].values[0]

                f.write("## Key Finding (Cap=2 vs Cap=3)\n\n")
                f.write(f"- Pearson correlation: r = {pearson_r:.4f}\n")
                f.write(f"- Spearman rank correlation: rho = {spearman_rho:.4f}\n")
                f.write(f"- Top 20 overlap: {top20_overlap}/20 structures\n")

    logger.info(f"Created sensitivity report at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for specificity score capping"
    )
    parser.add_argument('--expr_matrix', required=True,
                        help='Path to expression z-score matrix')
    parser.add_argument('--gene_weights', required=True,
                        help='Path to gene weights file')
    parser.add_argument('--mode', default='mouse_str_bias',
                        choices=['mouse_str_bias', 'mouse_ct_bias'],
                        help='Analysis mode')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--caps', nargs='+', type=float,
                        default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                        help='Cap values to test')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting specificity cap sensitivity analysis")

    summary_df, corr_df, all_bias_dfs = run_cap_sensitivity_analysis(
        expression_matrix_path=args.expr_matrix,
        gene_weights_path=args.gene_weights,
        cap_values=args.caps,
        mode=args.mode,
        output_dir=args.output_dir
    )

    # Create comparison report
    if 2.0 in all_bias_dfs and 3.0 in all_bias_dfs:
        comparison = compare_cap_to_reference(all_bias_dfs, reference_cap=3.0, test_cap=2.0)
        comparison_path = os.path.join(args.output_dir, 'cap2_vs_cap3_comparison.csv')
        comparison.to_csv(comparison_path)
        logger.info(f"Saved cap comparison to {comparison_path}")

    # Create formatted report
    report_path = os.path.join(args.output_dir, 'sensitivity_report.txt')
    create_sensitivity_report(summary_df, corr_df, report_path)

    logger.info("Sensitivity analysis complete")


if __name__ == '__main__':
    main()
