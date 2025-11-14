#!/usr/bin/env python
"""
Permutation test for comparing bias between two gene sets.

This script performs permutation-based statistical testing to compare bias patterns
between two gene sets (e.g., ASD vs DDD). It calculates p-values for bias, rank,
and residuals after controlling for shared signal via linear regression.

Usage:
    python script_permutation_bias_comparison.py \\
        --gw1 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \\
        --gw2 dat/Genetics/GeneWeights_DN/DDD.top293.ExcludeASD.DN.gw \\
        --expr_mat dat/BiasMatrices/MouseCT.TPM.1.Filt.Spec.clip.lowexp.100000.qn.parquet \\
        --cluster_ann dat/MouseCT_Cluster_Anno.csv \\
        --n_perms 10000 \\
        --output results/permutation_ASD_vs_DDD.csv \\
        --filter_class "09 CNU-LGE GABA" \\
        --seed 42
"""

import sys
import os
import argparse
from sklearn.linear_model import LinearRegression
import random
from tqdm import tqdm

# Add project directory to path
ProjDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, f'{ProjDIR}/src/')
from ASD_Circuits import *

def permute_gene_labels(dict1, dict2, seed=None):
    """
    Permute disease labels by randomly splitting pooled genes into two sets.
    Each gene keeps its original weight, but disease assignment is shuffled.

    Parameters:
    -----------
    dict1 : dict
        First gene set {gene_id: weight}
    dict2 : dict
        Second gene set {gene_id: weight}
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    perm_dict1 : dict
        Permuted first set with same size as dict1
    perm_dict2 : dict
        Permuted second set with same size as dict2
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Pool all (gene, weight) pairs
    all_genes = list(dict1.keys()) + list(dict2.keys())
    all_weights = list(dict1.values()) + list(dict2.values())

    # Shuffle the pooled genes while keeping gene-weight pairs together
    combined = list(zip(all_genes, all_weights))
    random.shuffle(combined)
    all_genes, all_weights = zip(*combined)

    # Split back into two sets of original sizes
    n1 = len(dict1)
    perm_dict1 = dict(zip(all_genes[:n1], all_weights[:n1]))
    perm_dict2 = dict(zip(all_genes[n1:], all_weights[n1:]))

    return perm_dict1, perm_dict2


def merge_bias_datasets(dataset1, dataset2, suffixes=('_1', '_2')):
    """
    Merge two structure bias datasets for comparison.

    Parameters:
    -----------
    dataset1 : DataFrame
        First dataset with 'Rank', 'EFFECT' columns
    dataset2 : DataFrame
        Second dataset with 'Rank' and 'EFFECT' columns
    suffixes : tuple of str
        Suffixes to append to column names for each dataset

    Returns:
    --------
    merged_data : DataFrame
        Merged dataset with comparison metrics for both Rank and EFFECT
    """
    # Select all relevant columns
    dataset1_cols = ['Rank', 'EFFECT']
    dataset2_cols = ['Rank', 'EFFECT']

    # Merge the datasets on structure names for comparison
    merged_data = pd.merge(dataset1[dataset1_cols], dataset2[dataset2_cols],
                          left_index=True, right_index=True, suffixes=suffixes)

    # Calculate differences for both Rank and EFFECT metrics
    merged_data[f'DIFF_Rank'] = merged_data[f'Rank{suffixes[0]}'] - merged_data[f'Rank{suffixes[1]}']
    merged_data[f'ABS_DIFF_Rank'] = np.abs(merged_data[f'DIFF_Rank'])

    merged_data[f'DIFF_EFFECT'] = merged_data[f'EFFECT{suffixes[0]}'] - merged_data[f'EFFECT{suffixes[1]}']
    merged_data[f'ABS_DIFF_EFFECT'] = np.abs(merged_data[f'DIFF_EFFECT'])

    # Sort by absolute difference in EFFECT by default
    merged_data = merged_data.sort_values('ABS_DIFF_EFFECT', ascending=False)

    return merged_data


def fit_structure_bias_linear_model(merged_data, metric='EFFECT', suffixes=('_1', '_2')):
    """
    Fit linear model to predict one bias from another and calculate residuals.

    Parameters:
    -----------
    merged_data : DataFrame
        Merged dataset from merge_bias_datasets
    metric : str
        Metric to use ('EFFECT' or 'Rank')
    suffixes : tuple of str
        Suffixes used in merge

    Returns:
    --------
    results_df : DataFrame
        DataFrame with predicted values and residuals added
    """
    X = merged_data[f'{metric}{suffixes[1]}'].values.reshape(-1, 1)
    y = merged_data[f'{metric}{suffixes[0]}'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    results_df = merged_data.copy()
    results_df['predicted'] = y_pred
    results_df['residual'] = residuals

    return results_df


def calculate_permutation_pvalues(obs_df, null_dfs, metric='residual'):
    """
    Calculate permutation p-values for all structures/cell types.

    Parameters:
    -----------
    obs_df : DataFrame
        Observed results
    null_dfs : list of DataFrame
        List of permutation results
    metric : str
        Metric to calculate p-values for

    Returns:
    --------
    pvalues : Series
        P-values for each structure/cell type
    """
    pvalues = []
    for idx in obs_df.index:
        obs = obs_df.loc[idx, metric]
        null = [df.loc[idx, metric] for df in null_dfs]
        # Two-tailed test: probability of seeing as large or larger absolute value
        pval = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
        pvalues.append(pval)

    return pd.Series(pvalues, index=obs_df.index, name=f'pval_{metric}')


def main():
    parser = argparse.ArgumentParser(
        description='Permutation test for comparing bias between two gene sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--gw1', required=True,
                        help='Path to first gene weight file')
    parser.add_argument('--gw2', required=True,
                        help='Path to second gene weight file')
    parser.add_argument('--expr_mat', required=True,
                        help='Path to expression matrix (CSV or Parquet)')
    parser.add_argument('--output', required=True,
                        help='Output file path for results')

    # Optional arguments
    parser.add_argument('--cluster_ann', default=None,
                        help='Path to cluster annotation file (for adding class labels)')
    parser.add_argument('--filter_class', default=None,
                        help='Filter to specific cell type class (e.g., "09 CNU-LGE GABA")')
    parser.add_argument('--n_perms', type=int, default=10000,
                        help='Number of permutations (default: 10000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--label1', default='GeneSet1',
                        help='Label for first gene set (default: GeneSet1)')
    parser.add_argument('--label2', default='GeneSet2',
                        help='Label for second gene set (default: GeneSet2)')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of parallel processes (default: 1)')

    args = parser.parse_args()

    print("=" * 80)
    print("Permutation Test for Gene Set Bias Comparison")
    print("=" * 80)

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Load gene weights
    print(f"\nLoading gene weights...")
    print(f"  Gene set 1 ({args.label1}): {args.gw1}")
    gw1 = Fil2Dict(args.gw1)
    print(f"    Loaded {len(gw1)} genes")

    print(f"  Gene set 2 ({args.label2}): {args.gw2}")
    gw2 = Fil2Dict(args.gw2)
    print(f"    Loaded {len(gw2)} genes")

    # Load expression matrix
    print(f"\nLoading expression matrix: {args.expr_mat}")
    if args.expr_mat.endswith('.parquet'):
        expr_mat = pd.read_parquet(args.expr_mat)
    elif args.expr_mat.endswith('.csv.gz'):
        expr_mat = pd.read_csv(args.expr_mat, index_col=0)
    elif args.expr_mat.endswith('.csv'):
        expr_mat = pd.read_csv(args.expr_mat, index_col=0)
    else:
        raise ValueError("Expression matrix must be .csv, .csv.gz, or .parquet")
    print(f"  Expression matrix shape: {expr_mat.shape}")

    # Load cluster annotation if provided
    cluster_ann = None
    if args.cluster_ann:
        print(f"\nLoading cluster annotations: {args.cluster_ann}")
        cluster_ann = pd.read_csv(args.cluster_ann, index_col="cluster_id_label")
        print(f"  Loaded annotations for {len(cluster_ann)} clusters")

    # Filter expression matrix by class if specified
    if args.filter_class and cluster_ann is not None:
        print(f"\nFiltering to class: {args.filter_class}")
        cluster_subset = cluster_ann[cluster_ann['class_id_label'] == args.filter_class].index.tolist()
        expr_mat = expr_mat.loc[:, cluster_subset]
        print(f"  Filtered to {len(cluster_subset)} cell types/clusters")

    # Calculate observed bias
    print(f"\nCalculating observed bias...")
    bias1 = MouseCT_AvgZ_Weighted(expr_mat, gw1)
    bias2 = MouseCT_AvgZ_Weighted(expr_mat, gw2)

    if cluster_ann is not None:
        bias1 = add_class(bias1, cluster_ann)
        bias2 = add_class(bias2, cluster_ann)

    print(f"  {args.label1} bias calculated for {len(bias1)} cell types")
    print(f"  {args.label2} bias calculated for {len(bias2)} cell types")

    # Merge and fit linear model
    print(f"\nFitting linear model...")
    merged_obs = merge_bias_datasets(bias1, bias2, suffixes=(f'_{args.label1}', f'_{args.label2}'))
    results_obs = fit_structure_bias_linear_model(merged_obs, metric='EFFECT',
                                                   suffixes=(f'_{args.label1}', f'_{args.label2}'))

    # Run permutations
    print(f"\nRunning {args.n_perms} permutations...")
    results_perm_list = []

    for i in tqdm(range(args.n_perms), desc="Permutations"):
        # Permute gene labels
        perm_seed = None if args.seed is None else args.seed + i
        perm_gw1, perm_gw2 = permute_gene_labels(gw1, gw2, seed=perm_seed)

        # Calculate bias for permuted sets
        perm_bias1 = MouseCT_AvgZ_Weighted(expr_mat, perm_gw1)
        perm_bias2 = MouseCT_AvgZ_Weighted(expr_mat, perm_gw2)

        # Merge and fit model
        merged_perm = merge_bias_datasets(perm_bias1, perm_bias2,
                                         suffixes=(f'_{args.label1}', f'_{args.label2}'))
        results_perm = fit_structure_bias_linear_model(merged_perm, metric='EFFECT',
                                                       suffixes=(f'_{args.label1}', f'_{args.label2}'))
        results_perm_list.append(results_perm)

    # Calculate p-values for different metrics
    print(f"\nCalculating p-values...")
    pval_residual = calculate_permutation_pvalues(results_obs, results_perm_list, metric='residual')
    pval_diff_effect = calculate_permutation_pvalues(results_obs, results_perm_list, metric='DIFF_EFFECT')
    pval_diff_rank = calculate_permutation_pvalues(results_obs, results_perm_list, metric='DIFF_Rank')

    # Compile final results
    final_results = results_obs.copy()
    final_results['pval_residual'] = pval_residual
    final_results['pval_DIFF_EFFECT'] = pval_diff_effect
    final_results['pval_DIFF_Rank'] = pval_diff_rank

    # Sort by residual p-value
    final_results = final_results.sort_values('pval_residual')

    # Save results
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_results.to_csv(args.output)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTop 10 cell types with most significant residuals:")
    print(final_results[['EFFECT_' + args.label1, 'EFFECT_' + args.label2,
                         'residual', 'pval_residual']].head(10).to_string())

    # Count significant results at different thresholds
    for alpha in [0.05, 0.01, 0.001]:
        n_sig = (final_results['pval_residual'] < alpha).sum()
        print(f"\nNumber of cell types with p < {alpha}: {n_sig} ({100*n_sig/len(final_results):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
