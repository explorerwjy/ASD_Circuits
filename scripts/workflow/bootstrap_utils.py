"""
Utility functions for bootstrap circuit search pipeline.

This module provides functions to discover and configure bootstrap datasets
dynamically from file patterns.
"""

import os
import glob
import re


def discover_bootstrap_datasets(bootstrap_bias_dir, bootstrap_bias_pattern,
                                 bootstrap_dataset_prefix, n_bootstrap=None):
    """
    Discover bootstrap bias files and create dataset configuration.

    Parameters:
    -----------
    bootstrap_bias_dir : str
        Directory containing bootstrap bias files
    bootstrap_bias_pattern : str
        Glob pattern for bootstrap files (e.g., "*.boot*.csv")
    bootstrap_dataset_prefix : str
        Prefix for dataset names (e.g., "ASD_Boot")
    n_bootstrap : int or None
        Number of bootstrap iterations to use. If None, use all found files.

    Returns:
    --------
    dict : Dictionary with dataset configurations for Input_str_bias
    """

    # Find all bootstrap files
    pattern = os.path.join(bootstrap_bias_dir, bootstrap_bias_pattern)
    bootstrap_files = sorted(glob.glob(pattern))

    if not bootstrap_files:
        raise ValueError(f"No bootstrap files found matching pattern: {pattern}")

    # Extract bootstrap indices from filenames
    # Assumes format: ...boot{N}.csv
    bootstrap_indices = []
    for fpath in bootstrap_files:
        fname = os.path.basename(fpath)
        match = re.search(r'boot(\d+)\.csv', fname)
        if match:
            bootstrap_indices.append(int(match.group(1)))

    if not bootstrap_indices:
        raise ValueError(f"Could not extract bootstrap indices from filenames: {bootstrap_files[:3]}")

    bootstrap_indices = sorted(bootstrap_indices)

    # Limit to n_bootstrap if specified
    if n_bootstrap is not None:
        bootstrap_indices = bootstrap_indices[:n_bootstrap]
        bootstrap_files = bootstrap_files[:n_bootstrap]

    # Create dataset configurations
    datasets = {}

    for idx, fpath in zip(bootstrap_indices, bootstrap_files):
        # Make path absolute
        abs_path = os.path.abspath(fpath)

        # Create dataset key and name
        dataset_key = f"Bootstrap_{idx}"
        dataset_name = f"{bootstrap_dataset_prefix}{idx}"

        datasets[dataset_key] = {
            'name': dataset_name,
            'bias_df': abs_path,
            'description': f"Bootstrap iteration {idx}",
            'bootstrap_id': idx
        }

    return datasets


def get_bootstrap_dataset_names(bootstrap_config):
    """
    Get list of bootstrap dataset names from configuration.

    Parameters:
    -----------
    bootstrap_config : dict
        Bootstrap datasets from Input_str_bias

    Returns:
    --------
    list : List of dataset names (for use in Snakemake expand())
    """
    return [config['name'] for config in bootstrap_config.values()]


def get_main_and_bootstrap_datasets(Input_str_bias, include_main=True):
    """
    Separate main dataset from bootstrap datasets.

    Parameters:
    -----------
    Input_str_bias : dict
        Full Input_str_bias configuration
    include_main : bool
        Whether to include main dataset in output

    Returns:
    --------
    tuple : (main_datasets_dict, bootstrap_datasets_dict)
    """
    main_datasets = {}
    bootstrap_datasets = {}

    for key, config in Input_str_bias.items():
        if 'bootstrap_id' in config:
            bootstrap_datasets[key] = config
        else:
            main_datasets[key] = config

    if include_main:
        return main_datasets, bootstrap_datasets
    else:
        return {}, bootstrap_datasets


if __name__ == "__main__":
    # Test the discovery function
    import sys

    if len(sys.argv) > 1:
        bootstrap_dir = sys.argv[1]
        pattern = sys.argv[2] if len(sys.argv) > 2 else "*.boot*.csv"
    else:
        bootstrap_dir = "results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling"
        pattern = "Spark_ExomeWide.GeneWeight.boot*.csv"

    print(f"Discovering bootstrap files in: {bootstrap_dir}")
    print(f"Pattern: {pattern}")
    print()

    datasets = discover_bootstrap_datasets(
        bootstrap_dir, pattern, "ASD_Boot", n_bootstrap=5
    )

    print(f"Found {len(datasets)} bootstrap datasets:")
    for key, config in datasets.items():
        print(f"  {key}: {config['name']} -> {os.path.basename(config['bias_df'])}")
