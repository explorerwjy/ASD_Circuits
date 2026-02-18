"""
Utility functions for sibling null circuit search pipeline.

Provides:
- load_bias_df(): Unified loader that handles both CSV and parquet+sim_id
  formats. Used by generate_bias_limits.py, run_sa_search.py, and
  create_pareto_front.py for backward-compatible bias loading.
- discover_sibling_datasets(): Reads parquet to build Snakemake config dicts
- get_real_sibling_bias(): Reads real (un-transferred) bias from parquet

Storage format (single parquet per model):
    Index:   structure names (alphabetical)
    Columns: '0', '1', ..., str(n_sims-1)
    Values:  float64 EFFECT scores (raw sibling bias)
    Shape:   (N_structures, N_sims)

This is the same format as the existing null bias parquets in
results/STR_ISH/null_bias/, making it directly usable for both
p-value computation (Process 1) and SA circuit search (Process 2).
"""

import os
import numpy as np
import pandas as pd


def load_bias_df(config_data):
    """
    Load bias DataFrame from dataset config, supporting CSV and parquet.

    For CSV datasets (backward compatible):
        config_data must have 'bias_df' → CSV path

    For parquet datasets (sibling null):
        config_data must have 'bias_parquet' → parquet path
        config_data must have 'sim_id' → int simulation index
        config_data may have 'reference_bias_df' → path for rank-based
            alignment (bias transfer). If present, structures are ranked
            by sibling bias but EFFECT values are replaced by the
            reference's EFFECT at the same rank.

    Parameters
    ----------
    config_data : dict
        Single dataset configuration dict from Input_str_bias.

    Returns
    -------
    pd.DataFrame
        Bias DataFrame with structure names as index, 'EFFECT' column,
        sorted descending by EFFECT, with 'Rank' column.
    """
    if 'bias_df' in config_data:
        return pd.read_csv(config_data['bias_df'], index_col=0)

    # Parquet path — read single column (efficient columnar read)
    pq_path = config_data['bias_parquet']
    sim_id = config_data['sim_id']
    df = pd.read_parquet(pq_path, columns=[str(sim_id)])
    df.columns = ['EFFECT']
    df = df.sort_values('EFFECT', ascending=False)

    # Rank-based bias transfer (if reference provided)
    if 'reference_bias_df' in config_data:
        ref_path = config_data['reference_bias_df']
        # Accept path (str) or pre-loaded DataFrame
        if isinstance(ref_path, str):
            ref = pd.read_csv(ref_path, index_col=0)
        else:
            ref = ref_path
        ref_sorted = ref.sort_values('EFFECT', ascending=False)
        n = len(df)
        n_ref = len(ref_sorted)
        transferred = np.empty(n)
        for r in range(n):
            transferred[r] = ref_sorted['EFFECT'].iloc[min(r, n_ref - 1)]
        df['EFFECT'] = transferred

    df['Rank'] = np.arange(1, len(df) + 1)
    return df


def discover_sibling_datasets(bias_parquet, sibling_dataset_prefix,
                              n_sibling=None):
    """
    Discover sibling simulations from a parquet file and create
    dataset configuration.

    Parameters
    ----------
    bias_parquet : str
        Path to the parquet file containing sibling bias data.
        Shape: (N_structures, N_sims), columns '0'..'N-1'.
    sibling_dataset_prefix : str
        Prefix for dataset names (e.g., "ASD_Sib_Random_").
    n_sibling : int or None
        Number of sibling simulations to use. If None, use all.

    Returns
    -------
    dict
        Dictionary with dataset configurations for Input_str_bias.
        Each entry has keys: name, bias_parquet, sim_id, description.
    """
    if not os.path.exists(bias_parquet):
        raise ValueError(f"Parquet file not found: {bias_parquet}")

    # Get simulation count from parquet metadata (no data read)
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(bias_parquet)
        total_sims = len(schema.names)
    except ImportError:
        # Fallback: read column names only via pandas
        pf = pd.read_parquet(bias_parquet)
        total_sims = len(pf.columns)
        del pf

    if total_sims == 0:
        raise ValueError(f"Parquet file has 0 simulations: {bias_parquet}")

    n_use = total_sims if n_sibling is None else min(n_sibling, total_sims)

    abs_pq = os.path.abspath(bias_parquet)
    datasets = {}

    for idx in range(n_use):
        dataset_key = f"Sibling_{idx}"
        dataset_name = f"{sibling_dataset_prefix}{idx}"

        datasets[dataset_key] = {
            'name': dataset_name,
            'bias_parquet': abs_pq,
            'sim_id': idx,
            'description': f"Sibling null simulation {idx}",
        }

    return datasets


def get_sibling_dataset_names(sibling_config):
    """
    Get list of sibling dataset names from configuration.

    Parameters
    ----------
    sibling_config : dict
        Sibling datasets from Input_str_bias.

    Returns
    -------
    list
        List of dataset names (for use in Snakemake expand()).
    """
    return [config['name'] for config in sibling_config.values()]


def get_real_sibling_bias(bias_parquet, sim_id, structures_list):
    """
    Compute real (un-transferred) mean sibling bias for a set of structures.

    Used in aggregation to report actual sibling bias after SA search was
    run with transferred (reference-aligned) bias values.

    Parameters
    ----------
    bias_parquet : str
        Path to sibling bias parquet file.
    sim_id : int
        Simulation index.
    structures_list : list of str
        Structure names in the circuit.

    Returns
    -------
    float
        Mean real sibling bias for the given structures.
    """
    df = pd.read_parquet(bias_parquet, columns=[str(sim_id)])
    df.columns = ['EFFECT']
    valid = [s for s in structures_list if s in df.index]
    return df.loc[valid, 'EFFECT'].mean() if valid else np.nan


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pq_path = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        pq_path = "results/Sibling_bias/Random/sibling_random_bias.parquet"
        n = 5

    print(f"Discovering sibling datasets from: {pq_path}")
    print()

    datasets = discover_sibling_datasets(pq_path, "ASD_Sib_Random_",
                                         n_sibling=n)

    print(f"Found {len(datasets)} sibling datasets:")
    for key, config in list(datasets.items())[:10]:
        print(f"  {key}: {config['name']} (sim_id={config['sim_id']})")
