"""
Generate sibling null bias files for circuit search pipeline.

For each simulation:
1. Set np.random.seed(BASE_SEED + i)
2. Sample genes from sibling pool (uniform or mutability-weighted)
3. Compute structure-level bias using MouseSTR_AvgZ_Weighted
4. Store results in a single compressed NPZ file

Two null models:
- random: uniform random sampling from sibling gene pool
- mutability: probability-weighted sampling using mutation rates

Usage:
    # Load all defaults from config (recommended):
    python scripts/generate_sibling_bias_files.py \
        --configfile config/circuit_config_sibling_random.yaml

    # Override n_sims for a test run:
    python scripts/generate_sibling_bias_files.py \
        --configfile config/circuit_config_sibling_random.yaml --n_sims 5

    # Fully manual (no config):
    python scripts/generate_sibling_bias_files.py \
        --model random --outdir results/Sibling_bias/Random/ --n_sims 10000

Output:
    Single parquet file: {outdir}/sibling_{model}_bias.parquet
    Format (matches existing null bias parquets):
        Index:   structure names (alphabetical)
        Columns: '0', '1', ..., str(n_sims-1)
        Shape:   (N_structures, N_sims)
    Plus generation_params.yaml metadata
"""

import argparse
import os
import re
import sys
import time
import yaml
import numpy as np
import pandas as pd

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ASD_Circuits import MouseSTR_AvgZ_Weighted


# ============================================================================
# Default paths (used when neither --configfile nor CLI args are provided)
# ============================================================================

DEFAULTS = {
    'expr_matrix': 'dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet',
    'gene_weights': 'dat/Genetics/GeneWeights/Spark_Meta_160.GeneWeight.csv',
    'sibling_pool': 'dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',
    'gene_prob': 'dat/Genetics/GeneWeights/GeneProb_LGD_Dmis.csv',
    'reference_bias_df': 'dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv',
    'n_sims': 10000,
    'base_seed': 20260210,
}


def load_config(configfile):
    """
    Load generation parameters from a sibling config YAML.

    Extracts model and outdir from the sibling_bias_npz path, and reads
    generation fields (expr_matrix, gene_weights, etc.) directly.
    """
    with open(configfile) as f:
        cfg = yaml.safe_load(f)

    result = {}

    # Infer model and outdir from sibling_bias_parquet path
    # Pattern: {outdir}/sibling_{model}_bias.parquet
    pq_path = cfg.get('sibling_bias_parquet', '')
    if pq_path:
        result['outdir'] = os.path.dirname(pq_path)
        m = re.search(r'sibling_(\w+)_bias\.parquet$', pq_path)
        if m:
            result['model'] = m.group(1)

    # Direct fields
    for key in ('expr_matrix', 'gene_weights', 'sibling_pool', 'gene_prob',
                'reference_bias_df', 'n_sims', 'base_seed'):
        if key in cfg and cfg[key] is not None:
            result[key] = cfg[key]

    return result


def load_expression_matrix(path):
    """Load expression matrix (genes x structures)."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, index_col=0)


def load_gene_weights(path):
    """Load gene weights file (no header, cols: entrez_id, weight)."""
    df = pd.read_csv(path, header=None)
    return df[0].values, df[1].values


def load_sibling_pool(path):
    """Load sibling gene pool (no header, cols: entrez_id, weight)."""
    df = pd.read_csv(path, header=None)
    return df[0].values


def prepare_gene_probabilities(gene_prob_path, valid_genes, sibling_genes):
    """
    Load mutation probabilities and filter to valid sibling genes.

    Returns (gene_pool, probs) arrays for np.random.choice.
    """
    prob_df = pd.read_csv(gene_prob_path, index_col=0)

    # Filter to genes in both sibling pool and expression matrix
    mask = prob_df.index.isin(sibling_genes) & prob_df.index.isin(valid_genes)
    prob_df = prob_df.loc[mask]

    probs = prob_df["Prob"].values.astype(np.float64)
    probs = probs / probs.sum()
    # Fix floating-point rounding
    probs[-1] = 1.0 - probs[:-1].sum()

    return prob_df.index.values, probs


def generate_bias_parquet(model, n_sims, base_seed, outdir,
                          expr_matrix_path, gene_weights_path,
                          sibling_pool_path, gene_prob_path=None,
                          reference_bias_df_path=None):
    """Generate all sibling null bias results and save as parquet.

    Output format matches existing null bias parquets:
        Index:   structure names (alphabetical)
        Columns: '0', '1', ..., str(n_sims-1)
        Values:  float64 EFFECT scores
        Shape:   (N_structures, N_sims)
    """

    os.makedirs(outdir, exist_ok=True)

    # Load data
    print(f"Loading expression matrix: {expr_matrix_path}")
    ExpMat = load_expression_matrix(expr_matrix_path)
    valid_genes = ExpMat.index.values
    print(f"  {len(valid_genes)} genes, {ExpMat.shape[1]} structures")

    print(f"Loading gene weights: {gene_weights_path}")
    asd_genes, asd_weights = load_gene_weights(gene_weights_path)
    # Filter to genes in expression matrix
    mask = np.isin(asd_genes, valid_genes)
    asd_genes = asd_genes[mask]
    asd_weights = asd_weights[mask]
    n_genes = len(asd_genes)
    print(f"  {n_genes} valid ASD genes (after filtering)")

    print(f"Loading sibling pool: {sibling_pool_path}")
    sibling_genes = load_sibling_pool(sibling_pool_path)
    # Filter to genes in expression matrix
    sibling_valid = sibling_genes[np.isin(sibling_genes, valid_genes)]
    print(f"  {len(sibling_valid)} valid sibling genes (of {len(sibling_genes)})")

    if len(sibling_valid) < n_genes:
        raise ValueError(
            f"Not enough sibling genes ({len(sibling_valid)}) "
            f"for gene set size ({n_genes})"
        )

    # Prepare probability weights for mutability model
    gene_pool = None
    gene_probs = None
    if model == "mutability":
        if gene_prob_path is None:
            raise ValueError("gene_prob required for mutability model")
        print(f"Loading gene probabilities: {gene_prob_path}")
        gene_pool, gene_probs = prepare_gene_probabilities(
            gene_prob_path, valid_genes, sibling_genes
        )
        print(f"  {len(gene_pool)} genes in probability-weighted pool")
        if len(gene_pool) < n_genes:
            raise ValueError(
                f"Not enough genes in probability pool ({len(gene_pool)}) "
                f"for gene set size ({n_genes})"
            )

    if reference_bias_df_path:
        print(f"Reference bias: {reference_bias_df_path}")

    # Run simulations
    print(f"\nGenerating {n_sims} sibling bias simulations (model={model})...")
    n_structures = ExpMat.shape[1]
    effects_matrix = np.empty((n_sims, n_structures), dtype=np.float64)

    # Get structure order from a reference computation
    ref_weights = dict(zip(asd_genes, asd_weights))
    ref_result = MouseSTR_AvgZ_Weighted(ExpMat, ref_weights)
    # Use alphabetical order for reproducibility — rank is stored separately
    structures = np.sort(ExpMat.columns.values)

    t0 = time.time()
    for i in range(n_sims):
        np.random.seed(base_seed + i)

        # Sample genes
        if model == "mutability":
            sampled = np.random.choice(
                gene_pool, size=n_genes, p=gene_probs, replace=False
            )
        else:  # random (uniform from sibling pool)
            sampled = np.random.choice(
                sibling_valid, size=n_genes, replace=False
            )

        # Build Gene2Weights dict: sampled gene -> original weight
        gene2weights = dict(zip(sampled, asd_weights))

        # Compute bias
        result = MouseSTR_AvgZ_Weighted(ExpMat, gene2weights)

        # Store in fixed column order
        effects_matrix[i, :] = result.loc[structures, "EFFECT"].values

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_sims - i - 1) / rate
            print(f"  [{i+1}/{n_sims}] {rate:.1f} sims/sec, ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"Completed {n_sims} simulations in {elapsed:.1f} sec")

    # Save as parquet (structures × sims) — same format as existing null bias
    # effects_matrix is (n_sims, n_structures); transpose to (n_structures, n_sims)
    effects_df = pd.DataFrame(
        effects_matrix.T,
        index=structures,
        columns=[str(i) for i in range(n_sims)]
    )
    pq_path = os.path.join(outdir, f"sibling_{model}_bias.parquet")
    effects_df.to_parquet(pq_path)
    file_size = os.path.getsize(pq_path) / (1024 * 1024)
    print(f"Saved: {pq_path} ({file_size:.1f} MB)")

    # Save generation parameters
    params = {
        'model': model,
        'n_sims': n_sims,
        'base_seed': base_seed,
        'gene_set_size': int(n_genes),
        'sibling_pool_size': int(len(sibling_valid)),
        'n_structures': int(n_structures),
        'expr_matrix': expr_matrix_path,
        'gene_weights': gene_weights_path,
        'sibling_pool': sibling_pool_path,
        'gene_prob': gene_prob_path,
        'reference_bias_df': reference_bias_df_path,
        'parquet_file': os.path.basename(pq_path),
    }
    params_path = os.path.join(outdir, "generation_params.yaml")
    with open(params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    print(f"Saved: {params_path}")

    return pq_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate sibling null bias files for circuit search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load defaults from config (recommended):
  python scripts/generate_sibling_bias_files.py \\
      --configfile config/circuit_config_sibling_random.yaml

  # Quick test run (5 sims):
  python scripts/generate_sibling_bias_files.py \\
      --configfile config/circuit_config_sibling_random.yaml --n_sims 5

  # Mutability model:
  python scripts/generate_sibling_bias_files.py \\
      --configfile config/circuit_config_sibling_mutability.yaml
""",
    )
    parser.add_argument(
        "--configfile",
        help="Sibling config YAML (loads model, outdir, paths from config). "
             "CLI args override config values."
    )
    parser.add_argument(
        "--model", choices=["random", "mutability"],
        help="Null model: 'random' (uniform) or 'mutability' (weighted)"
    )
    parser.add_argument(
        "--n_sims", type=int,
        help="Number of simulations (default: 10000)"
    )
    parser.add_argument(
        "--base_seed", type=int,
        help="Base random seed (default: 20260210)"
    )
    parser.add_argument(
        "--outdir",
        help="Output directory (default: derived from config sibling_bias_npz)"
    )
    parser.add_argument(
        "--expr_matrix",
        help="Expression matrix path"
    )
    parser.add_argument(
        "--gene_weights",
        help="Gene weights file (ASD genes)"
    )
    parser.add_argument(
        "--sibling_pool",
        help="Sibling gene pool file"
    )
    parser.add_argument(
        "--gene_prob",
        help="Gene mutation probability file (required for mutability model)"
    )
    parser.add_argument(
        "--reference_bias_df",
        help="Reference bias file (for traceability; default: ASD bias)"
    )

    args = parser.parse_args()

    # Build effective config: DEFAULTS <- configfile <- CLI args
    effective = dict(DEFAULTS)

    if args.configfile:
        print(f"Loading config: {args.configfile}")
        cfg = load_config(args.configfile)
        effective.update(cfg)
        print(f"  model={cfg.get('model', '?')}, "
              f"outdir={cfg.get('outdir', '?')}, "
              f"n_sims={cfg.get('n_sims', '?')}")

    # CLI args override (only if explicitly provided)
    cli_overrides = {}
    for key in ('model', 'n_sims', 'base_seed', 'outdir', 'expr_matrix',
                'gene_weights', 'sibling_pool', 'gene_prob',
                'reference_bias_df'):
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val
    if cli_overrides:
        effective.update(cli_overrides)
        print(f"  CLI overrides: {cli_overrides}")

    # Validate required fields
    if 'model' not in effective:
        parser.error("--model is required (or use --configfile)")
    if 'outdir' not in effective:
        parser.error("--outdir is required (or use --configfile)")

    print()
    generate_bias_parquet(
        model=effective['model'],
        n_sims=effective['n_sims'],
        base_seed=effective['base_seed'],
        outdir=effective['outdir'],
        expr_matrix_path=effective['expr_matrix'],
        gene_weights_path=effective['gene_weights'],
        sibling_pool_path=effective['sibling_pool'],
        gene_prob_path=effective.get('gene_prob'),
        reference_bias_df_path=effective.get('reference_bias_df'),
    )


if __name__ == "__main__":
    main()
