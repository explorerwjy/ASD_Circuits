# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_generate_geneweights.py
# Generate gene weights for control simulations
# Enhanced version with gene property matching (length, conservation, expression)
# ========================================================================================================

import argparse
import sys
import pandas as pd
import numpy as np
import os
import yaml
import logging

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

from ASD_Circuits import *
from gene_matching import GenePropertyMatcher, MatchingConfig, load_gene_annotations

def prepare_gene_probabilities(GeneProb, valid_genes, sibling_filter=None):
    """
    Load and prepare gene probabilities for sampling.
    
    Args:
        GeneProb: Path to gene probability CSV file
        valid_genes: List of valid genes to filter by
        sibling_filter: Optional list of sibling genes to further filter by
    
    Returns:
        tuple: (gene_pool, gene_probs) or (None, None) if GeneProb is None
    """
    if GeneProb is None:
        return None, None
    
    Gene2Prob = pd.read_csv(GeneProb, index_col=0)
    
    # Apply filters
    if sibling_filter is not None:
        # Filter to genes that are in sibling_filter, valid_genes, and Gene2Prob
        filtered_genes = [g for g in Gene2Prob.index.values if g in sibling_filter and g in valid_genes]
    else:
        # Filter to genes that are in valid_genes and Gene2Prob
        filtered_genes = [g for g in Gene2Prob.index.values if g in valid_genes]
    
    Gene2Prob = Gene2Prob.loc[filtered_genes, :]
    probs = Gene2Prob["Prob"].values
    total = np.sum(probs)
    probs = probs / total
    
    # Ensure probabilities sum to 1
    if len(probs) > 1:
        probs[-1] = 1 - np.sum(probs[:-1])
    Gene2Prob["Prob"] = probs
    
    return Gene2Prob.index.values, Gene2Prob["Prob"].values

def SiblingGenes(ExpMat, WeightDF, outfile, GeneProb, n_sims=10000):
    # Load sibling genes
    sibling_weights_path = os.path.join(ProjDIR, config["data_files"]["sibling_weights"])
    SibWeightDF = pd.read_csv(sibling_weights_path, header=None)
    SibGenes = SibWeightDF[0].values

    if ExpMat.endswith('.parquet'):
        ExpMat = pd.read_parquet(ExpMat)
    else:
        ExpMat = pd.read_csv(ExpMat, index_col=0)
    valid_genes = ExpMat.index.values

    # Load gene weights and filter to valid genes
    WeightDF = pd.read_csv(WeightDF, header=None)
    ValidWeightDF = WeightDF[WeightDF[0].isin(valid_genes)]
    entrez_ids = ValidWeightDF[0].values
    Gene_Weights = ValidWeightDF[1].values
    print(len(Gene_Weights))

    # Filter valid_genes to only include sibling genes
    sibling_valid_genes = [g for g in valid_genes if g in SibGenes]
    
    # Check if we have enough sibling genes
    if len(sibling_valid_genes) < len(Gene_Weights):
        print(f"Warning: Not enough sibling genes ({len(sibling_valid_genes)}) for gene set size ({len(Gene_Weights)})")
        print("Skipping sibling null generation - insufficient sibling genes available")
        # Create empty output file to indicate skipping
        outdir = os.path.dirname(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(outfile, 'w') as f:
            f.write("# Skipped: Insufficient sibling genes\n")
        return

    # Prepare simulation matrix: rows = genes, columns = [weight, sim0, sim1, ...]
    sim_matrix = np.empty((len(entrez_ids), n_sims), dtype=object)

    # Prepare gene probabilities
    gene_pool, gene_probs = prepare_gene_probabilities(GeneProb, valid_genes, SibGenes)

    if gene_pool is not None:
        # Check if gene_pool has enough genes
        if len(gene_pool) < len(Gene_Weights):
            print(f"Warning: Not enough genes in probability-weighted sibling pool ({len(gene_pool)}) for gene set size ({len(Gene_Weights)})")
            print("Skipping sibling null generation - insufficient genes in probability pool")
            # Create empty output file to indicate skipping
            outdir = os.path.dirname(outfile)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write("# Skipped: Insufficient genes in probability pool\n")
            return
            
        for i in range(n_sims):
            Genes = np.random.choice(gene_pool, size=len(Gene_Weights), p=gene_probs, replace=False)
            sim_matrix[:, i] = Genes
    else:
        for i in range(n_sims):
            Genes = np.random.choice(sibling_valid_genes, size=len(Gene_Weights), replace=False)
            sim_matrix[:, i] = Genes

    # Build output DataFrame
    out_df = pd.DataFrame(sim_matrix, index=entrez_ids, columns=[str(i) for i in range(n_sims)])
    out_df.insert(0, "GeneWeight", Gene_Weights)

    # make dir if not exist
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(outfile)
    print(f"Saved all {n_sims} sibling gene simulations to {outfile}")

def RandomGenes(ExpMat, WeightDF, outfile, GeneProb, n_sims=10000):
    if ExpMat.endswith('.parquet'):
        ExpMat = pd.read_parquet(ExpMat)
    else:
        ExpMat = pd.read_csv(ExpMat, index_col=0)
    valid_genes = ExpMat.index.values

    # Load gene weights and filter to valid genes
    WeightDF = pd.read_csv(WeightDF, header=None)
    ValidWeightDF = WeightDF[WeightDF[0].isin(valid_genes)]
    entrez_ids = ValidWeightDF[0].values
    Gene_Weights = ValidWeightDF[1].values
    print(len(Gene_Weights))

    # Prepare simulation matrix: rows = genes, columns = [weight, sim0, sim1, ...]
    sim_matrix = np.empty((len(entrez_ids), n_sims), dtype=object)

    # Prepare gene probabilities
    gene_pool, gene_probs = prepare_gene_probabilities(GeneProb, valid_genes)

    if gene_pool is not None:
        for i in range(n_sims):
            Genes = np.random.choice(gene_pool, size=len(Gene_Weights), p=gene_probs, replace=False)
            sim_matrix[:, i] = Genes
    else:
        for i in range(n_sims):
            Genes = np.random.choice(valid_genes, size=len(Gene_Weights), replace=False)
            sim_matrix[:, i] = Genes

    # Build output DataFrame
    out_df = pd.DataFrame(sim_matrix, index=entrez_ids, columns=[str(i) for i in range(n_sims)])
    out_df.insert(0, "GeneWeight", Gene_Weights)

    # make dir if not exist
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(outfile)
    print(f"Saved all {n_sims} simulations to {outfile}")


def MatchedGenes(ExpMat, WeightDF, outfile, n_sims=10000, stringency='medium',
                 gene_annotations_path=None, random_state=42):
    """
    Generate null gene sets with enhanced matching on gene properties.

    This function implements the enhanced null model that matches on:
    - Gene length (CDS length)
    - Conservation (phastCons)
    - Expression level

    Parameters
    ----------
    ExpMat : str
        Path to expression matrix (parquet or CSV)
    WeightDF : str
        Path to gene weights file
    outfile : str
        Output file path
    n_sims : int
        Number of simulations
    stringency : str
        Matching stringency level: 'loose', 'medium', 'tight', 'very_tight'
    gene_annotations_path : str, optional
        Path to pre-computed gene annotations file
    random_state : int
        Random seed for reproducibility
    """
    logger.info(f"Generating matched null with stringency: {stringency}")

    # Load expression matrix
    if ExpMat.endswith('.parquet'):
        ExpMatDF = pd.read_parquet(ExpMat)
    else:
        ExpMatDF = pd.read_csv(ExpMat, index_col=0)
    valid_genes = set(ExpMatDF.index.values)

    # Load gene weights and filter to valid genes
    WeightDFData = pd.read_csv(WeightDF, header=None)
    ValidWeightDF = WeightDFData[WeightDFData[0].isin(valid_genes)]
    entrez_ids = ValidWeightDF[0].values
    gene_weights_dict = dict(zip(ValidWeightDF[0].values, ValidWeightDF[1].values))
    logger.info(f"Query gene set size: {len(entrez_ids)}")

    # Load gene annotations
    if gene_annotations_path and os.path.exists(gene_annotations_path):
        logger.info(f"Loading pre-computed gene annotations from {gene_annotations_path}")
        gene_annotations = pd.read_csv(gene_annotations_path, index_col=0)
    else:
        # Load annotations from default paths
        gnomad_path = config.get("data_files", {}).get(
            "gnomad_constraints",
            "/home/jw3514/Work/Resources/gnomad.v2.1.1.lof_metrics.by_gene.txt"
        )
        hgnc_path = os.path.join(ProjDIR, config["data_files"]["protein_coding_genes"])
        cds_path = "/home/jw3514/Work/Resources/gencode_v19_longest_cds_per_gene.tsv"

        gene_annotations = load_gene_annotations(
            gnomad_path=gnomad_path,
            hgnc_path=hgnc_path,
            cds_length_path=cds_path
        )

    # Filter annotations to valid genes
    gene_annotations = gene_annotations[gene_annotations.index.isin(valid_genes)]
    logger.info(f"Gene annotations available for {len(gene_annotations)} genes")

    # Define stringency configurations
    stringency_configs = {
        'loose': MatchingConfig(
            length_bins=5, conservation_bins=5, expression_bins=5,
            tolerance=2.0, min_candidates=100,
            match_length=True, match_conservation=False, match_expression=True
        ),
        'medium': MatchingConfig(
            length_bins=10, conservation_bins=10, expression_bins=10,
            tolerance=1.0, min_candidates=50,
            match_length=True, match_conservation=True, match_expression=True
        ),
        'tight': MatchingConfig(
            length_bins=15, conservation_bins=15, expression_bins=15,
            tolerance=0.5, min_candidates=30,
            match_length=True, match_conservation=True, match_expression=True
        ),
        'very_tight': MatchingConfig(
            length_bins=20, conservation_bins=20, expression_bins=20,
            tolerance=0.0, min_candidates=20,
            match_length=True, match_conservation=True, match_expression=True
        )
    }

    if stringency not in stringency_configs:
        logger.warning(f"Unknown stringency '{stringency}', using 'medium'")
        stringency = 'medium'

    config_obj = stringency_configs[stringency]

    # Create matcher
    matcher = GenePropertyMatcher(
        gene_annotations=gene_annotations,
        expression_matrix=ExpMatDF,
        config=config_obj
    )

    # Generate matched null
    out_df = matcher.generate_matched_null(
        query_genes=list(entrez_ids),
        gene_weights=gene_weights_dict,
        n_permutations=n_sims,
        random_state=random_state
    )

    # Log matching quality
    stats = matcher.get_matching_statistics(list(entrez_ids))
    mean_candidates = stats['n_candidates'].mean()
    min_candidates = stats['n_candidates'].min()
    logger.info(f"Matching quality: mean {mean_candidates:.1f}, min {min_candidates} candidates per gene")

    if min_candidates < 10:
        low_cand_genes = stats[stats['n_candidates'] < 10]['gene'].tolist()
        logger.warning(f"{len(low_cand_genes)} genes have < 10 candidates. Consider using looser matching.")

    # Save output
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(outfile)
    logger.info(f"Saved {n_sims} matched null simulations to {outfile}")

    # Also save matching statistics
    stats_file = outfile.replace('.csv', '_matching_stats.csv')
    stats.to_csv(stats_file, index=False)
    logger.info(f"Saved matching statistics to {stats_file}")


def run_sensitivity_analysis_nulls(ExpMat, WeightDF, outdir, n_sims=1000, random_state=42):
    """
    Generate null gene sets across multiple matching stringencies for sensitivity analysis.

    This is useful for the reviewer response to show that results are robust
    across different matching strategies.
    """
    stringency_levels = ['loose', 'medium', 'tight', 'very_tight']

    for level in stringency_levels:
        outfile = os.path.join(outdir, f"null_weights_matched_{level}.csv")
        MatchedGenes(
            ExpMat=ExpMat,
            WeightDF=WeightDF,
            outfile=outfile,
            n_sims=n_sims,
            stringency=level,
            random_state=random_state
        )

    logger.info(f"Completed sensitivity analysis with {len(stringency_levels)} stringency levels")


###########################################################################
## Args and Main Functions
###########################################################################
def GetOptions():
    parser = argparse.ArgumentParser(
        description="Generate null gene weights for permutation testing. "
                    "Supports original sibling/random methods plus enhanced "
                    "property-matched null generation."
    )
    parser.add_argument('-o', '--outfile', type=str, help='Output file')
    parser.add_argument('-w', '--WeightDF', type=str, help='Weight DF for control geneset')
    parser.add_argument('-p', '--GeneProb', default=None, help='GeneProb Filename or None if dont use')
    parser.add_argument('--n_sims', type=int, default=10000, help='Number of ctrl simulations')
    parser.add_argument('--GW_Dir', type=str, help="directory of ctrl gene weights")
    parser.add_argument('--SpecMat', type=str, help="Filename of bias matrix")

    # New arguments for enhanced matching
    parser.add_argument('--method', type=str, default='all',
                        choices=['sibling', 'random', 'matched', 'all', 'sensitivity'],
                        help='Null generation method: sibling, random, matched, all, or sensitivity')
    parser.add_argument('--stringency', type=str, default='medium',
                        choices=['loose', 'medium', 'tight', 'very_tight'],
                        help='Matching stringency for property-matched null')
    parser.add_argument('--gene_annotations', type=str, default=None,
                        help='Path to pre-computed gene annotations file')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    SpecMat = args.SpecMat
    WeightDF = args.WeightDF
    outfile = args.outfile
    GeneProb = args.GeneProb
    n_sims = args.n_sims
    method = args.method

    # Determine output filenames
    if outfile.endswith('_random.csv'):
        base_outfile = outfile.replace('_random.csv', '')
    elif outfile.endswith('.csv'):
        base_outfile = outfile.replace('.csv', '')
    else:
        base_outfile = outfile

    sibling_outfile = f"{base_outfile}_sibling.csv"
    random_outfile = f"{base_outfile}_random.csv"
    matched_outfile = f"{base_outfile}_matched_{args.stringency}.csv"

    # Run requested methods
    if method in ['sibling', 'all']:
        logger.info("Generating sibling null...")
        SiblingGenes(SpecMat, WeightDF, sibling_outfile, GeneProb, n_sims)

    if method in ['random', 'all']:
        logger.info("Generating random null...")
        RandomGenes(SpecMat, WeightDF, random_outfile, GeneProb, n_sims)

    if method in ['matched', 'all']:
        logger.info(f"Generating matched null with {args.stringency} stringency...")
        MatchedGenes(
            ExpMat=SpecMat,
            WeightDF=WeightDF,
            outfile=matched_outfile,
            n_sims=n_sims,
            stringency=args.stringency,
            gene_annotations_path=args.gene_annotations,
            random_state=args.random_state
        )

    if method == 'sensitivity':
        logger.info("Running sensitivity analysis across all stringency levels...")
        outdir = os.path.dirname(outfile) or '.'
        run_sensitivity_analysis_nulls(
            ExpMat=SpecMat,
            WeightDF=WeightDF,
            outdir=outdir,
            n_sims=n_sims,
            random_state=args.random_state
        )

    logger.info("Null generation complete.")
    return

if __name__ == '__main__':
    main()
