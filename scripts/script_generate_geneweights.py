# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_generate_geneweights.py
# Generate gene weights for control simulations
# ========================================================================================================

import argparse
import sys
import pandas as pd
import numpy as np
import os
import yaml

# Load config to get project directory
def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
ProjDIR = config["ProjDIR"]
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

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


###########################################################################
## Args and Main Functions
###########################################################################
def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, help='Output file')
    parser.add_argument('-w', '--WeightDF', type=str, help='Weight DF for control geneset')
    parser.add_argument('-p', '--GeneProb', default=None, help='GeneProb Filname or None if dont use')
    parser.add_argument('--n_sims', type=int, default=10000, help='Number of ctrl simulations')
    parser.add_argument('--GW_Dir', type=str, help="dirctory of ctrl gene weights")
    parser.add_argument('--SpecMat', type=str, help="Filename of bias matrix")

    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    SpecMat = args.SpecMat
    WeightDF = args.WeightDF
    outfile = args.outfile
    GeneProb = args.GeneProb
    n_sims = args.n_sims
    
    # The Snakefile passes the _random.csv filename, but we need both _sibling.csv and _random.csv
    # So we derive the sibling filename from the random filename
    if outfile.endswith('_random.csv'):
        sibling_outfile = outfile.replace('_random.csv', '_sibling.csv')
        random_outfile = outfile  # Use as-is
    else:
        # Fallback: assume generic outfile, add suffixes
        base_name = outfile.replace('.csv', '')
        sibling_outfile = f"{base_name}_sibling.csv"
        random_outfile = f"{base_name}_random.csv"
    
    SiblingGenes(SpecMat, WeightDF, sibling_outfile, GeneProb, n_sims)
    RandomGenes(SpecMat, WeightDF, random_outfile, GeneProb, n_sims)


    return

if __name__ == '__main__':
    main()
