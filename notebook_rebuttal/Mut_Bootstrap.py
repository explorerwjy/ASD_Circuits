# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: gencic
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/" # Change to your project directory
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

try:
    os.chdir(f"{ProjDIR}/notebook_rebuttal/")
    print(f"Current working directory: {os.getcwd()}")
except FileNotFoundError as e:
    print(f"Error: Could not change directory - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# Load config file
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
Anno = STR2Region()

# %%
df = pd.read_excel("/home/jw3514/Work/data/DDD/41586_2020_2832_MOESM4_ESM.xlsx")
df = df.sort_values("denovoWEST_p_full")
hc_df = df[df["denovoWEST_p_full"]<=0.05/18762]
entrez_ids = [int(GeneSymbol2Entrez.get(x, -1)) for x in hc_df["symbol"].values]
hc_df["EntrezID"] = entrez_ids
hc_df.shape

# %%
hc_df.head(5)

# %%
hc_df["AutismMerged_LoF"] = (
    df.loc[hc_df.index, "frameshift_variant"].fillna(0)
    + df.loc[hc_df.index, "splice_acceptor_variant"].fillna(0)
    + df.loc[hc_df.index, "splice_donor_variant"].fillna(0)
    + df.loc[hc_df.index, "stop_gained"].fillna(0)
    + df.loc[hc_df.index, "stop_lost"].fillna(0)
).astype(int).clip(lower=0)

hc_df["AutismMerged_Dmis_REVEL0.5"] = df.loc[hc_df.index, "missense_variant"].fillna(0).astype(int).clip(lower=0)

hc_df = hc_df[["EntrezID", "symbol", "AutismMerged_LoF", "AutismMerged_Dmis_REVEL0.5"]]
hc_df = hc_df.set_index("EntrezID", drop=False)

# %%
# Exclude ASD genes from hc_df before bootstrap
ASD_GW = Fil2Dict(ProjDIR+"dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw")
ASD_GENES = list(ASD_GW.keys())
print(f"Total genes in hc_df before excluding ASD: {len(hc_df)}")
print(f"Number of ASD genes to exclude: {len(ASD_GENES)}")

# Filter out ASD genes
hc_df = hc_df[~hc_df["EntrezID"].isin(ASD_GENES)]
print(f"Total genes in hc_df after excluding ASD: {len(hc_df)}")
print(f"Excluded {len(ASD_GENES)} ASD genes")


# %%
hc_df.head(5)


# %%
# DDD_hc_GW = Aggregate_Gene_Weights_NDD(hc_df, out="../dat/GeneWeights/DDD.hc.gw.csv")
# NDD_top61_DF = hc_df.head(61)
# DDD_top61_GW = Aggregate_Gene_Weights_NDD(NDD_top61_DF, out="../dat/GeneWeights/DDD.top61.gw.csv")
#Dict2Fil(DDD_top61_GW, "../dat/GeneWeights/DDD.top61.gw.csv")

# %%
def bootstrap_gene_mutations(
    df,
    n_boot=10,
    weighted=True,
    lof_col="AutismMerged_LoF",
    dmis_col="AutismMerged_Dmis_REVEL0.5",
    rng=None,
):
    """
    Bootstrap mutation counts at the mutation level, preserving gene identity
    and total mutation load. Supports weighted (mutation-rate) or uniform resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with gene-level LOF and Dmis mutation counts.
    n_boot : int, optional
        Number of bootstrap replicates (default = 10).
    weighted : bool, optional
        If True, mutations are resampled with probability proportional to observed counts.
        If False, each gene has equal probability of receiving any mutation.
    lof_col, dmis_col : str
        Column names for LOF and Dmis counts.
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.

    Returns
    -------
    boot_DFs : list of pd.DataFrame
        List of bootstrapped dataframes with resampled mutation counts and 'bootstrap_iter' column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(df)
    boot_DFs = []

    # Ensure integer non-negative mutation counts
    # lof = (df["frameshift_variant"] + df["splice_acceptor_variant"] + df["splice_donor_variant"] + df["stop_gained"] + df["stop_lost"]).astype(int).clip(lower=0)
    # dmis = df["missense_variant"].astype(int).clip(lower=0) 

    lof = df[lof_col].astype(int).clip(lower=0)
    dmis = df[dmis_col].astype(int).clip(lower=0)

    total_lof = lof.sum()
    total_dmis = dmis.sum()

    # Probability vectors for mutation assignment
    if weighted:
        # Weighted by observed mutation burden per gene
        p_lof = lof / total_lof if total_lof > 0 else np.ones(n) / n
        p_dmis = dmis / total_dmis if total_dmis > 0 else np.ones(n) / n
    else:
        # Uniform: every gene equally likely
        p_lof = np.ones(n) / n
        p_dmis = np.ones(n) / n

    for i in range(1, n_boot + 1):
        # Draw total_lof mutation events, assign to genes
        new_lof_counts = np.bincount(
            rng.choice(n, size=total_lof, replace=True, p=p_lof),
            minlength=n
        )
        new_dmis_counts = np.bincount(
            rng.choice(n, size=total_dmis, replace=True, p=p_dmis),
            minlength=n
        )

        # Create bootstrap replicate
        df_boot = df.copy().reset_index(drop=True)
        df_boot[lof_col] = new_lof_counts
        df_boot[dmis_col] = new_dmis_counts
        df_boot["bootstrap_iter"] = i
        df_boot["bootstrap_type"] = "weighted" if weighted else "uniform"
        boot_DFs.append(df_boot)

    return boot_DFs


# %%
boot_DFs_weights = bootstrap_gene_mutations(hc_df, 1000, weighted=True, lof_col="AutismMerged_LoF", dmis_col="AutismMerged_Dmis_REVEL0.5")

# %%
boot_DFs_weights[0]


# %%
def Aggregate_Gene_Weights_NDD2(MutFil, usepLI=False, Bmis=False, out=None):
    gene2MutN = {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            print(g, "Error converting Entrez ID")

        nLGD = row["AutismMerged_LoF"] 
        nMis = row["AutismMerged_Dmis_REVEL0.5"] 

        gene2MutN[g] = nLGD * 0.347 + nMis * 0.194
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2MutN


# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def process_bootstrap_iter(args):
    """Worker function to process a single bootstrap iteration"""
    i, DF, save_dir, str_bias_mat = args
    boot_gw = Aggregate_Gene_Weights_NDD2(DF)
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"DDD_ExomeWide.GeneWeight.boot{i}.csv"))
    return i, boot_bias

save_dir = "../results/Bootstrap_bias/DDD_ExomeWide/Weighted_Resampling"
os.makedirs(save_dir, exist_ok=True)

# Prepare arguments for parallel processing
n_workers = mp.cpu_count()  # Use all available CPU cores
args_list = [(i, DF, save_dir, STR_BiasMat) for i, DF in enumerate(boot_DFs_weights)]

# Process in parallel
boot_bias_list_weights = [None] * len(boot_DFs_weights)  # Pre-allocate list to maintain order

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    # Submit all tasks
    future_to_idx = {executor.submit(process_bootstrap_iter, args): args[0] for args in args_list}
    
    # Collect results as they complete
    for future in as_completed(future_to_idx):
        i, boot_bias = future.result()
        boot_bias_list_weights[i] = boot_bias

# %%
## LOEUF25 Bootstrap Analysis

# Load gnomad4 constraint data and define LOEUF25 gene set
gnomad4 = pd.read_csv("/home/jw3514/Work/data/gnomad/gnomad.v4.0.constraint_metrics.tsv", sep="\t")
search_text = 'ENST'
gnomad4 = gnomad4[(gnomad4["transcript"].str.contains(search_text))]
gnomad4 = gnomad4[gnomad4["mane_select"]==True]

# Convert gene symbols to Entrez IDs
for i, row in gnomad4.iterrows():
    symbol = row["gene"]
    gnomad4.loc[i, "Entrez"] = int(GeneSymbol2Entrez.get(symbol, 0))

# Take subset where lof.oe_ci.upper is in the bottom 25% (most constrained)
bottom_25_percent_threshold = gnomad4["lof.oe_ci.upper"].quantile(0.25)
gnomad4_bottom25 = gnomad4[gnomad4["lof.oe_ci.upper"] <= bottom_25_percent_threshold]
columns_to_keep_g4 = ["Entrez", "gene", "lof.pLI", "lof.z_score", "lof.oe_ci.upper"]
gnomad4_bottom25 = gnomad4_bottom25[columns_to_keep_g4]
gnomad4_bottom25 = gnomad4_bottom25.sort_values(by="lof.oe_ci.upper", ascending=True)

# Make sure Entrez is int and exclude rows with Entrez = 0
gnomad4_bottom25["Entrez"] = gnomad4_bottom25["Entrez"].astype(int)
gnomad4_bottom25 = gnomad4_bottom25[gnomad4_bottom25["Entrez"] != 0]

# Get LOEUF25 gene list (Entrez IDs)
LOEUF25_genes = gnomad4_bottom25["Entrez"].unique().tolist()
print(f"LOEUF25 gene set: {len(LOEUF25_genes)} genes")
print(f"Bottom 25% threshold: {bottom_25_percent_threshold:.4f}")


# %%
def bootstrap_genes_from_set(gene_set, n_boot=1000, n_genes=None, rng=None):
    """
    Bootstrap sample genes from a given gene set.
    
    Parameters
    ----------
    gene_set : list
        List of Entrez gene IDs to sample from
    n_boot : int
        Number of bootstrap replicates
    n_genes : int, optional
        Number of genes to sample per bootstrap. If None, uses length of gene_set
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility
    
    Returns
    -------
    boot_gene_sets : list of lists
        List of bootstrap gene sets, each containing sampled Entrez IDs
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if n_genes is None:
        n_genes = len(gene_set)
    
    gene_set_array = np.array(gene_set)
    boot_gene_sets = []
    
    for i in range(1, n_boot + 1):
        # Sample n_genes with replacement from gene_set
        boot_genes = rng.choice(gene_set_array, size=n_genes, replace=True)
        boot_gene_sets.append(boot_genes.tolist())
    
    return boot_gene_sets

# Bootstrap genes from LOEUF25 gene set
print(f"Bootstrapping {len(LOEUF25_genes)} genes from LOEUF25 gene set...")
LOEUF25_boot_gene_sets = bootstrap_genes_from_set(LOEUF25_genes, n_boot=1000, rng=np.random.default_rng(42))
print(f"Created {len(LOEUF25_boot_gene_sets)} bootstrap replicates")


# %%
# Process LOEUF25 bootstrap iterations in parallel
def process_LOEUF25_bootstrap_iter(args):
    """Worker function to process a single LOEUF25 bootstrap iteration"""
    i, boot_genes, save_dir, str_bias_mat = args
    # Create gene weights dictionary (equal weights of 1)
    boot_gw = {gene: 1.0 for gene in boot_genes}
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"LOEUF25.GeneWeight.boot{i}.csv"))
    return i, boot_bias

save_dir_LOEUF25 = "../results/Bootstrap_bias/LOEUF25/Weighted_Resampling"
os.makedirs(save_dir_LOEUF25, exist_ok=True)

# Prepare arguments for parallel processing
n_workers = mp.cpu_count()
args_list_LOEUF25 = [(i, boot_genes, save_dir_LOEUF25, STR_BiasMat) 
                     for i, boot_genes in enumerate(LOEUF25_boot_gene_sets)]

# Process in parallel
print(f"Using {n_workers} workers to compute LOEUF25 bootstrap bias...")
LOEUF25_boot_bias_list = [None] * len(LOEUF25_boot_gene_sets)

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    future_to_idx = {executor.submit(process_LOEUF25_bootstrap_iter, args): args[0] 
                     for args in args_list_LOEUF25}
    
    completed = 0
    for future in as_completed(future_to_idx):
        i, boot_bias = future.result()
        LOEUF25_boot_bias_list[i] = boot_bias
        completed += 1
        if completed % 100 == 0:
            print(f"Completed {completed}/{len(LOEUF25_boot_gene_sets)} LOEUF25 bootstrap iterations")

print(f"Completed all {len(LOEUF25_boot_bias_list)} LOEUF25 bootstrap iterations")


