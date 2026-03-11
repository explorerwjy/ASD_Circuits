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

# %% [markdown]
# ## Generate DDD Gene Weight Files

# %%
# Produce DDD.top293.gw from full 293 significant genes (before ASD exclusion)
# This is the canonical source for this gene weight file
DDD_top293_GW = Aggregate_Gene_Weights_NDD(hc_df, out="../dat/Genetics/GeneWeights/DDD.top293.gw")
print(f"DDD.top293.gw: {len(DDD_top293_GW)} genes, top weight: {max(DDD_top293_GW.values()):.3f}")

# %%
# Exclude ASD genes (using DN weights) — done before column subsetting
# so Aggregate_Gene_Weights_NDD can read the raw Excel columns directly
ASD_GW = Fil2Dict(ProjDIR+"dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw")
ASD_GENES = list(ASD_GW.keys())
hc_df_excl = hc_df[~hc_df["EntrezID"].isin(ASD_GENES)]
print(f"DDD genes: {len(hc_df)}, after excluding {len(ASD_GENES)} ASD genes: {len(hc_df_excl)}")

# %%
# Produce DDD.top245.ExcludeASD.gw using library function (reads raw variant columns)
DDD_ExcludeASD_GW = Aggregate_Gene_Weights_NDD(hc_df_excl, out="../dat/Genetics/GeneWeights/DDD.top245.ExcludeASD.gw")
print(f"DDD.top245.ExcludeASD.gw: {len(DDD_ExcludeASD_GW)} genes")

# %%
# Prepare columns for bootstrap (aggregate LoF variants, rename for NDD context)
hc_df_excl["NDD_LoF"] = (
    hc_df_excl["frameshift_variant"].fillna(0)
    + hc_df_excl["splice_acceptor_variant"].fillna(0)
    + hc_df_excl["splice_donor_variant"].fillna(0)
    + hc_df_excl["stop_gained"].fillna(0)
    + hc_df_excl["stop_lost"].fillna(0)
).astype(int).clip(lower=0)
hc_df_excl["NDD_Dmis"] = hc_df_excl["missense_variant"].fillna(0).astype(int).clip(lower=0)
hc_df_excl = hc_df_excl[["EntrezID", "symbol", "NDD_LoF", "NDD_Dmis"]]
hc_df_excl = hc_df_excl.set_index("EntrezID", drop=False)
hc_df_excl.head(5)

# %%
boot_DFs_weights = bootstrap_gene_mutations(hc_df_excl, 1000, weighted=True, lof_col="NDD_LoF", dmis_col="NDD_Dmis")

# %%
boot_DFs_weights[0]


# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def ndd_gene_weights(df, lof_col="NDD_LoF", dmis_col="NDD_Dmis"):
    """Compute NDD gene weights from aggregated LoF/Dmis counts."""
    return {int(row["EntrezID"]): row[lof_col] * 0.347 + row[dmis_col] * 0.194
            for _, row in df.iterrows()}

def process_bootstrap_iter(args):
    """Worker function to process a single bootstrap iteration"""
    i, DF, save_dir, str_bias_mat = args
    boot_gw = ndd_gene_weights(DF)
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


