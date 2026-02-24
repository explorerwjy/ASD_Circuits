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
import yaml
ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/" # Change to your project directory
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# Load config file
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
Anno = STR2Region()

# %%
ASD_61 = pd.read_csv("../dat/Genetics/GeneWeights/Spark_Meta_EWS.GeneWeight.csv", index_col=0, header=None).index.values
ASD_61 = [x for x in ASD_61 if x in STR_BiasMat.index]

# %%
ASD_61

# %%
gencic_data = pd.read_csv('../results/GENCIC_MouseSTRBias.csv')
gencic_data.head(2)

# %%
Cir_STRs = gencic_data[gencic_data["Circuits.46"]==True]["Structure"].values
Cir_STRs

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Set style for better visualization
sns.set_style("white")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# %%
# Prepare gene symbols for ASD genes
ASD_genes_symbols = [Entrez2Symbol.get(g, f"Gene_{g}") for g in ASD_61]
print(f"Number of ASD genes: {len(ASD_genes_symbols)}")
print(f"Sample gene symbols: {ASD_genes_symbols[:5]}")

# Get structure names
all_structures = STR_BiasMat.columns.values
print(f"\nTotal number of structures: {len(all_structures)}")
print(f"Number of circuit structures: {len(Cir_STRs)}")

# %%
# Extract bias scores for ASD genes across all structures, skipping genes not present in STR_BiasMat
# STR_BiasMat has genes as rows (index) and structures as columns

# Only keep genes in ASD_61 that are also in STR_BiasMat
present_ASD_genes = [g for g in ASD_61 if g in STR_BiasMat.index]

# Map these to symbols, but preserve original order
present_ASD_genes_symbols = [Entrez2Symbol.get(g, f"Gene_{g}") for g in present_ASD_genes]

ASD_bias_all = STR_BiasMat.loc[present_ASD_genes, :].copy()
ASD_bias_all.index = present_ASD_genes_symbols

# Filter for circuit structures only
ASD_bias_circuits = ASD_bias_all[Cir_STRs].copy()

print(f"Total ASD genes in input: {len(ASD_61)}")
print(f"Number of ASD genes present in expression matrix: {len(present_ASD_genes)}")
print(f"Shape of full bias matrix (ASD genes x all structures): {ASD_bias_all.shape}")
print(f"Shape of circuit bias matrix (ASD genes x circuit structures): {ASD_bias_circuits.shape}")

# Check for missing values
print(f"\nMissing values in full matrix: {ASD_bias_all.isna().sum().sum()}")
print(f"Missing values in circuit matrix: {ASD_bias_circuits.isna().sum().sum()}")

# %%
# Examine missing data pattern in detail
print("Missing Data Analysis:")
print("="*80)

# Check missing values per gene
missing_per_gene = ASD_bias_all.isna().sum(axis=1)
print(f"\nGenes with missing values: {(missing_per_gene > 0).sum()}/{len(missing_per_gene)}")
print(f"Total missing values: {ASD_bias_all.isna().sum().sum()}")
print(f"Percentage of missing data: {ASD_bias_all.isna().sum().sum() / (ASD_bias_all.shape[0] * ASD_bias_all.shape[1]) * 100:.2f}%")

# Show genes with most missing values
print("\nTop 10 genes with most missing values:")
print(missing_per_gene.sort_values(ascending=False).head(10))

# Check missing values per structure
missing_per_structure = ASD_bias_all.isna().sum(axis=0)
print(f"\nStructures with missing values: {(missing_per_structure > 0).sum()}/{len(missing_per_structure)}")
print("\nTop 10 structures with most missing values:")
print(missing_per_structure.sort_values(ascending=False).head(10))

# Visualize missing data pattern
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Missing per gene
ax1.hist(missing_per_gene, bins=20, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Missing Structures', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Missing Values per Gene', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Missing per structure
missing_per_structure_sorted = missing_per_structure.sort_values(ascending=False)
ax2.bar(range(len(missing_per_structure_sorted[missing_per_structure_sorted > 0])), 
        missing_per_structure_sorted[missing_per_structure_sorted > 0].values,
        alpha=0.7, edgecolor='black')
ax2.set_xlabel('Structure Index (sorted)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Missing Genes', fontsize=11, fontweight='bold')
ax2.set_title('Missing Values per Structure', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/Missing_Data_Pattern.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Handling Missing Data
#
# We have several strategies to handle missing values:
#
# 1. **Zero Imputation**: Replace NaN with 0 (assumes missing = not expressed)
# 2. **Mean Imputation**: Replace NaN with mean of that structure across all genes
# 3. **Median Imputation**: Replace NaN with median (more robust to outliers)
# 4. **Drop Structures**: Remove structures with many missing values
# 5. **Gene Mean**: Replace NaN with that gene's mean across all structures
#
# For gene expression bias scores, **median imputation by structure** or **zero imputation** are most appropriate depending on whether NaN means "no data" or "not expressed".

# %%
# Implement different imputation strategies
print("Comparing Imputation Strategies:")
print("="*80)

# Strategy 1: Zero imputation
ASD_bias_all_zero = ASD_bias_all.fillna(0)
print(f"\n1. Zero Imputation:")
print(f"   Missing values after: {ASD_bias_all_zero.isna().sum().sum()}")
print(f"   Mean value: {ASD_bias_all_zero.values.mean():.3f}")
print(f"   Std value: {ASD_bias_all_zero.values.std():.3f}")

# Strategy 2: Mean imputation by structure (column-wise)
ASD_bias_all_mean = ASD_bias_all.fillna(ASD_bias_all.mean(axis=0))
print(f"\n2. Mean Imputation (by structure):")
print(f"   Missing values after: {ASD_bias_all_mean.isna().sum().sum()}")
print(f"   Mean value: {ASD_bias_all_mean.values.mean():.3f}")
print(f"   Std value: {ASD_bias_all_mean.values.std():.3f}")

# Strategy 3: Median imputation by structure (column-wise)
ASD_bias_all_median = ASD_bias_all.fillna(ASD_bias_all.median(axis=0))
print(f"\n3. Median Imputation (by structure):")
print(f"   Missing values after: {ASD_bias_all_median.isna().sum().sum()}")
print(f"   Mean value: {ASD_bias_all_median.values.mean():.3f}")
print(f"   Std value: {ASD_bias_all_median.values.std():.3f}")

# Strategy 4: Gene-wise mean imputation
ASD_bias_all_gene_mean = ASD_bias_all.T.fillna(ASD_bias_all.mean(axis=1)).T
print(f"\n4. Mean Imputation (by gene):")
print(f"   Missing values after: {ASD_bias_all_gene_mean.isna().sum().sum()}")
print(f"   Mean value: {ASD_bias_all_gene_mean.values.mean():.3f}")
print(f"   Std value: {ASD_bias_all_gene_mean.values.std():.3f}")

# Strategy 5: Drop structures with >X% missing
threshold = 0.2  # Drop structures where >20% of genes have missing values
structures_to_keep = missing_per_structure < (ASD_bias_all.shape[0] * threshold)
ASD_bias_all_filtered = ASD_bias_all.loc[:, structures_to_keep].fillna(0)
print(f"\n5. Filter structures (>{threshold*100:.0f}% missing) + Zero imputation:")
print(f"   Structures removed: {(~structures_to_keep).sum()}")
print(f"   Structures remaining: {structures_to_keep.sum()}")
print(f"   Missing values after: {ASD_bias_all_filtered.isna().sum().sum()}")
print(f"   Shape: {ASD_bias_all_filtered.shape}")

print("\n" + "="*80)
print("Recommendation: Use Median Imputation (Strategy 3) or Zero Imputation (Strategy 1)")
print("- Median is robust and maintains structure-specific patterns")
print("- Zero is appropriate if NaN represents 'not detected/expressed'")
print("="*80)

# %%
# Choose imputation strategy
# Options: 'zero', 'mean', 'median', 'gene_mean', 'filter'
IMPUTATION_STRATEGY = 'median'  # Change this to your preferred strategy

if IMPUTATION_STRATEGY == 'zero':
    ASD_bias_all_imputed = ASD_bias_all_zero.copy()
    strategy_name = "Zero Imputation"
elif IMPUTATION_STRATEGY == 'mean':
    ASD_bias_all_imputed = ASD_bias_all_mean.copy()
    strategy_name = "Mean Imputation (by structure)"
elif IMPUTATION_STRATEGY == 'median':
    ASD_bias_all_imputed = ASD_bias_all_median.copy()
    strategy_name = "Median Imputation (by structure)"
elif IMPUTATION_STRATEGY == 'gene_mean':
    ASD_bias_all_imputed = ASD_bias_all_gene_mean.copy()
    strategy_name = "Mean Imputation (by gene)"
elif IMPUTATION_STRATEGY == 'filter':
    ASD_bias_all_imputed = ASD_bias_all_filtered.copy()
    strategy_name = "Filtered structures + Zero Imputation"
else:
    raise ValueError(f"Unknown imputation strategy: {IMPUTATION_STRATEGY}")

print(f"Selected imputation strategy: {strategy_name}")
print(f"Shape after imputation: {ASD_bias_all_imputed.shape}")
print(f"Missing values: {ASD_bias_all_imputed.isna().sum().sum()}")
print(f"Ready for clustering!")

# %% [markdown]
# ## Hierarchical Clustering: Circuit Structures (46 structures)
#
# First, let's cluster ASD genes based on their bias scores across the 46 circuit structures identified by GENCIC.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.patches import Patch

# --------------------------------------------------------------------
# 1. PREP DATA & COLORS
# --------------------------------------------------------------------
genes_in_matrix = [gene for gene in ASD_genes_symbols if gene in ASD_bias_circuits.index]
ASD_bias_circuits_flipped = ASD_bias_circuits.loc[genes_in_matrix].T

# --- Gene Clusters (Columns) ---
bias_for_clustering = ASD_bias_circuits.loc[genes_in_matrix]
gene_linkage = linkage(bias_for_clustering, method='ward', metric='euclidean')
n_clusters = 4
gene_clusters = fcluster(gene_linkage, n_clusters, criterion='maxclust')
gene_to_cluster = dict(zip(genes_in_matrix, gene_clusters))

gene_cluster_df = pd.DataFrame({
    'Gene': genes_in_matrix,
    'Cluster': gene_clusters
})

cluster_colors_map = {1: '#E31A1C', 2: '#1F78B4', 3: '#33A02C', 4: '#FF7F00'}
gene_colors = pd.Series(genes_in_matrix, index=ASD_bias_circuits_flipped.columns).map(gene_to_cluster).map(cluster_colors_map)

# --- Region Colors (Rows) ---
regions = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 'Hippocampus','Amygdala','Striatum',
           "Thalamus", "Hypothalamus", "Midbrain", "Medulla", "Pallidum", "Pons", "Cerebellum"]
region_colors_list = ["#268ad5", "#D5DBDB", "#7ac3fa", "#2c9d39", "#742eb5", "#ed8921",
     "#e82315", "#E6B0AA", "#f6b26b", "#20124d", "#2ECC71", "#D2B4DE", "#ffd966"]
region_colors_dict = dict(zip(regions, region_colors_list))

def get_struct_color(struct_name):
    reg = Anno.get(struct_name, 'Unknown')
    if reg == "Amygdalar": reg = "Amygdala"
    return region_colors_dict.get(reg, 'gray')

row_colors = pd.Series(ASD_bias_circuits_flipped.index, index=ASD_bias_circuits_flipped.index).map(get_struct_color)

# --------------------------------------------------------------------
# 2. DRAW CLUSTERMAP
# --------------------------------------------------------------------
sns.set(style="white")

# We turn OFF the default cbar so we can place a custom one on the right
g = sns.clustermap(
    ASD_bias_circuits_flipped,
    cmap='RdYlBu_r',
    center=0,
    vmin=-2,
    vmax=2,
    figsize=(20, 14),
    
    # Clustering
    dendrogram_ratio=0.15,
    method='ward',
    metric='euclidean',
    
    # Native Colors (Guarantees Alignment)
    row_colors=row_colors,
    col_colors=gene_colors,
    
    # Labels
    xticklabels=True, 
    yticklabels=True,
    
    # Style
    # Remove grid: set linewidths=0 and do not set linecolor!
    linewidths=0,
    cbar_pos=None  # Important: We will draw this manually
)

# --------------------------------------------------------------------
# 3. FORMATTING
# --------------------------------------------------------------------
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=14, fontweight='bold', rotation=90)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=15, fontweight='bold', rotation=0)

g.ax_heatmap.tick_params(axis='x', length=3, pad=2)
g.ax_heatmap.tick_params(axis='y', length=3, pad=2)

g.ax_heatmap.set_xlabel('ASD Genes', fontsize=24, fontweight='bold', labelpad=16)
g.ax_heatmap.set_ylabel('')

# Remove labels from the color strips
g.ax_row_colors.set_xticklabels([])
g.ax_col_colors.set_yticklabels([])

# Style dendrogram spines
for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# --------------------------------------------------------------------
# 4. LEGENDS (right side)
# --------------------------------------------------------------------
legend_kw = dict(frameon=True, edgecolor='gray', fancybox=True,
                 framealpha=0.9, borderpad=0.8, handletextpad=0.6)

# A. Gene Clusters
cluster_legend_elements = [
    Patch(facecolor=cluster_colors_map[i], edgecolor='k', linewidth=0.5,
          label=f'Cluster {i}')
    for i in sorted(cluster_colors_map.keys())
]
g.fig.legend(
    handles=cluster_legend_elements,
    loc='upper left', bbox_to_anchor=(1.12, 0.83),
    fontsize=14, title='Gene Clusters', title_fontsize=15,
    **legend_kw
)

# B. Brain Regions
unique_regions = [r for r in regions if r in region_colors_dict]
region_legend_elements = [
    Patch(facecolor=region_colors_dict[r], edgecolor='k', linewidth=0.5,
          label=r.replace('_', ' '))
    for r in unique_regions
]
g.fig.legend(
    handles=region_legend_elements,
    loc='upper left', bbox_to_anchor=(1.0, 0.83),
    fontsize=14, title='Brain Regions', title_fontsize=15,
    **legend_kw
)

# C. Colorbar
cax = g.fig.add_axes([0.93, 0.58, 0.02, 0.2])
norm = plt.Normalize(vmin=-2, vmax=2)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdYlBu_r'),
                  cax=cax, orientation='vertical')
cb.set_label('Bias Score (Z-score)', fontsize=14, labelpad=10)
cb.ax.tick_params(labelsize=12)

# --------------------------------------------------------------------
# 5. LAYOUT
# --------------------------------------------------------------------
plt.subplots_adjust(right=0.75, top=0.95, bottom=0.12)

plt.show()
#g.savefig('Final_RightSide_Legends.pdf', dpi=480, bbox_inches='tight')

# %% [markdown]
# ## Analyze HIQ vs LIQ Mutation Differences Across Gene Clusters
#
# Now let's examine whether there are differences in high/low IQ mutation patterns across the 4 gene clusters.
#

# %%
# Load mutation data from Phenotype Analysis notebook
import numpy as np
from scipy import stats

# Load IQ phenotype data
ASC_IQ_dat = pd.read_excel("../dat/Genetics/1-s2.0-S0092867419313984-mmc4.xlsx", sheet_name="Phenotype")
ASC_IQ_dat = ASC_IQ_dat[ASC_IQ_dat["Role"]=="Proband"]
ASC_IQ_dat = ASC_IQ_dat.dropna(subset=['IQ'])

ASC_HIQ = ASC_IQ_dat[ASC_IQ_dat["IQ"]>70]["Phenotype_ID"].values
ASC_LIQ = ASC_IQ_dat[ASC_IQ_dat["IQ"]<=70]["Phenotype_ID"].values

Spark_IQ_dat = pd.read_csv(os.path.join(ProjDIR, "dat/Genetics/SPARK/core_descriptive_variables.csv"))
Spark_IQ_dat = Spark_IQ_dat[Spark_IQ_dat["asd"]==True]
Spark_IQ_dat = Spark_IQ_dat.dropna(subset=['fsiq'])

Spark_HIQ = Spark_IQ_dat[Spark_IQ_dat["fsiq"]>70]["subject_sp_id"].values
Spark_LIQ = Spark_IQ_dat[Spark_IQ_dat["fsiq"]<=70]["subject_sp_id"].values

HighIQ = np.concatenate([ASC_HIQ, Spark_HIQ])
LowIQ = np.concatenate([ASC_LIQ, Spark_LIQ])

# Load mutation data
ASD_Discov_Muts = pd.read_csv(os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Discov_DNVs.txt"), delimiter="\t")
ASD_Rep_Muts = pd.read_csv(os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Rep_DNVs.txt"), delimiter="\t")
ASD_Muts = pd.concat([ASD_Discov_Muts, ASD_Rep_Muts])

# Filter to high-confidence genes
Spark_Meta_2stage = pd.read_excel(os.path.join(ProjDIR, "dat/Genetics/TabS_DenovoWEST_Stage1+2.xlsx"),
                           skiprows=2, sheet_name="TopDnEnrich")
Spark_Meta_HC = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"]<=1.3e-6]
HighConfGenes = Spark_Meta_HC["HGNC"].values
HighConfMuts = ASD_Muts[ASD_Muts["HGNC"].isin(HighConfGenes)]

# Filter mutations by LGD and Dmis
def classify_mutation_type(row):
    """Classify a mutation as LGD or Dmis"""
    GeneEff = row["GeneEff"].split(";")[0]
    if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
        return "LGD"
    elif GeneEff == "missense":
        revel = row["REVEL"].split(";")[0]
        if revel != ".":
            try:
                if float(revel) > 0.5:
                    return "Dmis"
            except:
                pass
    return None

def Filt_LGD_Mis(DF, Dmis=True):
    dat= []
    for i, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
            dat.append(row.values)
        elif GeneEff == "missense":
            if Dmis:
                if GeneEff == "missense":
                    row["REVEL"] = row["REVEL"].split(";")[0]
                    if row["REVEL"] != ".":
                        if float(row["REVEL"]) > 0.5:
                            dat.append(row.values)
            else:
                if GeneEff == "missense":
                    dat.append(row.values)
    return pd.DataFrame(dat, columns = DF.columns.values)

HighConfMuts = Filt_LGD_Mis(HighConfMuts, Dmis=True)

# Separate by IQ
HIQ_Muts = HighConfMuts[HighConfMuts["IID"].isin(HighIQ)]
LIQ_Muts = HighConfMuts[HighConfMuts["IID"].isin(LowIQ)]

print(f"Total High Confidence Mutations: {len(HighConfMuts)}")
print(f"High IQ mutations: {len(HIQ_Muts)}")
print(f"Low IQ mutations: {len(LIQ_Muts)}")


# %%
# Map gene clusters to Entrez IDs for matching with mutation data
# gene_cluster_df has Gene symbols, we need to map to HGNC for mutation matching

# Create mapping from gene symbol to cluster
gene_to_cluster = dict(zip(gene_cluster_df['Gene'], gene_cluster_df['Cluster']))

# Map gene symbols to HGNC (they should be the same, but let's verify)
# The genes in gene_cluster_df are already HGNC symbols from Entrez2Symbol mapping

# Count mutations per gene for HIQ and LIQ
HIQ_mut_counts = HIQ_Muts.groupby('HGNC').size().to_dict()
LIQ_mut_counts = LIQ_Muts.groupby('HGNC').size().to_dict()

# Create cluster-level mutation counts
cluster_hiq_counts = {}
cluster_liq_counts = {}
cluster_gene_counts = {}

for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    cluster_hiq_counts[cluster_id] = sum(HIQ_mut_counts.get(gene, 0) for gene in cluster_genes)
    cluster_liq_counts[cluster_id] = sum(LIQ_mut_counts.get(gene, 0) for gene in cluster_genes)
    cluster_gene_counts[cluster_id] = len(cluster_genes)

# Create summary dataframe
cluster_mutation_summary = pd.DataFrame({
    'Cluster': sorted(gene_cluster_df['Cluster'].unique()),
    'N_Genes': [cluster_gene_counts[c] for c in sorted(gene_cluster_df['Cluster'].unique())],
    'HIQ_Mutations': [cluster_hiq_counts[c] for c in sorted(gene_cluster_df['Cluster'].unique())],
    'LIQ_Mutations': [cluster_liq_counts[c] for c in sorted(gene_cluster_df['Cluster'].unique())]
})

cluster_mutation_summary['Total_Mutations'] = cluster_mutation_summary['HIQ_Mutations'] + cluster_mutation_summary['LIQ_Mutations']
cluster_mutation_summary['HIQ_Rate'] = cluster_mutation_summary['HIQ_Mutations'] / cluster_mutation_summary['Total_Mutations']
cluster_mutation_summary['LIQ_Rate'] = cluster_mutation_summary['LIQ_Mutations'] / cluster_mutation_summary['Total_Mutations']

# Calculate overall HIQ mutation rate (default p for binomial test)
overall_hiq_rate = cluster_mutation_summary['HIQ_Mutations'].sum() / cluster_mutation_summary['Total_Mutations'].sum()

# Perform binomial test for each cluster
from scipy.stats import binomtest
binomial_pvalues = []
binomial_results = []

for idx, row in cluster_mutation_summary.iterrows():
    n = int(row['Total_Mutations'])
    k = int(row['HIQ_Mutations'])
    if n > 0:
        # Two-sided binomial test: test if observed HIQ rate differs from overall rate
        result = binomtest(k, n, p=overall_hiq_rate, alternative='two-sided')
        binomial_pvalues.append(result.pvalue)
        binomial_results.append(result)
    else:
        binomial_pvalues.append(np.nan)
        binomial_results.append(None)

cluster_mutation_summary['Binomial_P_Value'] = binomial_pvalues

print("Mutation counts by cluster:")
print("="*80)
print(cluster_mutation_summary.to_string(index=False))

print("\n" + "="*80)
print("Binomial Test Results:")
print(f"Overall HIQ mutation rate (default p): {overall_hiq_rate:.4f}")
print("="*80)
for idx, row in cluster_mutation_summary.iterrows():
    cluster_id = row['Cluster']
    pval = row['Binomial_P_Value']
    if not np.isnan(pval):
        sig_level = ""
        if pval < 0.001:
            sig_level = "***"
        elif pval < 0.01:
            sig_level = "**"
        elif pval < 0.05:
            sig_level = "*"
        print(f"Cluster {cluster_id}: p = {pval:.6f} {sig_level}")
        print(f"  Observed HIQ rate: {row['HIQ_Rate']:.4f}, Expected: {overall_hiq_rate:.4f}")
    else:
        print(f"Cluster {cluster_id}: No mutations (cannot test)")
print("="*80)


# %%
# Permutation test: Shuffle IQ-patient labels while preserving genetic architecture
# This is more stringent as it maintains the actual mutation-gene relationships

from scipy.stats import binomtest
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Number of permutations
n_permutations = 10000

# Get unique patients from high-confidence mutations only
# Only consider patients who have mutations in high-confidence genes
hiq_patients_with_muts = set(HIQ_Muts["IID"].unique())
liq_patients_with_muts = set(LIQ_Muts["IID"].unique())
all_patients = list(hiq_patients_with_muts) + list(liq_patients_with_muts)
n_hiq_patients = len(hiq_patients_with_muts)
n_liq_patients = len(liq_patients_with_muts)
n_total_patients = len(all_patients)

print(f"Total patients with high-confidence mutations: {n_total_patients} (HIQ: {n_hiq_patients}, LIQ: {n_liq_patients})")
print(f"Running permutation test with {n_permutations} permutations...")

# Store permuted HIQ mutation counts for each cluster
permuted_hiq_counts = {cluster_id: [] for cluster_id in sorted(gene_cluster_df['Cluster'].unique())}

# Get observed HIQ mutation counts per cluster (for comparison)
observed_hiq_counts = {}
for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    observed_hiq_counts[cluster_id] = cluster_hiq_counts[cluster_id]

# Perform permutations
for perm in range(n_permutations):
    # Shuffle patient labels: randomly assign n_hiq_patients as HIQ, rest as LIQ
    shuffled_patients = all_patients.copy()
    random.shuffle(shuffled_patients)
    
    permuted_hiq = set(shuffled_patients[:n_hiq_patients])
    permuted_liq = set(shuffled_patients[n_hiq_patients:])
    
    # Recalculate mutation counts with permuted labels
    permuted_hiq_muts = HighConfMuts[HighConfMuts["IID"].isin(permuted_hiq)]
    permuted_hiq_mut_counts = permuted_hiq_muts.groupby('HGNC').size().to_dict()
    
    # Count HIQ mutations per cluster
    for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
        cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
        cluster_hiq_count = sum(permuted_hiq_mut_counts.get(gene, 0) for gene in cluster_genes)
        permuted_hiq_counts[cluster_id].append(cluster_hiq_count)
    
    # Progress indicator
    if (perm + 1) % 1000 == 0:
        print(f"  Completed {perm + 1}/{n_permutations} permutations...")

print("Permutation test completed!\n")

# Calculate p-values for each cluster
permutation_pvalues = []

for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    observed = observed_hiq_counts[cluster_id]
    null_distribution = np.array(permuted_hiq_counts[cluster_id])
    
    # Two-sided p-value: proportion of permuted values as or more extreme than observed
    # For enrichment: count how many permuted values >= observed
    # For depletion: count how many permuted values <= observed
    # Two-sided: use the more extreme tail
    n_greater_equal = np.sum(null_distribution >= observed)
    n_less_equal = np.sum(null_distribution <= observed)
    
    # Two-sided p-value: use the smaller tail, then double it (but cap at 1.0)
    p_enrichment = (n_greater_equal + 1) / (n_permutations + 1)  # +1 for continuity correction
    p_depletion = (n_less_equal + 1) / (n_permutations + 1)
    p_two_sided = 2 * min(p_enrichment, p_depletion)
    p_two_sided = min(p_two_sided, 1.0)  # Cap at 1.0
    
    permutation_pvalues.append(p_two_sided)

# Add permutation p-values to summary dataframe
cluster_mutation_summary['Permutation_P_Value'] = permutation_pvalues

# Print results
print("="*80)
print("Permutation Test Results:")
print(f"Number of permutations: {n_permutations}")
print("="*80)
for idx, row in cluster_mutation_summary.iterrows():
    cluster_id = row['Cluster']
    pval = row['Permutation_P_Value']
    observed = observed_hiq_counts[cluster_id]
    null_mean = np.mean(permuted_hiq_counts[cluster_id])
    null_std = np.std(permuted_hiq_counts[cluster_id])
    
    sig_level = ""
    if pval < 0.001:
        sig_level = "***"
    elif pval < 0.01:
        sig_level = "**"
    elif pval < 0.05:
        sig_level = "*"
    
    print(f"Cluster {cluster_id}:")
    print(f"  Observed HIQ mutations: {observed}")
    print(f"  Null mean (permuted): {null_mean:.2f} ± {null_std:.2f}")
    print(f"  Permutation p-value: {pval:.6f} {sig_level}")
    print()

print("="*80)
print("\nComparison: Binomial vs Permutation Test")
print("="*80)
print(cluster_mutation_summary[['Cluster', 'HIQ_Mutations', 'Binomial_P_Value', 'Permutation_P_Value']].to_string(index=False))


# %%
# Statistical tests: Chi-square test for independence
# Test if HIQ/LIQ distribution differs across clusters

# Create contingency table
contingency_table = cluster_mutation_summary[['HIQ_Mutations', 'LIQ_Mutations']].values
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("Statistical Analysis:")
print("="*80)
print(f"Chi-square test for independence:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  P-value: {p_value:.6f}")
print(f"  Expected frequencies:\n{expected}")

if p_value < 0.001:
    print(f"\n*** Highly significant difference (p < 0.001) ***")
elif p_value < 0.01:
    print(f"\n** Significant difference (p < 0.01) **")
elif p_value < 0.05:
    print(f"\n* Significant difference (p < 0.05) *")
else:
    print(f"\nNo significant difference (p >= 0.05)")

# Calculate per-gene mutation rates
cluster_mutation_summary['HIQ_per_Gene'] = cluster_mutation_summary['HIQ_Mutations'] / cluster_mutation_summary['N_Genes']
cluster_mutation_summary['LIQ_per_Gene'] = cluster_mutation_summary['LIQ_Mutations'] / cluster_mutation_summary['N_Genes']
cluster_mutation_summary['Total_per_Gene'] = cluster_mutation_summary['Total_Mutations'] / cluster_mutation_summary['N_Genes']

print(f"\nPer-gene mutation rates:")
print(cluster_mutation_summary[['Cluster', 'N_Genes', 'HIQ_per_Gene', 'LIQ_per_Gene', 'Total_per_Gene']].to_string(index=False))


# %%
# Standalone plot: Number of mutations with permutation p-values
# Publication-ready figure for manuscript
# Uses permutation test (more stringent, preserves genetic architecture)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

x = np.arange(len(cluster_mutation_summary))
width = 0.35

# Create bars
bars_hiq = ax.bar(x - width/2, cluster_mutation_summary['HIQ_Mutations'], width,
                  label='High IQ', color='#AED6F1', edgecolor='black', linewidth=1.5)
bars_liq = ax.bar(x + width/2, cluster_mutation_summary['LIQ_Mutations'], width,
                  label='Low IQ', color='#21618C', edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (hiq_count, liq_count) in enumerate(zip(cluster_mutation_summary['HIQ_Mutations'],
                                                cluster_mutation_summary['LIQ_Mutations'])):
    if hiq_count > 0:
        ax.text(i - width/2, hiq_count + 1, f'{int(hiq_count)}', ha='center', va='bottom',
                fontweight='bold', fontsize=17)
    if liq_count > 0:
        ax.text(i + width/2, liq_count + 1, f'{int(liq_count)}', ha='center', va='bottom',
                fontweight='bold', fontsize=17)

# Add permutation p-values above the bars (more stringent test)
max_height = max(cluster_mutation_summary['HIQ_Mutations'].max(), 
                 cluster_mutation_summary['LIQ_Mutations'].max())
y_offset = max_height * 0.15  # Position p-values above bars

for i, (cluster_id, pval) in enumerate(zip(cluster_mutation_summary['Cluster'], 
                                          cluster_mutation_summary['Permutation_P_Value'])):
    if not np.isnan(pval):
        # Format p-value
        if pval < 0.001:
            p_text = 'p < 0.001***'
            color = 'red'
        elif pval < 0.01:
            p_text = f'p = {pval:.3f}**'
            color = 'red'
        elif pval < 0.05:
            p_text = f'p = {pval:.3f}*'
            color = 'orange'
        else:
            p_text = f'p = {pval:.3f}'
            color = 'black'
        
        # Position p-value above the cluster
        ax.text(i, max(cluster_mutation_summary.iloc[i]['HIQ_Mutations'], 
                       cluster_mutation_summary.iloc[i]['LIQ_Mutations']) + y_offset,
                p_text, ha='center', va='bottom', fontsize=18, fontweight='bold', color=color)

# Formatting
#ax.set_xlabel('Gene Cluster', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Mutations', fontsize=21, fontweight='bold')
#ax.set_title('Mutation Counts by Cluster', fontsize=24, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'Cluster {c}' for c in cluster_mutation_summary['Cluster']], fontsize=25)
ax.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
ax.tick_params(axis='y', labelsize=18)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Adjust y-axis to accommodate p-values
current_ylim = ax.get_ylim()
ax.set_ylim(0, current_ylim[1] * 1.3)

plt.tight_layout()
plt.savefig('../results/GeneCluster_PerGene_MutationRates_with_Pvalues.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../results/GeneCluster_PerGene_MutationRates_with_Pvalues.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved to:")
print("  ../results/GeneCluster_PerGene_MutationRates_with_Pvalues.pdf")
print("  ../results/GeneCluster_PerGene_MutationRates_with_Pvalues.png")


# %%
# Visualization: Bar plots comparing HIQ vs LIQ mutations across clusters
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Panel 1: Stacked bar chart - Total mutations by cluster
ax1 = axes[0]
x = np.arange(len(cluster_mutation_summary))
width = 0.6

bars_hiq = ax1.bar(x, cluster_mutation_summary['HIQ_Mutations'], width, 
                    label='High IQ (IQ > 70)', color='#AED6F1', edgecolor='black', linewidth=1.5)
bars_liq = ax1.bar(x, cluster_mutation_summary['LIQ_Mutations'], width,
                    bottom=cluster_mutation_summary['HIQ_Mutations'],
                    label='Low IQ (IQ ≤ 70)', color='#21618C', edgecolor='black', linewidth=1.5)

# Add value labels
for i, (hiq, liq) in enumerate(zip(cluster_mutation_summary['HIQ_Mutations'], 
                                    cluster_mutation_summary['LIQ_Mutations'])):
    if hiq > 0:
        ax1.text(i, hiq/2, f'{int(hiq)}', ha='center', va='center', fontweight='bold', fontsize=11)
    if liq > 0:
        ax1.text(i, hiq + liq/2, f'{int(liq)}', ha='center', va='center', fontweight='bold', fontsize=11, color='white')

ax1.set_xlabel('Gene Cluster', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Mutations', fontsize=12, fontweight='bold')
ax1.set_title('Total Mutations by Cluster and IQ Group', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Cluster {c}' for c in cluster_mutation_summary['Cluster']])
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_axisbelow(True)

# Panel 2: Grouped bar chart - Per-gene mutation rates
ax2 = axes[1]
x = np.arange(len(cluster_mutation_summary))
width = 0.35

bars_hiq_pg = ax2.bar(x - width/2, cluster_mutation_summary['HIQ_per_Gene'], width,
                      label='High IQ', color='#AED6F1', edgecolor='black', linewidth=1.5)
bars_liq_pg = ax2.bar(x + width/2, cluster_mutation_summary['LIQ_per_Gene'], width,
                      label='Low IQ', color='#21618C', edgecolor='black', linewidth=1.5)

# Add value labels
for i, (hiq_pg, liq_pg) in enumerate(zip(cluster_mutation_summary['HIQ_per_Gene'],
                                          cluster_mutation_summary['LIQ_per_Gene'])):
    if hiq_pg > 0:
        ax2.text(i - width/2, hiq_pg + 0.1, f'{hiq_pg:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    if liq_pg > 0:
        ax2.text(i + width/2, liq_pg + 0.1, f'{liq_pg:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2.set_xlabel('Gene Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mutations per Gene', fontsize=12, fontweight='bold')
ax2.set_title('Per-Gene Mutation Rates by Cluster', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Cluster {c}' for c in cluster_mutation_summary['Cluster']])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_axisbelow(True)

# Panel 3: Proportion of HIQ vs LIQ mutations
ax3 = axes[2]
x = np.arange(len(cluster_mutation_summary))
width = 0.6

bars_prop = ax3.bar(x, cluster_mutation_summary['HIQ_Rate'], width,
                    label='High IQ Proportion', color='#AED6F1', edgecolor='black', linewidth=1.5)

# Add horizontal line at overall HIQ rate
overall_hiq_rate = cluster_mutation_summary['HIQ_Mutations'].sum() / cluster_mutation_summary['Total_Mutations'].sum()
ax3.axhline(y=overall_hiq_rate, color='red', linestyle='--', linewidth=2, 
            label=f'Overall HIQ Rate ({overall_hiq_rate:.2%})')

# Add value labels
for i, rate in enumerate(cluster_mutation_summary['HIQ_Rate']):
    ax3.text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3.set_xlabel('Gene Cluster', fontsize=12, fontweight='bold')
ax3.set_ylabel('Proportion of High IQ Mutations', fontsize=12, fontweight='bold')
ax3.set_title('HIQ Mutation Proportion by Cluster', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'Cluster {c}' for c in cluster_mutation_summary['Cluster']])
ax3.set_ylim(0, 1)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_axisbelow(True)

# Panel 4: Number of genes with mutations in each cluster
ax4 = axes[3]
cluster_genes_with_muts = {}
for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    genes_with_hiq = sum(1 for g in cluster_genes if g in HIQ_mut_counts and HIQ_mut_counts[g] > 0)
    genes_with_liq = sum(1 for g in cluster_genes if g in LIQ_mut_counts and LIQ_mut_counts[g] > 0)
    genes_with_any = sum(1 for g in cluster_genes if g in HIQ_mut_counts or g in LIQ_mut_counts)
    cluster_genes_with_muts[cluster_id] = {
        'HIQ': genes_with_hiq,
        'LIQ': genes_with_liq,
        'Any': genes_with_any
    }

x = np.arange(len(cluster_mutation_summary))
width = 0.25

bars_hiq_genes = ax4.bar(x - width, [cluster_genes_with_muts[c]['HIQ'] for c in cluster_mutation_summary['Cluster']],
                         width, label='Genes with HIQ muts', color='#AED6F1', edgecolor='black', linewidth=1.5)
bars_liq_genes = ax4.bar(x, [cluster_genes_with_muts[c]['LIQ'] for c in cluster_mutation_summary['Cluster']],
                         width, label='Genes with LIQ muts', color='#21618C', edgecolor='black', linewidth=1.5)
bars_total_genes = ax4.bar(x + width, [cluster_genes_with_muts[c]['Any'] for c in cluster_mutation_summary['Cluster']],
                           width, label='Genes with any muts', color='gray', edgecolor='black', linewidth=1.5)

# Add value labels
for i, cluster_id in enumerate(cluster_mutation_summary['Cluster']):
    hiq_n = cluster_genes_with_muts[cluster_id]['HIQ']
    liq_n = cluster_genes_with_muts[cluster_id]['LIQ']
    any_n = cluster_genes_with_muts[cluster_id]['Any']
    if hiq_n > 0:
        ax4.text(i - width, hiq_n + 0.3, f'{hiq_n}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    if liq_n > 0:
        ax4.text(i, liq_n + 0.3, f'{liq_n}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    if any_n > 0:
        ax4.text(i + width, any_n + 0.3, f'{any_n}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax4.set_xlabel('Gene Cluster', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Genes', fontsize=12, fontweight='bold')
ax4.set_title('Genes with Mutations by Cluster', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Cluster {c}' for c in cluster_mutation_summary['Cluster']])
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)
ax4.set_axisbelow(True)

plt.tight_layout()
plt.savefig('../results/GeneCluster_HIQ_LIQ_Mutation_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved to: ../results/GeneCluster_HIQ_LIQ_Mutation_Analysis.png")


# %%
# Detailed breakdown: Show which genes in each cluster have HIQ vs LIQ mutations
print("\n" + "="*80)
print("DETAILED GENE-LEVEL MUTATION COUNTS BY CLUSTER")
print("="*80)

for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} ({len(cluster_genes)} genes)")
    print(f"{'='*80}")
    
    gene_mutation_details = []
    for gene in cluster_genes:
        hiq_count = HIQ_mut_counts.get(gene, 0)
        liq_count = LIQ_mut_counts.get(gene, 0)
        total = hiq_count + liq_count
        if total > 0:
            gene_mutation_details.append({
                'Gene': gene,
                'HIQ': hiq_count,
                'LIQ': liq_count,
                'Total': total,
                'HIQ_Rate': hiq_count / total if total > 0 else 0
            })
    
    if gene_mutation_details:
        gene_mut_df = pd.DataFrame(gene_mutation_details).sort_values('Total', ascending=False)
        print(f"\nGenes with mutations ({len(gene_mut_df)}/{len(cluster_genes)}):")
        print(gene_mut_df.to_string(index=False))
        
        # Summary stats for this cluster
        cluster_total_hiq = gene_mut_df['HIQ'].sum()
        cluster_total_liq = gene_mut_df['LIQ'].sum()
        cluster_total = cluster_total_hiq + cluster_total_liq
        print(f"\nCluster {cluster_id} Summary:")
        print(f"  Total mutations: {cluster_total} (HIQ: {cluster_total_hiq}, LIQ: {cluster_total_liq})")
        print(f"  HIQ rate: {cluster_total_hiq/cluster_total:.1%}" if cluster_total > 0 else "  HIQ rate: N/A")
        print(f"  Genes with mutations: {len(gene_mut_df)}/{len(cluster_genes)} ({len(gene_mut_df)/len(cluster_genes)*100:.1f}%)")
    else:
        print(f"\nNo mutations found for genes in this cluster.")


# %%
# Save results to CSV
cluster_mutation_summary.to_csv('../results/GeneCluster_HIQ_LIQ_Mutation_Summary.csv', index=False)
print("Summary saved to: ../results/GeneCluster_HIQ_LIQ_Mutation_Summary.csv")

# Create detailed gene-level summary
gene_level_summary = []
for cluster_id in sorted(gene_cluster_df['Cluster'].unique()):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    for gene in cluster_genes:
        hiq_count = HIQ_mut_counts.get(gene, 0)
        liq_count = LIQ_mut_counts.get(gene, 0)
        gene_level_summary.append({
            'Gene': gene,
            'Cluster': cluster_id,
            'HIQ_Mutations': hiq_count,
            'LIQ_Mutations': liq_count,
            'Total_Mutations': hiq_count + liq_count
        })

gene_level_df = pd.DataFrame(gene_level_summary)
gene_level_df.to_csv('../results/GeneCluster_GeneLevel_Mutation_Counts.csv', index=False)
print("Gene-level details saved to: ../results/GeneCluster_GeneLevel_Mutation_Counts.csv")


# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Hierarchical Clustering: All Brain Structures (213 structures)
#
# Now let's perform clustering using all 213 brain structures to get a more comprehensive view of gene expression patterns.

# %%
# Create hierarchical clustering heatmap for all structures with imputed data
fig = plt.figure(figsize=(24, 14))

# Use clustermap for automatic hierarchical clustering with imputed data
g_all = sns.clustermap(
    ASD_bias_all_imputed,
    cmap='RdYlBu_r',
    center=0,
    vmin=-2,
    vmax=2,
    cbar_kws={'label': 'Bias Score (Z-score)', 'shrink': 0.5},
    figsize=(24, 14),
    dendrogram_ratio=0.1,
    row_cluster=True,
    col_cluster=True,
    method='ward',
    metric='euclidean',
    xticklabels=True,
    yticklabels=True,
    linewidths=0.3,
    linecolor='lightgray'
)

# Adjust labels
g_all.ax_heatmap.set_xlabel('Brain Structures (All)', fontsize=12, fontweight='bold')
g_all.ax_heatmap.set_ylabel('ASD Genes', fontsize=12, fontweight='bold')
g_all.ax_heatmap.set_xticklabels(g_all.ax_heatmap.get_xticklabels(), rotation=90, ha='right', fontsize=6)
g_all.ax_heatmap.set_yticklabels(g_all.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)

plt.suptitle(f'Hierarchical Clustering of ASD Genes by All Structure Bias Scores (n={ASD_bias_all_imputed.shape[1]})\n{strategy_name}',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

# Save the figure as PNG
g_all.savefig('../results/ASD_Gene_Clustering_AllStructures.png', dpi=300, bbox_inches='tight')
print("\nFigure saved to: ../results/ASD_Gene_Clustering_AllStructures.png")

# Save the figure as PDF
g_all.savefig('../results/ASD_Gene_Clustering_AllStructures.pdf', dpi=300, bbox_inches='tight')
print("Figure also saved to: ../results/ASD_Gene_Clustering_AllStructures.pdf")

# %%
# Extract gene clusters from the dendrogram (all structures) using imputed data
gene_linkage_all = linkage(ASD_bias_all_imputed, method='ward', metric='euclidean')

# Cut the dendrogram to get clusters
n_clusters_all = 4
gene_clusters_all = fcluster(gene_linkage_all, n_clusters_all, criterion='maxclust')

# Create a dataframe with gene clusters
# Use the same gene list as ASD_bias_all_imputed index
genes_for_all = ASD_bias_all_imputed.index.tolist()
gene_cluster_df_all = pd.DataFrame({
    'Gene': genes_for_all,
    'Cluster': gene_clusters_all
})

# Show cluster distribution
print("Gene cluster distribution (All structures):")
print(gene_cluster_df_all['Cluster'].value_counts().sort_index())
print("\n" + "="*80)

# Show genes in each cluster
for cluster_id in sorted(gene_cluster_df_all['Cluster'].unique()):
    genes_in_cluster = gene_cluster_df_all[gene_cluster_df_all['Cluster'] == cluster_id]['Gene'].values
    print(f"\nCluster {cluster_id} ({len(genes_in_cluster)} genes):")
    print(", ".join(genes_in_cluster))

# %% [markdown]
# ## Cluster Characterization and Comparison
#
# Let's compare the clusters obtained from circuit structures vs all structures, and characterize each cluster by its average bias profile.

# %%
# Compare cluster assignments between circuit and all structures
# Ensure we only compare genes that are in both analyses
common_genes = list(set(gene_cluster_df['Gene']) & set(gene_cluster_df_all['Gene']))

# Create comparison dataframe for common genes only
gene_to_cluster_circuit = dict(zip(gene_cluster_df['Gene'], gene_cluster_df['Cluster']))
gene_to_cluster_all = dict(zip(gene_cluster_df_all['Gene'], gene_cluster_df_all['Cluster']))

comparison_df = pd.DataFrame({
    'Gene': common_genes,
    'Cluster_Circuit': [gene_to_cluster_circuit[g] for g in common_genes],
    'Cluster_All': [gene_to_cluster_all[g] for g in common_genes]
})

# Check agreement between the two clustering approaches
agreement = (comparison_df['Cluster_Circuit'] == comparison_df['Cluster_All']).sum()
print(f"Genes with identical cluster assignment: {agreement}/{len(common_genes)} ({agreement/len(common_genes)*100:.1f}%)")
print("\n" + "="*80)
print("\nCluster assignment comparison:")
print(comparison_df.sort_values('Cluster_Circuit'))

# %%
# Characterize clusters by their average bias across circuit structures
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for cluster_id in range(1, n_clusters + 1):
    ax = axes[cluster_id - 1]
    
    # Get genes in this cluster
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    
    # Calculate mean bias for this cluster across circuit structures
    cluster_bias = ASD_bias_circuits.loc[cluster_genes, :].mean(axis=0).sort_values(ascending=False)
    
    # Plot
    x_pos = np.arange(len(cluster_bias))
    colors = ['red' if val > 0 else 'blue' for val in cluster_bias.values]
    
    ax.bar(x_pos, cluster_bias.values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Circuit Structures', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Bias Score', fontsize=10, fontweight='bold')
    ax.set_title(f'Cluster {cluster_id} - Average Bias Profile (n={len(cluster_genes)} genes)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos[::5])  # Show every 5th label
    ax.set_xticklabels(cluster_bias.index[::5], rotation=90, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/ASD_Gene_Cluster_Profiles_Circuit.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nFigure saved to: ../results/ASD_Gene_Cluster_Profiles_Circuit.png")

# %%
# Identify top structures for each cluster
print("Top structures enriched in each gene cluster (Circuit structures):\n")
print("="*80)

for cluster_id in range(1, n_clusters + 1):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    cluster_bias = ASD_bias_circuits.loc[cluster_genes, :].mean(axis=0).sort_values(ascending=False)
    
    print(f"\nCluster {cluster_id} ({len(cluster_genes)} genes):")
    print(f"Genes: {', '.join(cluster_genes[:10])}{'...' if len(cluster_genes) > 10 else ''}")
    print(f"\nTop 10 enriched structures:")
    for i, (struct, bias) in enumerate(cluster_bias.head(10).items(), 1):
        print(f"  {i}. {struct}: {bias:.3f}")
    
    print(f"\nBottom 5 structures (least enriched):")
    for i, (struct, bias) in enumerate(cluster_bias.tail(5).items(), 1):
        print(f"  {i}. {struct}: {bias:.3f}")

# %%
# Save cluster assignments to file
comparison_df.to_csv('../results/ASD_Gene_Cluster_Assignments.csv', index=False)
print("Cluster assignments saved to: ../results/ASD_Gene_Cluster_Assignments.csv")

# Save cluster-specific bias matrices
for cluster_id in range(1, n_clusters + 1):
    cluster_genes = gene_cluster_df[gene_cluster_df['Cluster'] == cluster_id]['Gene'].values
    cluster_bias_data = ASD_bias_circuits.loc[cluster_genes, :]
    cluster_bias_data.to_csv(f'../results/ASD_Cluster{cluster_id}_BiasMatrix_Circuit.csv')
    print(f"Cluster {cluster_id} bias matrix saved to: ../results/ASD_Cluster{cluster_id}_BiasMatrix_Circuit.csv")

# %% [markdown]
# ## Alternative Clustering: Different Numbers of Clusters
#
# Let's explore different numbers of clusters to find the optimal grouping. We'll create a dendrogram to visualize the hierarchical structure.

# %%
# Create dendrogram to visualize gene clustering
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Dendrogram for circuit structures
dendrogram(
    gene_linkage,
    labels=genes_in_matrix,  # Use genes that are actually in the circuit matrix
    ax=ax1,
    leaf_font_size=8,
    orientation='right'
)
ax1.set_xlabel('Distance', fontsize=12, fontweight='bold')
ax1.set_ylabel('ASD Genes', fontsize=12, fontweight='bold')
ax1.set_title('Hierarchical Clustering Dendrogram\n(Circuit Structures, n=46)', 
              fontsize=12, fontweight='bold')
ax1.axvline(x=gene_linkage[-n_clusters, 2], color='red', linestyle='--', 
            label=f'{n_clusters} clusters cutoff')
ax1.legend()

# Dendrogram for all structures (with imputed data)
dendrogram(
    gene_linkage_all,
    labels=genes_for_all,  # Use genes from imputed matrix
    ax=ax2,
    leaf_font_size=8,
    orientation='right'
)
ax2.set_xlabel('Distance', fontsize=12, fontweight='bold')
ax2.set_ylabel('ASD Genes', fontsize=12, fontweight='bold')
ax2.set_title(f'Hierarchical Clustering Dendrogram\n(All Structures, n={ASD_bias_all_imputed.shape[1]}, {strategy_name})', 
              fontsize=12, fontweight='bold')
ax2.axvline(x=gene_linkage_all[-n_clusters_all, 2], color='red', linestyle='--', 
            label=f'{n_clusters_all} clusters cutoff')
ax2.legend()

plt.tight_layout()
plt.savefig('../results/ASD_Gene_Clustering_Dendrograms.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nDendrogram saved to: ../results/ASD_Gene_Clustering_Dendrograms.png")

# %%
# Evaluate different numbers of clusters using silhouette score
from sklearn.metrics import silhouette_score

cluster_range = range(2, 8)
silhouette_scores_circuit = []
silhouette_scores_all = []

for n_clust in cluster_range:
    # Circuit structures
    clusters_circ = fcluster(gene_linkage, n_clust, criterion='maxclust')
    sil_score_circ = silhouette_score(bias_for_clustering, clusters_circ, metric='euclidean')
    silhouette_scores_circuit.append(sil_score_circ)
    
    # All structures (with imputed data)
    clusters_all_temp = fcluster(gene_linkage_all, n_clust, criterion='maxclust')
    sil_score_all = silhouette_score(ASD_bias_all_imputed, clusters_all_temp, metric='euclidean')
    silhouette_scores_all.append(sil_score_all)

# Plot silhouette scores
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(cluster_range, silhouette_scores_circuit, 'o-', label='Circuit Structures (n=46)', 
        linewidth=2, markersize=8)
ax.plot(cluster_range, silhouette_scores_all, 's-', label=f'All Structures (n={ASD_bias_all_imputed.shape[1]}, {strategy_name})', 
        linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Cluster Quality by Number of Clusters', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xticks(cluster_range)

plt.tight_layout()
plt.savefig('../results/ASD_Gene_Clustering_SilhouetteScore.png', dpi=300, bbox_inches='tight')
plt.show()

print("Silhouette Scores:")
print("="*80)
print("\nCircuit Structures (n=46):")
for n_clust, score in zip(cluster_range, silhouette_scores_circuit):
    print(f"  {n_clust} clusters: {score:.4f}")
    
print(f"\nAll Structures (n={ASD_bias_all_imputed.shape[1]}, {strategy_name}):")
for n_clust, score in zip(cluster_range, silhouette_scores_all):
    print(f"  {n_clust} clusters: {score:.4f}")

optimal_n_circuit = cluster_range[np.argmax(silhouette_scores_circuit)]
optimal_n_all = cluster_range[np.argmax(silhouette_scores_all)]
print(f"\nOptimal number of clusters (Circuit): {optimal_n_circuit}")
print(f"Optimal number of clusters (All): {optimal_n_all}")

# %% [markdown]
# ## Summary
#
# This notebook performs comprehensive hierarchical clustering analysis of ASD genes based on GENCIC structure-level bias scores.
#
# ### Key Findings:
#
# 1. **Two Clustering Approaches:**
#    - **Circuit Structures (n=46):** Focuses on the 46 structures identified as part of the ASD-affected circuits (no missing data)
#    - **All Structures (n=213):** Uses the complete set of brain structures for a more comprehensive view (with imputation for missing values)
#
# 2. **Missing Data Handling:**
#    - Circuit structures (n=46): No missing values detected
#    - All structures (n=213): 46 missing values across the dataset (~0.36% of data)
#    - **Imputation strategy**: Configurable (default: median imputation by structure)
#    - Alternative strategies available: zero imputation, mean imputation, gene-wise mean, or filtering structures
#    - Median imputation maintains structure-specific patterns while being robust to outliers
#
# 3. **Outputs Generated:**
#    - Missing data analysis and visualization
#    - Hierarchical clustering heatmaps with dendrograms
#    - Cluster assignment tables
#    - Average bias profiles for each cluster
#    - Dendrograms showing gene relationships
#    - Silhouette score analysis for optimal cluster number
#    - Individual cluster bias matrices (CSV files)
#
# 4. **Files Saved:**
#    - `Missing_Data_Pattern.png`: Visualization of missing data distribution
#    - `ASD_Gene_Clustering_CircuitStructures.png`: Heatmap for circuit structures
#    - `ASD_Gene_Clustering_AllStructures.png`: Heatmap for all structures (with imputation method noted)
#    - `ASD_Gene_Cluster_Assignments.csv`: Cluster assignments for all genes
#    - `ASD_Gene_Cluster_Profiles_Circuit.png`: Average bias profiles per cluster
#    - `ASD_Gene_Clustering_Dendrograms.png`: Hierarchical dendrograms
#    - `ASD_Gene_Clustering_SilhouetteScore.png`: Cluster quality metrics
#    - `ASD_Cluster{N}_BiasMatrix_Circuit.csv`: Individual cluster bias matrices
#
# ### Interpretation:
# - Genes within the same cluster show similar bias patterns across brain structures
# - Different clusters may represent distinct functional or spatial expression patterns
# - Comparison between circuit and all structures reveals whether ASD genes cluster differently when considering the full brain vs. targeted circuits
# - The imputation strategy can be changed by modifying the `IMPUTATION_STRATEGY` variable (options: 'zero', 'mean', 'median', 'gene_mean', 'filter')
