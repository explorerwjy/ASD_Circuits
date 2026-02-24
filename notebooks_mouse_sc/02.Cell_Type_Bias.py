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
#     display_name: Python 3 (gencic)
#     language: python
#     name: gencic
# ---

# %% [markdown]
# # Cell-Type Bias Calculation (Allen Brain Cell Atlas)
#
# Computes ASD and DDD cell-type bias using the cluster-level Z2 expression
# specificity matrix. The pipeline:
#
# 1. Compute per-gene V2-V3 chemistry correlation (10x Chromium)
# 2. Create DN (DeNoise) gene weights: `weight_DN = weight_ISH × (V2_V3_CT_Corr)²`
# 3. Compute weighted bias per cluster
# 4. Test ASD-specific residuals (relative to DDD) in striatal CNU-LGE GABA clusters

# %% [markdown]
# ## 0. Setup

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(1, '../src')
from ASD_Circuits import (LoadGeneINFO, Fil2Dict, Dict2Fil,
                          MouseCT_AvgZ_Weighted, add_class,
                          merge_bias_datasets, fit_structure_bias_linear_model)
from plot import cluster_residual_boxplot

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)
ProjDIR = config["ProjDIR"]

# %% [markdown]
# ## 1. V2-V3 Subclass Expression Correlation
#
# For each gene, compute Spearman correlation of expression across 338 subclasses
# between 10x V2 and V3 chemistry. Also compute ISH-MERFISH structure correlation.
# This identifies genes with inconsistent expression across platforms.

# %%
CORR_FILE = f"../{config['data_files']['gene_cross_platform_corr']}"
EXPECTED_GENES = 16870

if os.path.exists(CORR_FILE):
    CorrDF = pd.read_csv(CORR_FILE, index_col="Genes")
    if len(CorrDF) == EXPECTED_GENES:
        print(f"Loaded cached correlation: {CorrDF.shape} from {CORR_FILE}")
    else:
        print(f"WARNING: cached file has {len(CorrDF)} genes, expected {EXPECTED_GENES}. Recomputing.")
        CorrDF = None
else:
    CorrDF = None

if CorrDF is None:
    # Load expression matrices
    V2_Exp = pd.read_csv(f"../{config['data_files']['subclass_v2_exp']}", index_col=0)
    V3_Exp = pd.read_csv(f"../{config['data_files']['subclass_v3_exp']}", index_col=0)
    ISH_Exp = pd.read_csv(f"../{config['data_files']['ish_log_mean_exp']}", index_col=0)
    MERFISH_Exp = pd.read_csv(f"../{config['data_files']['merfish_cell_mean_umi']}", index_col=0)

    print(f"V2: {V2_Exp.shape}, V3: {V3_Exp.shape}, ISH: {ISH_Exp.shape}, MERFISH: {MERFISH_Exp.shape}")

    # Shared genes across all four matrices
    shared_genes = (set(ISH_Exp.index) & set(MERFISH_Exp.index)
                    & set(V2_Exp.index) & set(V3_Exp.index))
    # Shared structures between ISH and MERFISH
    ish_cols = set(ISH_Exp.columns)
    merfish_cols = set(MERFISH_Exp.columns)
    shared_strs = sorted(ish_cols & merfish_cols)
    # Shared subclasses between V2 and V3
    shared_subs = sorted(set(V2_Exp.columns) & set(V3_Exp.columns))

    print(f"Shared genes: {len(shared_genes)}, shared structures: {len(shared_strs)}, "
          f"shared subclasses: {len(shared_subs)}")

    rows = []
    for gene in sorted(shared_genes):
        ish_vals = ISH_Exp.loc[gene, shared_strs].values.astype(float)
        mer_vals = MERFISH_Exp.loc[gene, shared_strs].values.astype(float)
        v2_vals = V2_Exp.loc[gene, shared_subs].values.astype(float)
        v3_vals = V3_Exp.loc[gene, shared_subs].values.astype(float)

        corr_ish_mer = spearmanr(ish_vals, mer_vals).correlation
        corr_v2_v3 = spearmanr(v2_vals, v3_vals).correlation
        symbol = Entrez2Symbol.get(gene, "")

        rows.append({
            'Genes': gene,
            'Corr': corr_ish_mer,
            'Symbol': symbol,
            'ISH_exp': np.mean(ish_vals),
            'MERFISH_exp': np.mean(mer_vals),
            'V2_V3_CT_Corr': corr_v2_v3,
        })

    CorrDF = pd.DataFrame(rows).set_index('Genes')
    CorrDF.to_csv(CORR_FILE)
    print(f"Computed and saved correlation: {CorrDF.shape}")

print(f"Median V2-V3 correlation: {CorrDF['V2_V3_CT_Corr'].median():.4f}")
print(f"Median ISH-MERFISH correlation: {CorrDF['Corr'].median():.4f}")

# %% [markdown]
# ## 2. DN Gene Weights
#
# Transform ISH gene weights by V2-V3 correlation to downweight genes
# with inconsistent expression across 10x chemistry versions:
#
# `weight_DN = weight_ISH × (V2_V3_CT_Corr)²`

# %%
os.makedirs("../dat/Genetics/GeneWeights_DN", exist_ok=True)

v2v3_corr = CorrDF['V2_V3_CT_Corr']

# ASD DN weights (gene_sets paths are absolute in config)
ASD_GW_raw = Fil2Dict(config['gene_sets']['ASD_All']['geneweights'])
ASD_GW_DN = {}
for gene, weight in ASD_GW_raw.items():
    if gene in v2v3_corr.index:
        ASD_GW_DN[gene] = weight * (v2v3_corr.loc[gene] ** 2)

asd_dn_path = f"../{config['data_files']['asd_gene_weights_dn']}"
Dict2Fil(ASD_GW_DN, asd_dn_path)
print(f"ASD: {len(ASD_GW_raw)} raw genes -> {len(ASD_GW_DN)} DN genes -> {asd_dn_path}")

# DDD DN weights
DDD_GW_raw = Fil2Dict(config['gene_sets']['DDD_293_ExcludeASD']['geneweights'])
DDD_GW_DN = {}
for gene, weight in DDD_GW_raw.items():
    if gene in v2v3_corr.index:
        DDD_GW_DN[gene] = weight * (v2v3_corr.loc[gene] ** 2)

ddd_dn_path = f"../{config['data_files']['ddd_gene_weights_dn']}"
Dict2Fil(DDD_GW_DN, ddd_dn_path)
print(f"DDD: {len(DDD_GW_raw)} raw genes -> {len(DDD_GW_DN)} DN genes -> {ddd_dn_path}")

# %%
# Validate against existing DN files
for label, computed_gw, existing_path in [
    ("ASD", ASD_GW_DN, asd_dn_path),
    ("DDD", DDD_GW_DN, ddd_dn_path),
]:
    existing_gw = Fil2Dict(existing_path)
    assert set(computed_gw.keys()) == set(existing_gw.keys()), f"{label}: gene sets differ"
    max_diff = max(abs(computed_gw[g] - existing_gw[g]) for g in computed_gw)
    print(f"{label} DN validation: {len(computed_gw)} genes, max weight diff = {max_diff:.2e}")

# %% [markdown]
# ## 3. Load Data

# %%
# Cluster annotations (5312 clusters with class/subclass labels)
ClusterAnn = pd.read_csv(f"../{config['data_files']['mouse_ct_annotation']}", index_col="cluster_id_label")

# Cell type hierarchy -> Class2Cluster mapping
CellTypesDF = pd.read_csv(f"../{config['data_files']['cell_type_hierarchy']}")
Class2Cluster = {}
for _, row in CellTypesDF.iterrows():
    _cluster, _class = row.iloc[0], row.iloc[1]
    Class2Cluster.setdefault(_class, []).append(_cluster)
print(f"Loaded {len(ClusterAnn)} clusters, {len(Class2Cluster)} classes")

# %%
# Z2 expression specificity matrix (16916 genes x 5312 clusters)
MouseSC_Z2 = pd.read_parquet(f"../{config['analysis_types']['CT_Z2']['expr_matrix']}")
print(f"Z2 matrix: {MouseSC_Z2.shape}")

# %% [markdown]
# ## 4. Compute ASD and DDD Bias

# %%
os.makedirs("../results/CT_Z2", exist_ok=True)

# ASD bias (60 DN genes)
ASD_GW = Fil2Dict(f"../{config['data_files']['asd_gene_weights_dn']}")
ASD_SC_Bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, ASD_GW)
ASD_SC_Bias = add_class(ASD_SC_Bias, ClusterAnn)
ASD_SC_Bias.to_csv("../results/CT_Z2/ASD_Spark61_DN.csv")
print(f"ASD bias: {ASD_SC_Bias.shape}, top cluster: {ASD_SC_Bias.index[0]}, "
      f"EFFECT = {ASD_SC_Bias['EFFECT'].iloc[0]:.4f}")

# %%
# DDD bias (204 DN genes)
DDD_GW = Fil2Dict(f"../{config['data_files']['ddd_gene_weights_dn']}")
DDD_SC_Bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, DDD_GW)
DDD_SC_Bias = add_class(DDD_SC_Bias, ClusterAnn)
DDD_SC_Bias.to_csv("../results/CT_Z2/DDD_293_ExcludeASD_DN.csv")
print(f"DDD bias: {DDD_SC_Bias.shape}, top cluster: {DDD_SC_Bias.index[0]}, "
      f"EFFECT = {DDD_SC_Bias['EFFECT'].iloc[0]:.4f}")

# %% [markdown]
# ## 5. Bias Boxplots by Cell Class

# %%
def plot_bias_by_class(DF, Class2Cluster, title='Bias Across Cell Classes'):
    """Horizontal boxplot of bias EFFECT sorted by median, one box per class."""
    Class = sorted(Class2Cluster.keys())
    dat, medians = [], []
    for _CT in Class:
        subdf = DF[DF["class_id_label"] == _CT]
        vals = subdf["EFFECT"].dropna().values
        dat.append(vals)
        medians.append(np.median(vals) if len(vals) > 0 else 0)

    order = np.argsort(medians)
    fig, ax = plt.subplots(dpi=240, figsize=(8, 8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    bp = ax.boxplot([dat[i] for i in order],
                    tick_labels=[Class[i] for i in order],
                    vert=False, patch_artist=True)
    colors = sns.color_palette("muted", len(Class))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Bias (EFFECT)')
    ax.set_title(title, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_bias_by_class(ASD_SC_Bias, Class2Cluster, title='ASD Bias by Cell Class')
plot_bias_by_class(DDD_SC_Bias, Class2Cluster, title='DDD Bias by Cell Class')

# %% [markdown]
# ## 6. ASD vs DDD Residual Analysis
#
# Fit linear model: ASD_EFFECT ~ DDD_EFFECT, then examine residuals.
# Positive residuals = clusters with higher ASD bias than expected from DDD.

# %%
merged_data = merge_bias_datasets(ASD_SC_Bias, DDD_SC_Bias, suffixes=('_ASD', '_DDD'),
                                  cols1=['Rank', 'EFFECT'], cols2=['Rank', 'EFFECT'])
print(f"Merged: {merged_data.shape}")

# %%
# Define cell class groups for comparison
class_groups = {
    'CNU_LGE_GABA': '09 CNU-LGE GABA',
    'IT_ET_Glut': '01 IT-ET Glut',
    'NP_CT_L6b_Glut': '02 NP-CT-L6b Glut',
    'CTX_CGE_GABA': '06 CTX-CGE GABA',
    'CTX_MGE_GABA': '07 CTX-MGE GABA',
    'TH_Glut': '18 TH Glut',
}
cluster_dict = {}
for name, class_label in class_groups.items():
    cluster_dict[name] = [x for x in ClusterAnn[ClusterAnn['class_id_label'] == class_label].index
                          if x in merged_data.index]

palette = ["orange", "green", "purple", "red", "blue", "yellow"]
ref = "CNU_LGE_GABA"
pairwise = [(ref, g) for g in class_groups if g != ref]
_ = cluster_residual_boxplot(merged_data, cluster_dict, metric="residual",
                             palette=palette, pairwise_tests=pairwise)

# %% [markdown]
# ## 7. Permutation Test (Gene Label Shuffling)
#
# Shuffle gene labels between ASD and DDD 1000 times.
# For each permutation, recompute bias and residuals.

# %%
CACHE_FILE = "../results/cache/CT_permutation_residuals.npz"
os.makedirs("../results/cache", exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"Loading cached permutation results from {CACHE_FILE}")
    cached = np.load(CACHE_FILE, allow_pickle=True)
    perm_residuals = cached['residuals']
    perm_index = cached['index']
    n_perms = perm_residuals.shape[0]
    print(f"  {n_perms} permutations, {len(perm_index)} clusters")
else:
    import random
    n_perms = 1000
    all_genes = list(ASD_GW.keys()) + list(DDD_GW.keys())
    all_weights = list(ASD_GW.values()) + list(DDD_GW.values())
    n_asd = len(ASD_GW)

    perm_residuals_list = []
    perm_index = None
    for i in range(n_perms):
        random.seed(i)
        shuffled = all_genes.copy()
        random.shuffle(shuffled)
        perm_asd = dict(zip(shuffled[:n_asd], all_weights[:n_asd]))
        perm_ddd = dict(zip(shuffled[n_asd:], all_weights[n_asd:]))

        perm_asd_bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, perm_asd)
        perm_ddd_bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, perm_ddd)
        perm_merged = merge_bias_datasets(perm_asd_bias, perm_ddd_bias,
                                          suffixes=('_ASD', '_DDD'),
                                          cols1=['Rank', 'EFFECT'], cols2=['Rank', 'EFFECT'])
        if perm_index is None:
            perm_index = perm_merged.index.values
        perm_residuals_list.append(perm_merged.loc[perm_index, 'residual'].values)

        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

    perm_residuals = np.array(perm_residuals_list)
    np.savez_compressed(CACHE_FILE, residuals=perm_residuals, index=perm_index)
    print(f"Saved {n_perms} permutations to {CACHE_FILE}")

# %% [markdown]
# ### Permutation Null Distribution for Top CNU-LGE GABA Clusters

# %%
def plot_permutation_null(obs_residuals, perm_residuals, perm_index, cluster_id):
    """Plot permutation null for a single cluster's residual."""
    idx = np.where(perm_index == cluster_id)[0][0]
    obs = obs_residuals.loc[cluster_id]
    null = perm_residuals[:, idx]
    pval = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.hist(null, bins=20, color="skyblue", edgecolor="k", alpha=0.7)
    ax.axvline(obs, color="red", linestyle="--", lw=2, label=f"Observed: {obs:.3f}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"{cluster_id}\nperm p = {pval:.3g}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return pval


cnu_lge = cluster_dict.get('CNU_LGE_GABA', [])
if cnu_lge:
    cnu_residuals = merged_data.loc[cnu_lge, 'residual'].sort_values(ascending=False)
    print("Top CNU-LGE GABA residuals:")
    for ct in cnu_residuals.head(3).index:
        pval = plot_permutation_null(merged_data['residual'], perm_residuals, perm_index, ct)
        print(f"  {ct}: residual={merged_data.loc[ct, 'residual']:.4f}, perm p={pval:.4f}")

# %% [markdown]
# ## 8. Validation vs Existing Results

# %%
# Compare against pipeline-produced bias (same Z2 matrix and DN weights)
pipeline_file = "../results/CT_Z2/ASD_All_bias_addP_random.csv"
if os.path.exists(pipeline_file):
    pipeline_bias = pd.read_csv(pipeline_file, index_col=0)
    shared_idx = ASD_SC_Bias.index.intersection(pipeline_bias.index)
    r_pipeline = np.corrcoef(ASD_SC_Bias.loc[shared_idx, 'EFFECT'],
                             pipeline_bias.loc[shared_idx, 'EFFECT'])[0, 1]
    max_diff = np.max(np.abs(ASD_SC_Bias.loc[shared_idx, 'EFFECT'] - pipeline_bias.loc[shared_idx, 'EFFECT']))
    print(f"Pipeline comparison: {len(shared_idx)} shared clusters, Pearson r = {r_pipeline:.6f}, "
          f"max |diff| = {max_diff:.2e}")
else:
    print(f"Pipeline file not found: {pipeline_file} (run Snakefile.bias first)")

# %%
# Compare against legacy results
legacy_file = "/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.addP.csv"
if os.path.exists(legacy_file):
    legacy_bias = pd.read_csv(legacy_file, index_col=0)
    shared_idx = ASD_SC_Bias.index.intersection(legacy_bias.index)
    r_legacy = np.corrcoef(ASD_SC_Bias.loc[shared_idx, 'EFFECT'],
                           legacy_bias.loc[shared_idx, 'EFFECT'])[0, 1]
    print(f"Legacy comparison: {len(shared_idx)} shared clusters, Pearson r = {r_legacy:.6f}")

    # Top-10 side by side
    top10 = ASD_SC_Bias.head(10).index
    comparison = pd.DataFrame({
        'EFFECT_new': ASD_SC_Bias.loc[top10, 'EFFECT'],
        'EFFECT_legacy': legacy_bias.loc[top10, 'EFFECT'] if all(t in legacy_bias.index for t in top10) else np.nan,
    })
    print("\nTop-10 clusters (new vs legacy):")
    print(comparison.to_string())
else:
    print(f"Legacy file not found: {legacy_file}")

# %% [markdown]
# ## 9. Sibling Pipeline Note
#
# The sibling control analysis for cell-type bias is run via Snakemake:
#
# ```bash
# snakemake -s Snakefile.bias --configfile config/config.SC.DN.yaml --cores 10
# ```
#
# This produces:
# - `results/CT_Z2/ASD_All_bias_addP_sibling.csv` — sibling null p-values
# - `results/CT_Z2/ASD_All_bias_addP_random.csv` — random null p-values
#
# The config points to DN gene weight files under `dat/Genetics/GeneWeights_DN/`.
