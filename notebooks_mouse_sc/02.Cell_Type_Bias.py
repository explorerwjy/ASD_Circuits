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
# specificity matrix, then tests whether ASD-specific bias residuals
# (relative to DDD) are enriched in striatal CNU-LGE GABA clusters.

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import yaml
sys.path.insert(1, '../src')
from ASD_Circuits import *
from plot import cluster_residual_boxplot

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)
ProjDIR = config["ProjDIR"]

# %% [markdown]
# ## 1. Load Data

# %%
# Cluster annotations
ClusterAnn = pd.read_csv(f"../{config['data_files']['mouse_ct_annotation']}", index_col="cluster_id_label")

# Cell type hierarchy (class → subclass → cluster mapping)
CellTypesDF = pd.read_csv(f"../{config['data_files']['cell_type_hierarchy']}")
Class2Cluster = {}
for _, row in CellTypesDF.iterrows():
    _cluster, _class = row.iloc[0], row.iloc[1]
    Class2Cluster.setdefault(_class, []).append(_cluster)
print(f"Loaded {len(ClusterAnn)} clusters, {len(Class2Cluster)} classes")

# %%
# Z2 expression specificity matrix (produced by scripts/build_celltype_z2_matrix.py)
MouseSC_Z2 = pd.read_parquet(f"../{config['analysis_types']['CT_Z2']['expr_matrix']}")
print(f"Z2 matrix: {MouseSC_Z2.shape}")

# %% [markdown]
# ## 2. Compute ASD and DDD Bias

# %%
# ASD bias (60 DN genes)
ASD_GW = Fil2Dict(f"../{config['data_files']['asd_gene_weights_dn']}")
ASD_SC_Bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, ASD_GW)
ASD_SC_Bias = add_class(ASD_SC_Bias, ClusterAnn)

os.makedirs("../results/CT_Z2", exist_ok=True)
ASD_SC_Bias.to_csv("../results/CT_Z2/ASD_Spark61_DN.csv")
print(f"ASD bias: {ASD_SC_Bias.shape}, top EFFECT = {ASD_SC_Bias['EFFECT'].iloc[0]:.4f}")

# %%
# DDD bias (293 genes, excluding ASD overlap, DN)
DDD_GW = Fil2Dict(f"../{config['data_files']['ddd_gene_weights_dn']}")
DDD_SC_Bias = MouseCT_AvgZ_Weighted(MouseSC_Z2, DDD_GW)
DDD_SC_Bias = add_class(DDD_SC_Bias, ClusterAnn)
DDD_SC_Bias.to_csv("../results/CT_Z2/DDD_293_ExcludeASD_DN.csv")
print(f"DDD bias: {DDD_SC_Bias.shape}, top EFFECT = {DDD_SC_Bias['EFFECT'].iloc[0]:.4f}")

# %% [markdown]
# ## 3. Bias Boxplots by Cell Class

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
# ## 4. ASD vs DDD Residual Analysis
#
# Fit a linear model: ASD_EFFECT ~ DDD_EFFECT, then examine residuals.
# Positive residuals = clusters with higher ASD bias than expected from DDD.

# %%
merged_data = merge_bias_datasets(ASD_SC_Bias, DDD_SC_Bias, suffixes=('_ASD', '_DDD'),
                                  cols1=['Rank', 'EFFECT'], cols2=['Rank', 'EFFECT'])
results_df = fit_structure_bias_linear_model(merged_data, metric='EFFECT', suffixes=('_ASD', '_DDD'))
print(f"Merged: {results_df.shape}")

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
                          if x in results_df.index]

palette = ["orange", "green", "purple", "red", "blue", "yellow"]
ref = "CNU_LGE_GABA"
pairwise = [(ref, g) for g in class_groups if g != ref]
_ = cluster_residual_boxplot(results_df, cluster_dict, metric="residual",
                             palette=palette, pairwise_tests=pairwise)

# %% [markdown]
# ## 5. Permutation Test (Gene Label Shuffling)
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
        perm_result = fit_structure_bias_linear_model(perm_merged, metric='EFFECT',
                                                      suffixes=('_ASD', '_DDD'))
        if perm_index is None:
            perm_index = perm_result.index.values
        perm_residuals_list.append(perm_result.loc[perm_index, 'residual'].values)

        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

    perm_residuals = np.array(perm_residuals_list)
    np.savez_compressed(CACHE_FILE, residuals=perm_residuals, index=perm_index)
    print(f"Saved {n_perms} permutations to {CACHE_FILE}")

# %% [markdown]
# ## 6. Permutation Null Distribution for Individual Clusters

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


# Example: CNU-LGE GABA clusters with largest residuals
cnu_lge = cluster_dict.get('CNU_LGE_GABA', [])
if cnu_lge:
    cnu_residuals = results_df.loc[cnu_lge, 'residual'].sort_values(ascending=False)
    print("Top CNU-LGE GABA residuals:")
    for ct in cnu_residuals.head(3).index:
        pval = plot_permutation_null(results_df['residual'], perm_residuals, perm_index, ct)
        print(f"  {ct}: residual={results_df.loc[ct, 'residual']:.4f}, perm p={pval:.4f}")
