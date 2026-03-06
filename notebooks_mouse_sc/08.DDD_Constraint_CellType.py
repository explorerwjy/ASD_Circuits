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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *
from plot import *

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# Load config and cell type expression matrix
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CT_BiasMat = pd.read_parquet(f"../{config['analysis_types']['CT_Z2']['expr_matrix']}")
CT_Anno = pd.read_csv(os.path.join(ProjDIR, "dat/MouseCT_Cluster_Anno.csv"), index_col="cluster_id_label")

# %%
# Load cell type bias results (from Snakefile.bias pipeline)
ASD_SC_Bias = pd.read_csv(os.path.join(ProjDIR, "results/CT_Z2/ASD_All_bias_addP_sibling.csv"), index_col=0)
DDD_SC_Bias = pd.read_csv(os.path.join(ProjDIR, "results/CT_Z2/DDD_293_bias_addP_sibling.csv"), index_col=0)
DDD_rmASD_SC_Bias = pd.read_csv(os.path.join(ProjDIR, "results/CT_Z2/DDD_293_ExcludeASD_bias_addP_sibling.csv"), index_col=0)

# Load ASD gene weights (for filtering DDD genes)
ASD_GW = Fil2Dict(os.path.join(ProjDIR, "dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw"))
ASD_GENES = list(ASD_GW.keys())

# %% [markdown]
# # Section 1: DDD vs ASD -- Cell Type Level

# %% [markdown]
# ## 1.1 DDD vs ASD cell type correlation

# %%
plot_correlation_scatter_mouseCT(DDD_SC_Bias, ASD_SC_Bias,
                                 name1="DD Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 1.2 DDD (exclude ASD) vs ASD cell type correlation

# %%
plot_correlation_scatter_mouseCT(DDD_rmASD_SC_Bias, ASD_SC_Bias,
                                 name1="DD (ASD Excluded) Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 1.3 Cell type cluster definitions

# %%
CT_COLS = ['Rank', 'EFFECT', 'class_id_label', 'subclass_id_label', 'CCF_broad.freq', 'CCF_acronym.freq']
ct_merged_data = merge_bias_datasets(ASD_SC_Bias, DDD_rmASD_SC_Bias,
                                     suffixes=('_ASD', '_DD'), cols1=CT_COLS)

# Define cell type clusters (used throughout CT analyses)
CNU_LGE_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '09 CNU-LGE GABA'].index if x in ct_merged_data.index]
IT_ET_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '01 IT-ET Glut'].index if x in ct_merged_data.index]
NP_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '02 NP-CT-L6b Glut'].index if x in ct_merged_data.index]
CGE_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '06 CTX-CGE GABA'].index if x in ct_merged_data.index]
MGE_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '07 CTX-MGE GABA'].index if x in ct_merged_data.index]

D1D2_labels = ['061 STR D1 Gaba', '062 STR D2 Gaba']
STR_D1D2 = [idx for idx in CT_Anno[CT_Anno['subclass_id_label'].isin(D1D2_labels)].index if idx in ct_merged_data.index]
Other_LGE = [idx for idx in CNU_LGE_Cluster if idx not in STR_D1D2]

HIP = ['016 CA1-ProS Glut', '017 CA3 Glut']
HIP_Glut = [x for x in CT_Anno[CT_Anno['subclass_id_label'].isin(HIP)].index if x in ct_merged_data.index]

AMY = ['012 MEA Slc17a7 Glut', '013 COAp Grxcr2 Glut', '014 LA-BLA-BMA-PA Glut', '015 ENTmv-PA-COAp Glut']
AMY_Glut = [x for x in CT_Anno[CT_Anno['subclass_id_label'].isin(AMY)].index if x in ct_merged_data.index]
Other_IT_ET = [x for x in IT_ET_Cluster if x not in AMY_Glut and x not in HIP_Glut]

RU_Cluster = [x for x in CT_Anno[CT_Anno['subclass_id_label'] == '152 RE-Xi Nox4 Glut'].index if x in ct_merged_data.index]
PF_Cluster = [x for x in CT_Anno[CT_Anno['subclass_id_label'] == '154 PF Fzd5 Glut'].index if x in ct_merged_data.index]
RU_PF = RU_Cluster + PF_Cluster
Other_TH_Cluster = [x for x in CT_Anno[CT_Anno['class_id_label'] == '18 TH Glut'].index if x in ct_merged_data.index and x not in RU_PF]

AMY_HYA_Glut = [x for x in CT_Anno[CT_Anno['class_id_label'] == '13 CNU-HYa Glut'].index if x in ct_merged_data.index]
AMY_HYA_GABA = [x for x in CT_Anno[CT_Anno['class_id_label'] == '11 CNU-HYa GABA'].index if x in ct_merged_data.index]

# Shared cluster dict and palette for boxplots
cluster_dict_main = {
    "D1/D2 MSN": STR_D1D2,
    "CNU_LGE_GABA (Other)": Other_LGE,
    "PF_RE_TH_Glut": RU_PF,
    "TH_Glut (Other)": Other_TH_Cluster,
    "CNU_HYA_Glut": AMY_HYA_Glut,
    "CNU_HYA_GABA": AMY_HYA_GABA,
    "CTX_CGE_GABA": CGE_Cluster,
    "IT_ET_Glut": IT_ET_Cluster,
    "NP_CT_L6b_Glut": NP_Cluster,
    "CTX_MGE_GABA": MGE_Cluster,
}
palette_main = ["orange", "green", "purple", "red", "blue", "gold",
                "pink", "teal", "sienna", "indigo"]

pairwise_tests_main = [
    ("D1/D2 MSN", "CNU_LGE_GABA (Other)"),
    ("PF_RE_TH_Glut", "TH_Glut (Other)"),
    ("PF_RE_TH_Glut",  ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
    ("D1/D2 MSN", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
    ("CNU_HYA_Glut", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
    ("CNU_HYA_GABA", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
]
cortical_ref_bracket = [{"groups": ["CTX_CGE_GABA", "IT_ET_Glut", "NP_CT_L6b_Glut", "CTX_MGE_GABA"],
                          "label": "Cortical Reference"}]

# %% [markdown]
# ## 1.4 Residual boxplot (DDD excl ASD vs ASD)

# %%
_ = cluster_residual_boxplot(
    ct_merged_data, cluster_dict_main, metric="residual",
    palette=palette_main, figsize=(12, 8),
    pairwise_tests=pairwise_tests_main,
    p_adjust="fdr_bh", p_style="stars", show_ns=False,
    wrap_xticks=True, wrap_len=16, point_size=2.2, point_alpha=0.16,
    group_brackets=cortical_ref_bracket
)

# %% [markdown]
# ## 1.5 All-class residual boxplot

# %%
all_class_labels = sorted(ct_merged_data["class_id_label"].unique())
cluster_dict_all = {
    label: [idx for idx in CT_Anno[CT_Anno['class_id_label'] == label].index if idx in ct_merged_data.index]
    for label in all_class_labels
}
palette_all = sns.color_palette("tab20", len(cluster_dict_all))

_ = cluster_residual_boxplot(
    ct_merged_data, cluster_dict_all, metric="residual",
    palette=palette_all, figsize=(max(12, len(cluster_dict_all) * 0.7), 8),
    pairwise_tests=[]
)

# %% [markdown]
# # Section 2: Constraint Gene Analysis -- Cell Type Level

# %% [markdown]
# ## 2.1 pLI >= 0.99 cell type analysis

# %%
# Load constraint gene weights from saved files
constraint_gw_top_PLI = Fil2Dict(os.path.join(ProjDIR, "dat/Genetics/GeneWeights/constraint_top_decile_PLI.gw"))

pLI_SC_Bias = MouseCT_AvgZ_Weighted(CT_BiasMat, constraint_gw_top_PLI)
pLI_SC_Bias = add_class(pLI_SC_Bias, CT_Anno)
pLI_SC_Bias.to_csv(os.path.join(ProjDIR, "results/CT_Z2/pLI_SC_Bias.csv"))

plot_correlation_scatter_mouseCT(pLI_SC_Bias, ASD_SC_Bias, name1="Constrained Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)
plot_correlation_scatter_mouseCT(pLI_SC_Bias, DDD_SC_Bias, name1="Constrained Cell Type Bias", name2="DD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 2.2 LOEUF top 25% cell type analysis

# %%
LOEUF25_SC_Bias = pd.read_csv(os.path.join(ProjDIR, "results/CT_Z2/Constraint_top25_LOEUF_bias_addP_random.csv"), index_col=0)

plot_correlation_scatter_mouseCT(LOEUF25_SC_Bias, ASD_SC_Bias,
                                 name1="Constrained Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=120)
plot_correlation_scatter_mouseCT(LOEUF25_SC_Bias, DDD_rmASD_SC_Bias,
                                 name1="Constrained Cell Type Bias", name2="DD (exclude ASD) \nCell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=120)

# %% [markdown]
# ## 2.3 Cell type residual: ASD vs Constrained (LOEUF top 25%)

# %%
ct_merged_data_LOEUF25 = merge_bias_datasets(ASD_SC_Bias, LOEUF25_SC_Bias,
                                              suffixes=('_ASD', '_Constrained'), cols1=CT_COLS)

_ = cluster_residual_boxplot(
    ct_merged_data_LOEUF25, cluster_dict_main, metric="residual",
    palette=palette_main, figsize=(12, 8),
    pairwise_tests=[("D1/D2 MSN", "CNU_LGE_GABA (Other)"),
                    ("PF_RE_TH_Glut", "TH_Glut (Other)"),
                    ("D1/D2 MSN", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"])],
    p_adjust="fdr_bh", p_style="stars", show_ns=False,
    wrap_xticks=True, wrap_len=16, point_size=2.2, point_alpha=0.16,
    group_brackets=cortical_ref_bracket
)

# %% [markdown]
# ## 2.4 Cell type correlation summary

# %%
merged_pLI_ASD_CT = pd.merge(pLI_SC_Bias[['EFFECT']], ASD_SC_Bias[['EFFECT']],
                              left_index=True, right_index=True, suffixes=('_pLI', '_ASD'))
corr_ct_pLI_ASD, pval_ct_pLI_ASD = pearsonr(merged_pLI_ASD_CT['EFFECT_pLI'], merged_pLI_ASD_CT['EFFECT_ASD'])

merged_pLI_DDD_CT = pd.merge(pLI_SC_Bias[['EFFECT']], DDD_SC_Bias[['EFFECT']],
                              left_index=True, right_index=True, suffixes=('_pLI', '_DD'))
corr_ct_pLI_DDD, pval_ct_pLI_DDD = pearsonr(merged_pLI_DDD_CT['EFFECT_pLI'], merged_pLI_DDD_CT['EFFECT_DD'])

merged_LOEUF25_ASD_CT = pd.merge(LOEUF25_SC_Bias[['EFFECT']], ASD_SC_Bias[['EFFECT']],
                                  left_index=True, right_index=True, suffixes=('_LOEUF25', '_ASD'))
corr_ct_LOEUF25_ASD, pval_ct_LOEUF25_ASD = pearsonr(merged_LOEUF25_ASD_CT['EFFECT_LOEUF25'], merged_LOEUF25_ASD_CT['EFFECT_ASD'])

merged_LOEUF25_DDD_CT = pd.merge(LOEUF25_SC_Bias[['EFFECT']], DDD_SC_Bias[['EFFECT']],
                                  left_index=True, right_index=True, suffixes=('_LOEUF25', '_DD'))
corr_ct_LOEUF25_DDD, pval_ct_LOEUF25_DDD = pearsonr(merged_LOEUF25_DDD_CT['EFFECT_LOEUF25'], merged_LOEUF25_DDD_CT['EFFECT_DD'])

print("=" * 60)
print("Cell Type Bias Correlations")
print("=" * 60)
print(f"\npLI>=0.99:")
print(f"  ASD correlation:  r = {corr_ct_pLI_ASD:.3f}, p = {pval_ct_pLI_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_ct_pLI_DDD:.3f}, p = {pval_ct_pLI_DDD:.2e}")
print(f"\nLOEUF top 25%:")
print(f"  ASD correlation:  r = {corr_ct_LOEUF25_ASD:.3f}, p = {pval_ct_LOEUF25_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_ct_LOEUF25_DDD:.3f}, p = {pval_ct_LOEUF25_DDD:.2e}")
print("=" * 60)

# %%
