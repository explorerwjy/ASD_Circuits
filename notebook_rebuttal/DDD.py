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
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *
from plot import *

try:
    os.chdir(f"{ProjDIR}/notebook_rebuttal/")
    print(f"Current working directory: {os.getcwd()}")
except Exception as e:
    print(f"Error: {e}")

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# Load config and expression matrices
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

STR_BiasMat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")
STR_Anno = STR2Region()

CT_BiasMat = pd.read_parquet(f"../{config['analysis_types']['CT_Z2']['expr_matrix']}")
CT_Anno = pd.read_csv(ProjDIR + "dat/MouseCT_Cluster_Anno.csv", index_col="cluster_id_label")

# %%
# Load connectivity matrices and null CCS
ScoreMatDir = "/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v3/"
WeightMat = pd.read_csv(ScoreMatDir + "WeightMat.Ipsi.csv", index_col=0)
IpsiInfoMat = pd.read_csv(ScoreMatDir + "InfoMat.Ipsi.csv", index_col=0)

DIR = "/home/jw3514/Work/ASD_Circuits/scripts/RankScores/"
Cont_Distance = np.load(f"{DIR}/RankScore.Ipsi.Cont.npy")
topNs = list(range(200, 5, -1))

# %%
# Load ASD bias and gene weights
Spark_ASD_STR_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col=0)
Spark_ASD_STR_Bias["Region"] = Spark_ASD_STR_Bias["REGION"]
ASD_GW = Fil2Dict(ProjDIR + "dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw")
ASD_GENES = list(ASD_GW.keys())

# Load DDD bias and gene weights
DDD_GW = Fil2Dict(config["gene_sets"]["DDD_293"]["geneweights"])
DDD_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, DDD_GW)
DDD_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in DDD_STR_Bias.index]

# DDD excluding ASD genes
DDD_GW_filt_ASD = {k: v for k, v in DDD_GW.items() if k not in ASD_GENES}
print(f"DDD genes: {len(DDD_GW)}, after excluding ASD: {len(DDD_GW_filt_ASD)}")
DDD_rmASD_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, DDD_GW_filt_ASD)
DDD_rmASD_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in DDD_rmASD_STR_Bias.index]

# Save gene weight files
Dict2Fil(DDD_GW, ProjDIR + "/dat/Genetics/GeneWeights/DDD.top293.gw")
Dict2Fil(DDD_GW_filt_ASD, ProjDIR + "/dat/Genetics/GeneWeights/DDD.top245.ExcludeASD.gw")

# %%
# Load cell type bias data
ASD_SC_Bias = pd.read_csv(ProjDIR + "/results/CT_Z2/ASD_All_bias_addP_sibling.csv", index_col=0)
DDD_SC_Bias = pd.read_csv(ProjDIR + "/results/CT_Z2/DDD_293_bias_addP_sibling.csv", index_col=0)
DDD_rmASD_SC_Bias = pd.read_csv(ProjDIR + "/results/CT_Z2/DDD_293_ExcludeASD_bias_addP_sibling.csv", index_col=0)

# %%
# Load circuit structures
GENCIC = pd.read_csv('../results/GENCIC_MouseSTRBias.csv', index_col=0)
Circuit_STRs = GENCIC[GENCIC["Circuits.46"] == 1]["Structure"].values

# %% [markdown]
# # Section 1: DDD vs ASD -- Structure Level

# %% [markdown]
# ## 1.1 All DDD genes: CCS plot

# %%
score_DDD_all = calculate_circuit_scores(DDD_STR_Bias, IpsiInfoMat, sort_by="EFFECT")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1, 1, dpi=480, figsize=(12, 6), facecolor='none')
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1
ax1.plot(topNs, score_DDD_all, color='#1f77b4', marker="o", markersize=5, lw=1,
         ls="dashed", label="DD", alpha=0.5)

cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50 - BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50 + BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)
ax1.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

# %% [markdown]
# ## 1.2 DDD vs ASD correlation

# %%
merged_data = merge_bias_datasets(Spark_ASD_STR_Bias, DDD_STR_Bias, suffixes=('_ASD', '_DD'))
plot_structure_bias_comparison(merged_data, suffixes=('_ASD', '_DD'), metric="EFFECT")

# %% [markdown]
# ## 1.3 DDD (exclude ASD genes) vs ASD correlation

# %%
merged_data2 = merge_bias_datasets(Spark_ASD_STR_Bias, DDD_rmASD_STR_Bias, suffixes=('_ASD', '_DD_ExcludeASD'))
plot_structure_bias_comparison(merged_data2, suffixes=('_ASD', '_DD_ExcludeASD'), metric='EFFECT')

# %% [markdown]
# ## 1.4 Residual structures with bootstrap CI

# %%
DDD_CI_path = "../results/Bootstrap_bias/DDD_ExomeWide/Residual_CI/DDD_ExomeWide.Residual_CI_95.csv"
DDD_residual_ci_df = pd.read_csv(DDD_CI_path, index_col=0)
merged_data_eval = merged_data2[merged_data2.index.isin(Circuit_STRs)]
top_diff_ci = plot_top_residual_structures_with_CI(merged_data_eval, DDD_residual_ci_df, top_n=20, top_threshold=40,
                                                   name1="ASD", name2="DD_ExcludeASD", figsize=(6, 8))

# %% [markdown]
# # Section 2: DDD vs ASD -- Cell Type Level

# %% [markdown]
# ## 2.1 DDD vs ASD cell type correlation

# %%
plot_correlation_scatter_mouseCT(DDD_SC_Bias, ASD_SC_Bias,
                                 name1="DD Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 2.2 DDD (exclude ASD) vs ASD cell type correlation

# %%
plot_correlation_scatter_mouseCT(DDD_rmASD_SC_Bias, ASD_SC_Bias,
                                 name1="DD (ASD Excluded) Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 2.3 Cell type cluster definitions

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
    ("D1/D2 MSN", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
    ("CNU_HYA_Glut", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
    ("CNU_HYA_GABA", ["CTX_CGE_GABA", "CTX_MGE_GABA", "NP_CT_L6b_Glut", "IT_ET_Glut"]),
]

# %% [markdown]
# ## 2.4 Residual boxplot (DDD excl ASD vs ASD)

# %%
_ = cluster_residual_boxplot(
    ct_merged_data, cluster_dict_main, metric="residual",
    palette=palette_main, figsize=(12, 8),
    pairwise_tests=pairwise_tests_main,
    p_adjust="fdr_bh", p_style="stars", show_ns=False,
    wrap_xticks=True, wrap_len=16, point_size=2.2, point_alpha=0.16
)

# %% [markdown]
# ## 2.5 All-class residual boxplot

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
# # Section 3: Constraint Gene Analysis

# %%
# Load gnomAD v4 constraint data
gnomad4 = pd.read_csv("/home/jw3514/Work/data/gnomad/gnomad.v4.0.constraint_metrics.tsv", sep="\t")
gnomad4 = gnomad4[(gnomad4["transcript"].str.contains('ENST'))]
gnomad4 = gnomad4[gnomad4["mane_select"] == True]
for i, row in gnomad4.iterrows():
    gnomad4.loc[i, "Entrez"] = int(GeneSymbol2Entrez.get(row["gene"], 0))

# %% [markdown]
# ## 3.1 pLI >= 0.99 analysis

# %%
gnomad4_top_PLI = gnomad4[gnomad4["lof.pLI"] > 0.99]
print(f"pLI>=0.99 genes: {gnomad4_top_PLI.shape[0]}")
constraint_gw_top_PLI = dict(zip(gnomad4_top_PLI["Entrez"], [1] * len(gnomad4_top_PLI)))
Dict2Fil(constraint_gw_top_PLI, ProjDIR + "/dat/Genetics/GeneWeights/constraint_top_decile_PLI.gw")

constraint_top_PLI_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, constraint_gw_top_PLI)
constraint_top_PLI_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in constraint_top_PLI_STR_Bias.index]

# Structure bias comparisons
merged_data_ASD_Constraint_PLI = merge_bias_datasets(Spark_ASD_STR_Bias, constraint_top_PLI_STR_Bias, suffixes=('_ASD', '_Constrained'))
plot_structure_bias_comparison(merged_data_ASD_Constraint_PLI, suffixes=('_ASD', '_Constrained'), metric="EFFECT", show_region_legend=True)

merged_data_DDD_Constraint_PLI = merge_bias_datasets(DDD_STR_Bias, constraint_top_PLI_STR_Bias, suffixes=('_DD', '_Constrained'))
plot_structure_bias_comparison(merged_data_DDD_Constraint_PLI, suffixes=('_DD', '_Constrained'), metric="EFFECT")

# %%
# CCS for pLI constraint genes
score_Constraint_PLI = calculate_circuit_scores(constraint_top_PLI_STR_Bias, IpsiInfoMat, sort_by="EFFECT")

# %%
# Residual: ASD vs Constrained (pLI)
merged_data_eval = merged_data_ASD_Constraint_PLI[merged_data_ASD_Constraint_PLI.index.isin(Circuit_STRs)]
_ = plot_top_residual_structures_with_CI(merged_data_eval, top_n=20, top_threshold=40,
                                         name1="ASD", name2="Constrained", figsize=(6, 6))

# Residual: DDD vs Constrained (pLI)
merged_data_eval = merged_data_DDD_Constraint_PLI[merged_data_DDD_Constraint_PLI.index.isin(Circuit_STRs)]
_ = plot_top_residual_structures_with_CI(merged_data_eval, top_n=20, top_threshold=40,
                                         name1="DD", name2="Constrained", figsize=(6, 8))

# %%
# Cell type: pLI
pLI_SC_Bias = MouseCT_AvgZ_Weighted(CT_BiasMat, constraint_gw_top_PLI)
pLI_SC_Bias = add_class(pLI_SC_Bias, CT_Anno)
pLI_SC_Bias.to_csv(ProjDIR + "/results/CT_Z2/pLI_SC_Bias.csv")

plot_correlation_scatter_mouseCT(pLI_SC_Bias, ASD_SC_Bias, name1="Constrained Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)
plot_correlation_scatter_mouseCT(pLI_SC_Bias, DDD_SC_Bias, name1="Constrained Cell Type Bias", name2="DD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=240)

# %% [markdown]
# ## 3.2 LOEUF top 25% analysis

# %%
bottom_25_percent_threshold = gnomad4["lof.oe_ci.upper"].quantile(0.25)
gnomad4_bottom25 = gnomad4[gnomad4["lof.oe_ci.upper"] <= bottom_25_percent_threshold]
columns_to_keep_g4 = ["Entrez", "gene", "lof.pLI", "lof.z_score", "lof.oe_ci.upper"]
gnomad4_bottom25 = gnomad4_bottom25[columns_to_keep_g4].copy()
gnomad4_bottom25["Entrez"] = gnomad4_bottom25["Entrez"].astype(int)
gnomad4_bottom25 = gnomad4_bottom25[gnomad4_bottom25["Entrez"] != 0]
gnomad4_bottom25 = gnomad4_bottom25.sort_values(by="lof.oe_ci.upper", ascending=True)
print(f"LOEUF top 25% genes: {gnomad4_bottom25.shape[0]}")

constraint_gw_top_LOEUF25 = dict(zip(gnomad4_bottom25["Entrez"], [1] * len(gnomad4_bottom25)))
Dict2Fil(constraint_gw_top_LOEUF25, ProjDIR + "/dat/Genetics/GeneWeights/constraint_top25_LOEUF.gw")

constraint_top_LOEUF25_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, constraint_gw_top_LOEUF25)
constraint_top_LOEUF25_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in constraint_top_LOEUF25_STR_Bias.index]

# Compare with ASD
merged_data_ASD_Constraint_LOEUF25 = merge_bias_datasets(Spark_ASD_STR_Bias, constraint_top_LOEUF25_STR_Bias, suffixes=('_ASD', '_Constrained'))
plot_structure_bias_comparison(merged_data_ASD_Constraint_LOEUF25, suffixes=('_ASD', '_Constrained'), metric="EFFECT")

# Compare with DDD (exclude ASD)
merged_data_DDD_Constraint_LOEUF25 = merge_bias_datasets(DDD_rmASD_STR_Bias, constraint_top_LOEUF25_STR_Bias, suffixes=('_DD (exclude ASD)', '_Constrained'))
plot_structure_bias_comparison(merged_data_DDD_Constraint_LOEUF25, suffixes=('_DD (exclude ASD)', '_Constrained'), metric="EFFECT")

# %%
# Residual: ASD vs Constrained (LOEUF top 25%)
merged_data_eval_LOEUF25 = merged_data_ASD_Constraint_LOEUF25[merged_data_ASD_Constraint_LOEUF25.index.isin(Circuit_STRs)]
_ = plot_top_residual_structures_with_CI(merged_data_eval_LOEUF25, top_n=20, top_threshold=40,
                                         name1="ASD", name2="Constrained", figsize=(6, 6))

# %%
# Load LOEUF25 bootstrap CI and plot with error bars
LOEUF25_CI_path = "../results/Bootstrap_bias/LOEUF25/Residual_CI/LOEUF25.Residual_CI_95.csv"
LOEUF25_residual_ci_df = pd.read_csv(LOEUF25_CI_path, index_col=0)
_ = plot_top_residual_structures_with_CI(merged_data_eval_LOEUF25, LOEUF25_residual_ci_df, top_n=20, top_threshold=40,
                                         name1="ASD", name2="Constrained", figsize=(6, 8))

# %% [markdown]
# ## 3.3 CCS comparison: DD (excl ASD) vs Constrained

# %%
# Calculate all circuit scores (once)
score_ASD = calculate_circuit_scores(Spark_ASD_STR_Bias, IpsiInfoMat, sort_by="EFFECT")
score_DDD = calculate_circuit_scores(DDD_STR_Bias, IpsiInfoMat, sort_by="EFFECT")
score_DDD_rmASD = calculate_circuit_scores(DDD_rmASD_STR_Bias, IpsiInfoMat, sort_by="EFFECT")
score_Constraint_LOEUF25 = calculate_circuit_scores(constraint_top_LOEUF25_STR_Bias, IpsiInfoMat, sort_by="EFFECT")

# %%
# CCS plot: DD (excl ASD) vs Constrained (pLI)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1, 1, dpi=480, figsize=(10, 6), facecolor='none')
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1
cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50 - BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50 + BarLen, axis=0)

ax1.plot(topNs, score_DDD_rmASD, color="#ff7f0e", marker="o", markersize=5, lw=1,
         ls="dashed", label="DD (exclude ASD)", alpha=0.9)
ax1.plot(topNs, score_Constraint_PLI, color="#2ca02c", marker="o", markersize=5, lw=1,
         ls="dashed", label="Constrained Genes (pLI>=0.99)", alpha=0.9)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)
ax1.legend(fontsize=13, loc='upper right', frameon=True)
plt.tight_layout()

# %%
# Cell type: LOEUF top 25%
LOEUF25_SC_Bias = pd.read_csv(ProjDIR + "/results/CT_Z2/Constraint_top25_LOEUF_bias_addP_random.csv", index_col=0)

plot_correlation_scatter_mouseCT(LOEUF25_SC_Bias, ASD_SC_Bias,
                                 name1="Constrained Cell Type Bias", name2="ASD Cell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=120)
plot_correlation_scatter_mouseCT(LOEUF25_SC_Bias, DDD_rmASD_SC_Bias,
                                 name1="Constrained Cell Type Bias", name2="DD (exclude ASD) \nCell Type Bias",
                                 effect_col1="EFFECT", effect_col2="EFFECT", dpi=120)

# %% [markdown]
# ## 3.4 Cell type residual: ASD vs Constrained (LOEUF top 25%)

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
    wrap_xticks=True, wrap_len=16, point_size=2.2, point_alpha=0.16
)

# %%
# CCS: DD (excl ASD) vs Constrained (LOEUF top 25%)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1, 1, dpi=480, figsize=(10, 6), facecolor='none')
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

ax1.plot(topNs, score_DDD_rmASD, color="#ff7f0e", marker="o", markersize=5, lw=1,
         ls="dashed", label="DD (exclude ASD)", alpha=0.9)
ax1.plot(topNs, score_Constraint_LOEUF25, color="#2ca02c", marker="o", markersize=5, lw=1,
         ls="dashed", label="Constrained Genes (LOEUF top 25%)", alpha=0.9)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)
ax1.legend(fontsize=13, loc='upper right', frameon=True)
plt.tight_layout()

# %% [markdown]
# ## 3.5 pLI vs LOEUF comparison

# %%
# Gene overlap
pLI_genes = set(constraint_gw_top_PLI.keys())
LOEUF25_genes = set(constraint_gw_top_LOEUF25.keys())

print(f"pLI>=0.99 genes: {len(pLI_genes)}")
print(f"LOEUF top 25% genes: {len(LOEUF25_genes)}")
print(f"Overlap: {len(pLI_genes & LOEUF25_genes)}")
print(f"Only in pLI: {len(pLI_genes - LOEUF25_genes)}")
print(f"Only in LOEUF top 25%: {len(LOEUF25_genes - pLI_genes)}")
print(f"Jaccard index: {len(pLI_genes & LOEUF25_genes) / len(pLI_genes | LOEUF25_genes):.3f}")

# %%
# Direct structure bias correlation: pLI vs LOEUF
merged_data_pLI_LOEUF25 = merge_bias_datasets(constraint_top_PLI_STR_Bias, constraint_top_LOEUF25_STR_Bias, suffixes=('_pLI', '_LOEUF25'))
plot_structure_bias_comparison(merged_data_pLI_LOEUF25, suffixes=('_pLI', '_LOEUF25'), metric="EFFECT")

# %%
# Summary correlations
print("=" * 60)
print("Structure Bias Correlations")
print("=" * 60)

corr_pLI_ASD, pval_pLI_ASD = pearsonr(merged_data_ASD_Constraint_PLI["EFFECT_ASD"], merged_data_ASD_Constraint_PLI["EFFECT_Constrained"])
corr_pLI_DDD, pval_pLI_DDD = pearsonr(merged_data_DDD_Constraint_PLI["EFFECT_DD"], merged_data_DDD_Constraint_PLI["EFFECT_Constrained"])
corr_LOEUF25_ASD, pval_LOEUF25_ASD = pearsonr(merged_data_ASD_Constraint_LOEUF25["EFFECT_ASD"], merged_data_ASD_Constraint_LOEUF25["EFFECT_Constrained"])
corr_LOEUF25_DDD, pval_LOEUF25_DDD = pearsonr(merged_data_DDD_Constraint_LOEUF25["EFFECT_DD (exclude ASD)"], merged_data_DDD_Constraint_LOEUF25["EFFECT_Constrained"])

print(f"\npLI>=0.99 ({len(pLI_genes)} genes):")
print(f"  ASD correlation:  r = {corr_pLI_ASD:.3f}, p = {pval_pLI_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_pLI_DDD:.3f}, p = {pval_pLI_DDD:.2e}")
print(f"\nLOEUF top 25% ({len(LOEUF25_genes)} genes):")
print(f"  ASD correlation:  r = {corr_LOEUF25_ASD:.3f}, p = {pval_LOEUF25_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_LOEUF25_DDD:.3f}, p = {pval_LOEUF25_DDD:.2e}")
print("=" * 60)

# %%
# Cell type correlations
print("=" * 60)
print("Cell Type Bias Correlations")
print("=" * 60)

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

print(f"\npLI>=0.99:")
print(f"  ASD correlation:  r = {corr_ct_pLI_ASD:.3f}, p = {pval_ct_pLI_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_ct_pLI_DDD:.3f}, p = {pval_ct_pLI_DDD:.2e}")
print(f"\nLOEUF top 25%:")
print(f"  ASD correlation:  r = {corr_ct_LOEUF25_ASD:.3f}, p = {pval_ct_LOEUF25_ASD:.2e}")
print(f"  DDD correlation:  r = {corr_ct_LOEUF25_DDD:.3f}, p = {pval_ct_LOEUF25_DDD:.2e}")
print("=" * 60)

# %%
# Side-by-side CCS: pLI vs LOEUF top 25%
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=480, figsize=(18, 6), facecolor='none')
fig.patch.set_alpha(0)
for ax in [ax1, ax2]:
    ax.patch.set_alpha(0)

ASD_color, DDD_color, rmASD_color = "#d62728", "#1f77b4", "#ff7f0e"
Constraint_color, siblings_color = "#2ca02c", "grey"

# Panel 1: pLI
ax1.plot(topNs, score_DDD, color=DDD_color, marker="o", markersize=5, lw=1, ls="dashed", label="DD", alpha=0.9)
ax1.plot(topNs, score_DDD_rmASD, color=rmASD_color, marker="o", markersize=5, lw=1, ls="dashed", label="DD (exclude ASD)", alpha=0.9)
ax1.plot(topNs, score_Constraint_PLI, color=Constraint_color, marker="o", markersize=5, lw=1, ls="dashed", label="Constrained (pLI>=0.99)", alpha=0.9)
ax1.errorbar(topNs, cont, color=siblings_color, marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank", fontsize=15)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.set_title("pLI>=0.99", fontsize=16, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)
ax1.legend(fontsize=11, loc='upper right', frameon=True)

# Panel 2: LOEUF top 25%
ax2.plot(topNs, score_DDD, color=DDD_color, marker="o", markersize=5, lw=1, ls="dashed", label="DD", alpha=0.9)
ax2.plot(topNs, score_DDD_rmASD, color=rmASD_color, marker="o", markersize=5, lw=1, ls="dashed", label="DD (exclude ASD)", alpha=0.9)
ax2.plot(topNs, score_Constraint_LOEUF25, color=Constraint_color, marker="o", markersize=5, lw=1, ls="dashed", label="Constrained (LOEUF top 25%)", alpha=0.9)
ax2.errorbar(topNs, cont, color=siblings_color, marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax2.set_xlabel("Structure Rank", fontsize=15)
ax2.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax2.set_title("LOEUF top 25%", fontsize=16, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 121)
ax2.legend(fontsize=11, loc='upper right', frameon=True)

plt.tight_layout()

# %% [markdown]
# # Section 4: Constraint Decile Analysis
#
# Correlation between ASD/DDD structure bias and constraint genes across all 10 deciles of LOEUF.
# - **Decile 1**: Most constrained (lowest LOEUF)
# - **Decile 10**: Least constrained (highest LOEUF)

# %% [markdown]
# ## 4.1 Decile correlation analysis (with p-values)

# %%
decile_results_pval = []

for decile_num in range(1, 11):
    lower_quantile = (decile_num - 1) / 10
    upper_quantile = decile_num / 10
    lower_threshold = gnomad4["lof.oe_ci.upper"].quantile(lower_quantile)
    upper_threshold = gnomad4["lof.oe_ci.upper"].quantile(upper_quantile)

    gnomad4_decile = gnomad4[
        (gnomad4["lof.oe_ci.upper"] > lower_threshold) &
        (gnomad4["lof.oe_ci.upper"] <= upper_threshold)
    ]
    columns_to_keep = ["Entrez", "gene", "lof.pLI", "lof.z_score", "lof.oe_ci.upper"]
    gnomad4_decile = gnomad4_decile[columns_to_keep].copy()
    gnomad4_decile["Entrez"] = gnomad4_decile["Entrez"].astype(int)
    gnomad4_decile = gnomad4_decile[gnomad4_decile["Entrez"] != 0]

    constraint_gw_decile = dict(zip(gnomad4_decile["Entrez"], 1 / gnomad4_decile["lof.oe_ci.upper"]))
    constraint_STR_Bias_decile = MouseSTR_AvgZ_Weighted(STR_BiasMat, constraint_gw_decile)
    constraint_STR_Bias_decile["Region"] = [STR_Anno.get(s, "Unknown") for s in constraint_STR_Bias_decile.index]

    merged_ASD = merge_bias_datasets(Spark_ASD_STR_Bias, constraint_STR_Bias_decile, suffixes=('_ASD', '_Constrained'))
    corr_ASD, pval_ASD = pearsonr(merged_ASD["EFFECT_ASD"], merged_ASD["EFFECT_Constrained"])

    merged_DDD = merge_bias_datasets(DDD_rmASD_STR_Bias, constraint_STR_Bias_decile, suffixes=('_DD', '_Constrained'))
    corr_DDD, pval_DDD = pearsonr(merged_DDD["EFFECT_DD"], merged_DDD["EFFECT_Constrained"])

    decile_results_pval.append({
        'Decile': decile_num,
        'N_genes': len(gnomad4_decile),
        'LOEUF_mean': gnomad4_decile["lof.oe_ci.upper"].mean(),
        'Correlation_ASD': corr_ASD,
        'P_value_ASD': pval_ASD,
        'Correlation_DDD': corr_DDD,
        'P_value_DDD': pval_DDD,
        'Sig_ASD': '***' if pval_ASD < 0.001 else '**' if pval_ASD < 0.01 else '*' if pval_ASD < 0.05 else 'ns',
        'Sig_DDD': '***' if pval_DDD < 0.001 else '**' if pval_DDD < 0.01 else '*' if pval_DDD < 0.05 else 'ns'
    })
    print(f"Decile {decile_num}: N={len(gnomad4_decile)}, LOEUF=[{lower_threshold:.3f}, {upper_threshold:.3f}], "
          f"Corr_ASD={corr_ASD:.3f} ({pval_ASD:.2e}), Corr_DDD={corr_DDD:.3f} ({pval_DDD:.2e})")

decile_results_df = pd.DataFrame(decile_results_pval)
decile_results_df

# %% [markdown]
# ## 4.2 Decile visualizations

# %%
# Line plot: Correlation vs Decile / Mean LOEUF
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=240)

ax1 = axes[0]
ax1.plot(decile_results_df['Decile'], decile_results_df['Correlation_ASD'],
         marker='o', markersize=8, linewidth=2, label='ASD vs Constrained', color='#1f77b4')
ax1.plot(decile_results_df['Decile'], decile_results_df['Correlation_DDD'],
         marker='s', markersize=8, linewidth=2, label='DD (excl. ASD) vs Constrained', color='#ff7f0e')
ax1.set_xlabel('Constrained Decile\n(1=Most Constrained, 10=Least Constrained)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Correlation with Structure Bias', fontsize=14, fontweight='bold')
ax1.set_title('Correlation vs Constrained Decile', fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(range(1, 11))
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax2 = axes[1]
ax2.plot(decile_results_df['LOEUF_mean'], decile_results_df['Correlation_ASD'],
         marker='o', markersize=8, linewidth=2, label='ASD vs Constrained', color='#1f77b4')
ax2.plot(decile_results_df['LOEUF_mean'], decile_results_df['Correlation_DDD'],
         marker='s', markersize=8, linewidth=2, label='DD (excl. ASD) vs Constrained', color='#ff7f0e')
ax2.set_xlabel('Mean LOEUF\n(Lower = More Constrained)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Correlation with Structure Bias', fontsize=14, fontweight='bold')
ax2.set_title('Correlation vs Mean LOEUF', fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Bar plot with significance annotations
fig, ax = plt.subplots(figsize=(14, 7), dpi=240)
x = np.arange(len(decile_results_df))
width = 0.35

bars1 = ax.bar(x - width / 2, decile_results_df['Correlation_ASD'], width,
               label='ASD vs Constrained', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width / 2, decile_results_df['Correlation_DDD'], width,
               label='DD (excl. ASD) vs Constrained', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)

for i, (idx, row) in enumerate(decile_results_df.iterrows()):
    if row['Sig_ASD'] != 'ns':
        y_pos = row['Correlation_ASD'] + (0.02 if row['Correlation_ASD'] > 0 else -0.05)
        va = 'bottom' if row['Correlation_ASD'] > 0 else 'top'
        ax.text(i - width / 2, y_pos, row['Sig_ASD'], ha='center', va=va, fontsize=10, fontweight='bold')
    if row['Sig_DDD'] != 'ns':
        y_pos = row['Correlation_DDD'] + (0.02 if row['Correlation_DDD'] > 0 else -0.05)
        va = 'bottom' if row['Correlation_DDD'] > 0 else 'top'
        ax.text(i + width / 2, y_pos, row['Sig_DDD'], ha='center', va=va, fontsize=10, fontweight='bold')

ax.set_xlabel('Constrained Decile (1=Most Constrained, 10=Least Constrained)', fontsize=14, fontweight='bold')
ax.set_ylabel('Correlation with Structure Bias', fontsize=14, fontweight='bold')
ax.set_title('Structure Bias Correlation Across Constrained Deciles\n(*** p<0.001, ** p<0.01, * p<0.05)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{i}\n({row["LOEUF_mean"]:.2f})' for i, row in decile_results_df.iterrows()], fontsize=10)
ax.legend(fontsize=12, loc='best')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Section 5: Permutation Test -- ASD vs Constrained Gene Pool

# %% [markdown]
# ## 5.1 Batch permutation (with caching)

# %%
# LOEUF top 10% gene pool for permutation
bottom_10_percent_threshold = gnomad4["lof.oe_ci.upper"].quantile(0.1)
gnomad4_bottom10 = gnomad4[gnomad4["lof.oe_ci.upper"] <= bottom_10_percent_threshold]
gnomad4_bottom10 = gnomad4_bottom10[["Entrez", "gene", "lof.pLI", "lof.z_score", "lof.oe_ci.upper"]].copy()
gnomad4_bottom10["Entrez"] = gnomad4_bottom10["Entrez"].astype(int)
gnomad4_bottom10 = gnomad4_bottom10[gnomad4_bottom10["Entrez"] != 0]
gnomad4_bottom10 = gnomad4_bottom10.sort_values(by="lof.oe_ci.upper", ascending=True)
print(f"LOEUF top 10% genes: {gnomad4_bottom10.shape[0]}")

constraint_gw = dict(zip(gnomad4_bottom10["Entrez"], 1 / gnomad4_bottom10["lof.oe_ci.upper"]))
Geneset = list(constraint_gw.keys())
Weights = list(ASD_GW.values())

# %%
cache_path = "../results/cache/DDD_constraint_permutation_10K.pkl"
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

if os.path.exists(cache_path):
    print("Loading cached permutation results...")
    with open(cache_path, "rb") as f:
        tmp_bias_dfs = pickle.load(f)
    print(f"Loaded {len(tmp_bias_dfs)} permutations from cache")
else:
    print("Running 10K permutations (batch mode)...")
    tmp_bias_dfs = batch_permutation_bias(STR_BiasMat, Geneset, Weights, n_perm=10000, seed=42)
    with open(cache_path, "wb") as f:
        pickle.dump(tmp_bias_dfs, f)
    print(f"Saved {len(tmp_bias_dfs)} permutations to cache")

# %% [markdown]
# ## 5.2 Per-structure null tests

# %%
p_value, observed_effect, null_effects = plot_null_distribution_analysis("Nucleus_accumbens", tmp_bias_dfs, Spark_ASD_STR_Bias)

# %%
p_value, observed_effect, null_effects = plot_null_distribution_analysis("Caudoputamen", tmp_bias_dfs, Spark_ASD_STR_Bias)

# %%
# Run for all structures
P_constraint = {}
for structure in Spark_ASD_STR_Bias.index:
    p_value, observed_effect, null_effects = plot_null_distribution_analysis(
        structure, tmp_bias_dfs, Spark_ASD_STR_Bias, title_prefix="", plot=False)
    P_constraint[structure] = p_value

Spark_ASD_STR_Bias_with_p = Spark_ASD_STR_Bias.copy()
Spark_ASD_STR_Bias_with_p['P_constraint'] = Spark_ASD_STR_Bias_with_p.index.map(P_constraint)

# %%
Spark_ASD_STR_Bias_with_p[Spark_ASD_STR_Bias_with_p["P_constraint"] < 0.05].sort_values(by="P_constraint")

# %%
Spark_ASD_STR_Bias_with_p[Spark_ASD_STR_Bias_with_p["P_constraint"] > 0.1]

# %%
p_value, observed_effect, null_effects = plot_null_distribution_analysis("Facial_motor_nucleus", tmp_bias_dfs, Spark_ASD_STR_Bias)

# %% [markdown]
# ## 5.3 Correlation null distribution

# %%
# Top-50 average EFFECT null
records = [tmp_bias_dfs[i].head(50)["EFFECT"].mean() for i in range(len(tmp_bias_dfs))]
null_effects = np.array(records)
observed_effect = Spark_ASD_STR_Bias.head(50)["EFFECT"].mean()
p_value = (np.sum(null_effects >= observed_effect) + 1) / (len(null_effects) + 1)

plt.figure(figsize=(10, 6))
plt.hist(null_effects, bins=50, alpha=0.7, color='lightblue', edgecolor='black', label='Null distribution (Constrained Genes)')
plt.axvline(observed_effect, color='red', linestyle='--', linewidth=2, label=f'Observed (Spark ASD): {observed_effect:.4f}')
plt.xlabel('EFFECT')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'P-value: {p_value:.4f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.show()
print(f"Observed Spark ASD effect: {observed_effect:.4f}")
print(f"Null mean: {np.mean(null_effects):.4f}, Null std: {np.std(null_effects):.4f}")
print(f"P-value: {p_value:.4f}")

# %%
# Correlation null: ASD and DDD vs random constrained subsets
Corrs_ASD_Constraint = []
Corrs_DDD_Constraint = []
for i in range(len(tmp_bias_dfs)):
    top_avg_bias = tmp_bias_dfs[i]

    tmp_merged = merge_bias_datasets(Spark_ASD_STR_Bias, top_avg_bias, suffixes=('_ASD', '_Constrained'))
    Corrs_ASD_Constraint.append(tmp_merged["EFFECT_ASD"].corr(tmp_merged["EFFECT_Constrained"]))

    tmp_merged = merge_bias_datasets(DDD_rmASD_STR_Bias, top_avg_bias, suffixes=('_DD', '_Constrained'))
    Corrs_DDD_Constraint.append(tmp_merged["EFFECT_DD"].corr(tmp_merged["EFFECT_Constrained"]))

Corrs_ASD_Constraint = np.array(Corrs_ASD_Constraint)
Corrs_DDD_Constraint = np.array(Corrs_DDD_Constraint)

# %%
# Compute observed correlations from data (not hard-coded)
constraint_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, constraint_gw)
constraint_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in constraint_STR_Bias.index]

merged_obs_asd = merge_bias_datasets(Spark_ASD_STR_Bias, constraint_STR_Bias, suffixes=('_ASD', '_Constrained'))
observed_effect_asd = pearsonr(merged_obs_asd["EFFECT_ASD"], merged_obs_asd["EFFECT_Constrained"])[0]

merged_obs_ddd = merge_bias_datasets(DDD_rmASD_STR_Bias, constraint_STR_Bias, suffixes=('_DD', '_Constrained'))
observed_effect_ddd = pearsonr(merged_obs_ddd["EFFECT_DD"], merged_obs_ddd["EFFECT_Constrained"])[0]

print(f"Observed ASD vs Constraint correlation: {observed_effect_asd:.4f}")
print(f"Observed DDD vs Constraint correlation: {observed_effect_ddd:.4f}")

# %%
# Plot null distribution vs observed for both ASD and DDD
null_effects_asd = Corrs_ASD_Constraint
p_value_asd = (np.sum(null_effects_asd >= observed_effect_asd) + 1) / (len(null_effects_asd) + 1)

null_effects_ddd = Corrs_DDD_Constraint
p_value_ddd = (np.sum(null_effects_ddd >= observed_effect_ddd) + 1) / (len(null_effects_ddd) + 1)

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

ax = axes[0]
ax.hist(null_effects_asd, bins=50, alpha=0.7, color='lightblue', edgecolor='black', label='Null distribution (Constrained Genes)')
ax.axvline(observed_effect_asd, color='red', linestyle='--', linewidth=2, label=f'Observed (Spark ASD): {observed_effect_asd:.4f}')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'P-value: {p_value_asd:.4f}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
ax.set_title('ASD (Spark): Correlation Null Distribution vs Observed')

ax = axes[1]
ax.hist(null_effects_ddd, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', label='Null distribution (Constrained Genes)')
ax.axvline(observed_effect_ddd, color='red', linestyle='--', linewidth=2, label=f'Observed (DDD): {observed_effect_ddd:.4f}')
ax.set_xlabel('EFFECT')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'P-value: {p_value_ddd:.4f}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
ax.set_title('DDD: Correlation Null Distribution vs Observed')

plt.tight_layout()
plt.show()

print(f"ASD: observed={observed_effect_asd:.4f}, null mean={np.mean(null_effects_asd):.4f}, std={np.std(null_effects_asd):.4f}, P={p_value_asd:.4f}")
print(f"DDD: observed={observed_effect_ddd:.4f}, null mean={np.mean(null_effects_ddd):.4f}, std={np.std(null_effects_ddd):.4f}, P={p_value_ddd:.4f}")

# %%
