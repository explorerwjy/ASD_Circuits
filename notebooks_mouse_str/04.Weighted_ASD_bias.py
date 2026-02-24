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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, SPARK_Gene_Weights,
    MouseSTR_AvgZ_Weighted, ScoreCircuit_SI_Joint,
    Dict2Fil, Fil2Dict,
)
from plot import (
    plot_structure_bias_correlation,
    compute_circuit_scores_for_profiles,
    plot_circuit_connectivity_scores_multi,
)

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 04. ASD Gene Weights
#
# Compute mutation-count-based gene weights for **all ASD gene sets** from raw
# mutation tables, then compare structure-level bias and circuit connectivity
# score (CCS) profiles across gene sets.
#
# **Gene sets produced:**
# 1. SPARK 61 (Zhou et al. 2022 — exome-wide significant, p < 1.3e-6)
# 2. SPARK ~159 (Zhou et al. 2022 — Stage 1 top 160)
# 3. ASC 102 (Satterstrom et al. 2020)
# 4. Fu ASD 72 (Fu et al. 2022)
# 5. Fu ASD 185 (Fu et al. 2022)

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
# Config, expression matrix, gene annotations
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
Anno = STR2Region()

print(f"Expression matrix: {STR_BiasMat.shape}")
print(f"Gene annotations: {len(GeneSymbol2Entrez)} symbol→Entrez mappings")

# %%
# Background mutation rate (BGMR) for mutability correction
bgmr_path = config["data_files"]["Denovo_mut_rate"]
BGMR = pd.read_csv(bgmr_path, sep=None, engine='python', index_col=0)

if "GeneName" in BGMR.columns:
    BGMR["entrez_id"] = BGMR["GeneName"].map(GeneSymbol2Entrez)
    BGMR = BGMR[~BGMR["entrez_id"].isna()].copy()
    BGMR["entrez_id"] = BGMR["entrez_id"].astype(int)
    BGMR = BGMR.set_index("entrez_id")
print(f"BGMR: {BGMR.shape[0]} genes with mutation rates")

# %%
# Gene weight output directory
GW_DIR = os.path.join(ProjDIR, "dat/Genetics/GeneWeights")
BIAS_DIR = os.path.join(ProjDIR, "dat/Unionize_bias")

# Collect all gene weights for later comparison
all_geneweights = {}

# %% [markdown]
# ## 2. SPARK 61 Genes (Zhou et al. 2022 — Exome-Wide Significant)
#
# Source: Table S7 from Zhou et al. 2022 Nature Genetics supplement.
# Filter: pDenovoWEST_Meta < 1.3e-6 (exome-wide significance threshold).
# Weight formula: `SPARK_Gene_Weights()` — uses LoF and Dmis counts weighted
# by pLI-dependent priors (0.554/0.333 for pLI≥0.5, 0.138/0.130 otherwise).

# %%
Spark_Meta_2stage = pd.read_excel(
    "../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
    skiprows=2, sheet_name="Table S7"
)
Spark_Meta_2stage = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] != "."]
Spark_Meta_ExomeWide = Spark_Meta_2stage[
    Spark_Meta_2stage["pDenovoWEST_Meta"] <= 1.3e-6
]

# Gene weights — without BGMR correction (primary)
_, GW_Spark61 = SPARK_Gene_Weights(
    Spark_Meta_ExomeWide, None,
    out=os.path.join(BIAS_DIR, "Spark_Meta_EWS.GeneWeight.v2.csv")
)
Dict2Fil(GW_Spark61, os.path.join(GW_DIR, "ASD_All.gw"))

# Gene weights — with BGMR correction
_, GW_Spark61_bgmr = SPARK_Gene_Weights(
    Spark_Meta_ExomeWide, BGMR,
    out=os.path.join(BIAS_DIR, "Spark_Meta_EWS.GeneWeight.bgmr.csv")
)

all_geneweights["SPARK 61"] = GW_Spark61

n_in_z2 = len(set(GW_Spark61.keys()) & set(STR_BiasMat.index))
print(f"SPARK 61: {len(GW_Spark61)} genes ({n_in_z2} in Z2 expression matrix)")
print(f"  Total LoF: {Spark_Meta_ExomeWide['AutismMerged_LoF'].sum()}")
print(f"  Total Dmis: {Spark_Meta_ExomeWide['AutismMerged_Dmis_REVEL0.5'].sum()}")
top5 = sorted(GW_Spark61.items(), key=lambda x: x[1], reverse=True)[:5]
for entrez, w in top5:
    sym = Entrez2Symbol.get(entrez, "?")
    print(f"  {sym} ({entrez}): {w:.3f}")

# %% [markdown]
# ## 3. SPARK ~159 Genes (Zhou et al. 2022 — Stage 1 Top 160)
#
# Source: DenovoWEST Stage 1 results (AllGenes sheet).
# Filter: top 160 genes by pDenovoWEST (Stage 1 p-value).
# Same weight formula as SPARK 61 via `SPARK_Gene_Weights()`.
# Note: 160 input genes → ~153 in Z2 expression matrix (7 lack mouse orthologs).

# %%
Stage1 = pd.read_excel(
    "../dat/Genetics/TabS_DenovoWEST_Stage1.xlsx",
    skiprows=1, sheet_name="AllGenes"
)
Stage1 = Stage1[Stage1["pDenovoWEST"] != "."]
Stage1_top160 = Stage1.sort_values("pDenovoWEST").head(160)

_, GW_Spark159 = SPARK_Gene_Weights(
    Stage1_top160, None,
    out=os.path.join(GW_DIR, "Spark_Meta_160.GeneWeight.csv")
)

all_geneweights["SPARK 159"] = GW_Spark159

n_in_z2 = len(set(GW_Spark159.keys()) & set(STR_BiasMat.index))
overlap_61 = len(set(GW_Spark159.keys()) & set(GW_Spark61.keys()))
print(f"SPARK 159: {len(GW_Spark159)} genes ({n_in_z2} in Z2 expression matrix)")
print(f"  Overlap with SPARK 61: {overlap_61}")

# %% [markdown]
# ## 4. ASC 102 Genes (Satterstrom et al. 2020)
#
# Source: ASC-102Genes.xlsx — 102 ASD risk genes from Satterstrom et al. 2020 Cell.
# Weight formula: `dn.ptv × γ_ptv + dn.misb × γ_misb + dn.misa × γ_misa`
# where γ (gamma) priors come from the TADA model (Autosomal sheet).

# %%
asc_102_df = pd.read_excel(
    "../dat/Genetics/ASC-102Genes.xlsx", sheet_name="102_ASD"
)
asc_autosomal = pd.read_excel(
    "../dat/Genetics/ASC-102Genes.xlsx", sheet_name="Autosomal"
)

# Merge gamma priors from Autosomal sheet
asc_102_df = asc_102_df.merge(
    asc_autosomal[["gene", "gamma_dn.ptv", "gamma_dn.misa", "gamma_dn.misb"]],
    on="gene", how="left"
)


def GeneWeights_ASC_102(df):
    """Compute gene weights for ASC 102 genes using TADA gamma priors."""
    gene2MutN = {}
    for _, row in df.iterrows():
        symbol = row["gene"]
        try:
            g = GeneSymbol2Entrez[symbol]
        except KeyError:
            continue
        PR_LGD = row["gamma_dn.ptv"]
        PR_MisA = row["gamma_dn.misa"]
        PR_MisB = row["gamma_dn.misb"]
        if pd.isna(PR_LGD):
            continue
        weight = (
            row["dn.ptv"] * PR_LGD +
            row["dn.misb"] * PR_MisB +
            row["dn.misa"] * PR_MisA
        )
        gene2MutN[int(g)] = gene2MutN.get(int(g), 0) + weight
    return gene2MutN


GW_ASC102 = GeneWeights_ASC_102(asc_102_df)
Dict2Fil(GW_ASC102, os.path.join(GW_DIR, "GW_ASC_102.gw"))

all_geneweights["ASC 102"] = GW_ASC102

n_in_z2 = len(set(GW_ASC102.keys()) & set(STR_BiasMat.index))
overlap_61 = len(set(GW_ASC102.keys()) & set(GW_Spark61.keys()))
print(f"ASC 102: {len(GW_ASC102)} genes ({n_in_z2} in Z2 expression matrix)")
print(f"  Overlap with SPARK 61: {overlap_61}")

# %% [markdown]
# ## 5. Fu et al. 2022 — 72 and 185 Genes
#
# Source: Fu et al. 2022 Nature Genetics supplement.
# - Table S5: SSC+ASC cohort de novo mutation counts
# - Table S6: SPARK cohort de novo mutation counts
# - Table S8: TADA prior probabilities
# - Table S11: Gene significance flags (ASD72, ASD185)
#
# Weight formula (summed across SSC+ASC and SPARK cohorts):
# `weight = Σ(dn.ptv × prior.dn.ptv + dn.misb × prior.dn.misb + dn.misa × prior.dn.misa)`

# %%
FU_FILE = "../dat/Genetics/Fu_et_al_2022.xlsx"

fu_SSCASC = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 5")
fu_SPARK = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 6")
fu_TADA_PR = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 8")
fu_S11 = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 11")

# TADA priors indexed by gene symbol
fu_TADA_PR = fu_TADA_PR.set_index("gene_gencodeV33")

# Gene lists
fu_ASD72_genes = fu_S11[fu_S11["ASD72"] == 1]["gene_gencodeV33"]
fu_ASD185_genes = fu_S11[fu_S11["ASD185"] == 1]["gene_gencodeV33"]

print(f"Fu ASD72 gene list: {len(fu_ASD72_genes)}")
print(f"Fu ASD185 gene list: {len(fu_ASD185_genes)}")


def GeneWeights_Fu2022(gene_list, prior_df, mut_dfs):
    """Compute gene weights for Fu et al. gene sets using TADA priors.

    Aggregates weights across multiple cohort mutation tables (SSC+ASC, SPARK).
    """
    gene2MutN = {}
    for mut_df in mut_dfs:
        df_filt = mut_df[mut_df["gene_gencodeV33"].isin(gene_list)]
        for _, row in df_filt.iterrows():
            symbol = row["gene_gencodeV33"]
            try:
                g = GeneSymbol2Entrez[symbol]
            except KeyError:
                continue
            try:
                PR_LGD = prior_df.loc[symbol, "prior.dn.ptv"]
                PR_MisA = prior_df.loc[symbol, "prior.dn.misa"]
                PR_MisB = prior_df.loc[symbol, "prior.dn.misb"]
            except KeyError:
                continue
            weight = (
                row["dn.ptv"] * PR_LGD +
                row["dn.misb"] * PR_MisB +
                row["dn.misa"] * PR_MisA
            )
            gene2MutN[int(g)] = gene2MutN.get(int(g), 0) + weight
    return gene2MutN


# %%
# Fu ASD 72
GW_Fu72 = GeneWeights_Fu2022(fu_ASD72_genes, fu_TADA_PR, [fu_SSCASC, fu_SPARK])
Dict2Fil(GW_Fu72, os.path.join(GW_DIR, "GW_Fu_ASD_72.gw"))

all_geneweights["Fu ASD 72"] = GW_Fu72

n_in_z2 = len(set(GW_Fu72.keys()) & set(STR_BiasMat.index))
overlap_61 = len(set(GW_Fu72.keys()) & set(GW_Spark61.keys()))
print(f"Fu ASD 72: {len(GW_Fu72)} genes ({n_in_z2} in Z2 expression matrix)")
print(f"  Overlap with SPARK 61: {overlap_61}")

# %%
# Fu ASD 185
GW_Fu185 = GeneWeights_Fu2022(fu_ASD185_genes, fu_TADA_PR, [fu_SSCASC, fu_SPARK])
Dict2Fil(GW_Fu185, os.path.join(GW_DIR, "GW_Fu_ASD_185.gw"))

all_geneweights["Fu ASD 185"] = GW_Fu185

n_in_z2 = len(set(GW_Fu185.keys()) & set(STR_BiasMat.index))
overlap_61 = len(set(GW_Fu185.keys()) & set(GW_Spark61.keys()))
print(f"Fu ASD 185: {len(GW_Fu185)} genes ({n_in_z2} in Z2 expression matrix)")
print(f"  Overlap with SPARK 61: {overlap_61}")

# %% [markdown]
# ## 6. Bias Comparison with SPARK 61

# %%
# Compute structure bias for each gene set
bias_results = {}
for name, gw in all_geneweights.items():
    bias_results[name] = MouseSTR_AvgZ_Weighted(STR_BiasMat, gw)

# %%
# Scatter plots: each gene set vs SPARK 61
ref_bias = bias_results["SPARK 61"]
comparisons = ["SPARK 159", "ASC 102", "Fu ASD 72", "Fu ASD 185"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, comparisons):
    other_bias = bias_results[name]
    # Align on common structures
    common = ref_bias.index.intersection(other_bias.index)
    x = ref_bias.loc[common, "EFFECT"]
    y = other_bias.loc[common, "EFFECT"]
    r = np.corrcoef(x, y)[0, 1]

    ax.scatter(x, y, s=10, alpha=0.5)
    ax.set_xlabel("SPARK 61 bias")
    ax.set_ylabel(f"{name} bias")
    ax.set_title(f"{name} vs SPARK 61\nr = {r:.3f}")
    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

fig.patch.set_alpha(0)
for ax in axes:
    ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. CCS vs Bias Rank Profile (All Gene Sets)
#
# Circuit Connectivity Score (CCS) profiles across circuit sizes (N=6..200)
# for all gene sets, compared against sibling null IQR.

# %%
# Load connectivity scoring matrices
CONN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-conn")
SCORE_DIR = os.path.join(CONN_DIR, "ConnectomeScoringMat")
RANK_DIR = os.path.join(CONN_DIR, "RankScores")

IpsiInfoMat = pd.read_csv(os.path.join(SCORE_DIR, "InfoMat.Ipsi.csv"), index_col=0)
IpsiInfoMatShort = pd.read_csv(os.path.join(SCORE_DIR, "InfoMat.Ipsi.Short.3900.csv"), index_col=0)
IpsiInfoMatLong = pd.read_csv(os.path.join(SCORE_DIR, "InfoMat.Ipsi.Long.3900.csv"), index_col=0)

Cont_Distance = np.load(os.path.join(RANK_DIR, "RankScore.Ipsi.Cont.npy"))
Cont_DistanceShort = np.load(os.path.join(RANK_DIR, "RankScore.Ipsi.Short.3900.Cont.npy"))
Cont_DistanceLong = np.load(os.path.join(RANK_DIR, "RankScore.Ipsi.Long.3900.Cont.npy"))

info_mats = {
    "Standard": IpsiInfoMat,
    "Short": IpsiInfoMatShort,
    "Long": IpsiInfoMatLong,
}
cont_distance_dict = {
    "Standard": Cont_Distance,
    "Short": Cont_DistanceShort,
    "Long": Cont_DistanceLong,
}
print(f"Sibling null CCS profiles: {Cont_Distance.shape}")

# %%
# Compute CCS profiles for all gene sets
topNs = list(range(200, 5, -1))

profiles = {name: bias for name, bias in bias_results.items()}
circuit_scores = compute_circuit_scores_for_profiles(profiles, topNs, info_mats)

fig = plot_circuit_connectivity_scores_multi(
    topNs, circuit_scores, cont_distance_dict, xlim=(0, 121)
)
plt.show()

# %% [markdown]
# ## 8. Summary
#
# | Gene Set | Source | Genes | In Z2 | Output File |
# |----------|--------|-------|-------|-------------|
# | SPARK 61 | Zhou et al. 2022, Table S7 (p < 1.3e-6) | 61 | 60 | `ASD_All.gw` |
# | SPARK ~159 | Zhou et al. 2022, Stage 1 top 160 | 153 | ~148 | `Spark_Meta_160.GeneWeight.csv` |
# | ASC 102 | Satterstrom et al. 2020, 102 ASD risk genes | 100 | ~98 | `GW_ASC_102.gw` |
# | Fu ASD 72 | Fu et al. 2022, ASD72 flag | 72 | ~70 | `GW_Fu_ASD_72.gw` |
# | Fu ASD 185 | Fu et al. 2022, ASD185 flag | 185 | ~178 | `GW_Fu_ASD_185.gw` |
#
# Additional outputs:
# - `dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.v2.csv` — SPARK 61 (same as ASD_All.gw)
# - `dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.bgmr.csv` — SPARK 61 with mutability correction

# %%
# Print summary table
print(f"{'Gene Set':<16} {'Genes':>5} {'In Z2':>5}  Output")
print("-" * 60)
for name, gw in all_geneweights.items():
    n_in_z2 = len(set(gw.keys()) & set(STR_BiasMat.index))
    print(f"{name:<16} {len(gw):>5} {n_in_z2:>5}")

# %% [markdown]
# ## 8. Confound Controls
#
# Structure bias comparisons against neuronal density, neuro-to-glia ratio,
# and male vs female donor normalization.

# %%
ASD_Neuron_den_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuron.density.norm.bias.csv",
                                        index_col="STR")
ASD_Glia_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuro2glia.norm.bias.csv",
                                  index_col="STR")
ASD_Male = pd.read_csv("../dat/Unionize_bias/ASD.Male.ALL.bias.csv", index_col="STR")
ASD_Female = pd.read_csv("../dat/Unionize_bias/ASD.Female.ALL.bias.csv", index_col="STR")

ref_bias = bias_results["SPARK 61"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (other, label) in zip(axes, [
    (ASD_Neuron_den_norm_bias, "Neuronal Density Norm"),
    (ASD_Glia_norm_bias, "Neuro-to-Glia Norm"),
    (ASD_Female, "Female Donors"),
]):
    common = ref_bias.index.intersection(other.index)
    x = ref_bias.loc[common, "EFFECT"]
    y = other.loc[common, "EFFECT"]
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=10, alpha=0.5)
    ax.set_xlabel("SPARK 61 bias")
    ax.set_ylabel(f"{label} bias")
    ax.set_title(f"r = {r:.3f}")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)

fig.patch.set_alpha(0)
for ax in axes:
    ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %%
# Male vs Female bias comparison
common = ASD_Male.index.intersection(ASD_Female.index)
x = ASD_Male.loc[common, "EFFECT"]
y = ASD_Female.loc[common, "EFFECT"]
r = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x, y, s=10, alpha=0.5)
ax.set_xlabel("Male Mutation Bias")
ax.set_ylabel("Female Mutation Bias")
ax.set_title(f"Male vs Female (r = {r:.3f})")
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Gene Set Size Stability
#
# How does the structure bias profile change as we include more genes beyond
# the core set?  Expanding from the top-ranked genes outward, we compute bias
# for successively larger gene sets and measure correlation with the core set.

# %%
from scipy.stats import spearmanr

TopN_list = np.concatenate([np.arange(200, 2000, 100), np.arange(2000, 5000, 500)])

# SPARK expanding gene sets (genes ranked 62..topN, beyond the core 61)
Spark_Bias_DF_list = []
Spark_Bias_DF_list_Unif = []
for topN in TopN_list:
    SPARK_topN = Stage1.sort_values("pDenovoWEST").head(topN).iloc[61:]
    GW_Unif, GW = SPARK_Gene_Weights(SPARK_topN, None)
    Spark_Bias_DF_list.append(MouseSTR_AvgZ_Weighted(STR_BiasMat, GW))
    Spark_Bias_DF_list_Unif.append(MouseSTR_AvgZ_Weighted(STR_BiasMat, GW_Unif))

# Fu expanding gene sets (genes ranked 73..topN, beyond the core 72)
fu_S11_sorted = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 11")
fu_S11_sorted = fu_S11_sorted[fu_S11_sorted["gene_id"].notna()]
fu_S11_sorted = fu_S11_sorted.sort_values(by="p_TADA_ASD", ascending=True)

Fu_Bias_DF_list = []
for topN in TopN_list:
    Fu_topN = fu_S11_sorted.head(topN).iloc[72:]
    GW_Fu_topN = GeneWeights_Fu2022(Fu_topN["gene_gencodeV33"], fu_TADA_PR, [fu_SSCASC, fu_SPARK])
    Fu_Bias_DF_list.append(MouseSTR_AvgZ_Weighted(STR_BiasMat, GW_Fu_topN))

# %%
# Plot correlation with core set vs number of additional genes
ref_spark = bias_results["SPARK 61"]
ref_fu = bias_results["Fu ASD 72"]

cors_spark = [spearmanr(
    df.loc[df.index.intersection(ref_spark.index), "EFFECT"],
    ref_spark.loc[df.index.intersection(ref_spark.index), "EFFECT"]
).correlation for df in Spark_Bias_DF_list]

cors_spark_unif = [spearmanr(
    df.loc[df.index.intersection(ref_spark.index), "EFFECT"],
    ref_spark.loc[df.index.intersection(ref_spark.index), "EFFECT"]
).correlation for df in Spark_Bias_DF_list_Unif]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(TopN_list, cors_spark, marker='o', linewidth=2.5, label="Weighted", color='royalblue')
ax.plot(TopN_list, cors_spark_unif, marker='s', linewidth=2.5, label="Uniform", color='orange')
ax.set_xlabel("Number of Top ASD Genes (Excluding Top 61)", fontsize=14)
ax.set_ylabel("Spearman r with Top 61 Gene Bias", fontsize=14)
ax.legend()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. SCZ Comparison
#
# Compare expanding gene set stability between ASD and schizophrenia (SCZ).

# %%
SCZ_GeneDF = pd.read_csv("../dat/Genetics/SCZ.ALLGENE.MutCountModified.csv", index_col=0)

TopGeneToTest = 100
TopN_list2 = np.arange(200, 1600, 100)


def Aggregate_Gene_Weights_SCZ(MutFil, allen_mouse_genes, usepLI=False, mis2_weight=0):
    gene2MutN = {}
    for i, row in MutFil.iterrows():
        try:
            g = int(i)
            if g not in allen_mouse_genes:
                continue
        except Exception:
            continue
        if usepLI:
            try:
                pLI = float(row["pLI"])
            except Exception:
                pLI = 0.0
            if pLI >= 0.5:
                gene2MutN[g] = row["nLGD"] * 0.26 + row["nMis3"] * 0.25 + row["nMis2"] * 0.06
            else:
                gene2MutN[g] = row["nLGD"] * 0.01 + row["nMis3"] * 0.01
        else:
            gene2MutN[g] = row["nLGD"] * 0.33 + row["nMis3"] * 0.27 + row["nMis2"] * mis2_weight
    return gene2MutN


SCZ_top100_GW = Aggregate_Gene_Weights_SCZ(SCZ_GeneDF.head(TopGeneToTest), STR_BiasMat.index.values, usepLI=True, mis2_weight=0)
SCZ_top100_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, SCZ_top100_GW)

SCZ_Bias_DF_list = []
for topN in TopN_list2:
    SCZ_topN = SCZ_GeneDF.head(topN).iloc[TopGeneToTest:]
    GW_SCZ_topN = Aggregate_Gene_Weights_SCZ(SCZ_topN, STR_BiasMat.index.values, usepLI=True, mis2_weight=0)
    SCZ_Bias_DF_list.append(MouseSTR_AvgZ_Weighted(STR_BiasMat, GW_SCZ_topN))

ASD_top100_GW_U, ASD_top100_GW = SPARK_Gene_Weights(Stage1.sort_values("pDenovoWEST").head(TopGeneToTest), None)
ASD_top100_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, ASD_top100_GW)

ASD_Bias_DF_list2 = []
for topN in TopN_list2:
    SPARK_topN = Stage1.sort_values("pDenovoWEST").head(topN).iloc[TopGeneToTest:]
    _, GW_tmp = SPARK_Gene_Weights(SPARK_topN, None)
    ASD_Bias_DF_list2.append(MouseSTR_AvgZ_Weighted(STR_BiasMat, GW_tmp))

# %%
from scipy.stats import pearsonr

ref_asd = ASD_top100_Bias
ref_scz = SCZ_top100_Bias

sp_asd = [spearmanr(df.loc[df.index.intersection(ref_asd.index), "EFFECT"],
                     ref_asd.loc[df.index.intersection(ref_asd.index), "EFFECT"]).correlation
          for df in ASD_Bias_DF_list2]
pe_asd = [pearsonr(df.loc[df.index.intersection(ref_asd.index), "EFFECT"],
                    ref_asd.loc[df.index.intersection(ref_asd.index), "EFFECT"])[0]
          for df in ASD_Bias_DF_list2]

sp_scz = [spearmanr(df.loc[df.index.intersection(ref_scz.index), "EFFECT"],
                     ref_scz.loc[df.index.intersection(ref_scz.index), "EFFECT"]).correlation
          for df in SCZ_Bias_DF_list]
pe_scz = [pearsonr(df.loc[df.index.intersection(ref_scz.index), "EFFECT"],
                    ref_scz.loc[df.index.intersection(ref_scz.index), "EFFECT"])[0]
          for df in SCZ_Bias_DF_list]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.plot(TopN_list2, sp_asd, marker='o', linewidth=2.5, label="ASD (Spearman)", color='royalblue')
ax.plot(TopN_list2, pe_asd, marker='o', linewidth=2.5, label="ASD (Pearson)", color='darkblue', linestyle='--')
ax.plot(TopN_list2, sp_scz, marker='s', linewidth=2.5, label="SCZ (Spearman)", color='orange')
ax.plot(TopN_list2, pe_scz, marker='s', linewidth=2.5, label="SCZ (Pearson)", color='darkorange', linestyle='--')
ax.set_xlabel("Number of Genes (Excluding Top 100)", fontsize=14)
ax.set_ylabel("Bias Correlation with Top 100 Genes", fontsize=14)
ax.legend()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Top 20 + Longtail Bootstrap
#
# Bootstrap stability of bias: fix top 20 ASD genes, randomly sample 41
# additional genes from the top 500 longtail, repeat 1000 times.

# %%
rng = np.random.default_rng(42)
top20_df = Stage1.sort_values("pDenovoWEST").head(20)
_, top_20_GW = SPARK_Gene_Weights(top20_df, None)
Dict2Fil(top_20_GW, os.path.join(GW_DIR, "Spark_top20.gw"))
top20_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, top_20_GW)

save_dir = "../results/Bootstrap_bias/Spark_top20_Random40/"
os.makedirs(save_dir, exist_ok=True)

Bootstrap_Bias_list = []
for i in range(1000):
    Random_longtail = Stage1.sort_values("pDenovoWEST").head(500).iloc[20:].sample(n=41, random_state=rng)
    _, GW_tmp = SPARK_Gene_Weights(Random_longtail, None)
    combined_gw = {**top_20_GW, **GW_tmp}
    bias_df = MouseSTR_AvgZ_Weighted(STR_BiasMat, combined_gw)
    Bootstrap_Bias_list.append(bias_df)

# %%
# Correlation of each bootstrap sample with the top-20 reference
ref_idx = top20_Bias.index
ref_eff = top20_Bias["EFFECT"]
pearson_cors = []
for df in Bootstrap_Bias_list:
    common = ref_idx.intersection(df.index)
    pearson_cors.append(pearsonr(ref_eff.loc[common], df.loc[common, "EFFECT"])[0])
pearson_cors = np.array(pearson_cors)

ci_lo, ci_hi = np.percentile(pearson_cors, [2.5, 97.5])
print(f"Pearson r: mean={pearson_cors.mean():.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(pearson_cors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(pearson_cors.mean(), color='red', ls='--', lw=2, label=f'Mean: {pearson_cors.mean():.4f}')
ax.axvline(ci_lo, color='green', ls=':', lw=2, label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]')
ax.axvline(ci_hi, color='green', ls=':', lw=2)
ax.set_xlabel('Pearson Correlation')
ax.set_ylabel('Frequency')
ax.legend()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
plt.show()
