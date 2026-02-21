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

ProjDIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, SPARK_Gene_Weights,
    MouseSTR_AvgZ_Weighted, ScoreCircuit_SI_Joint,
    bootstrap_gene_mutations, BiasCorrelation,
)
from plot import (
    plot_structure_bias_correlation,
    compute_circuit_scores_for_profiles,
    plot_circuit_connectivity_scores_multi,
    plot_circuit_scores_with_bootstrap_ci,
)

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 04. Weighted ASD Mutation Bias
#
# This notebook computes structure-level mutation bias for ASD genes
# using mutation-count-based gene weights (SPARK/DeNovoWEST, Zhou et al. 2022).
#
# **Sections**:
# 1. Load data and compute gene weights (with/without mutability correction)
# 2. Compare recomputed bias vs reference
# 3. Mutation bootstrap (1000 iterations)
# 4. CCS profiles with bootstrap confidence intervals

# %% [markdown]
# ## 1. Load Data & Compute Gene Weights

# %%
# Load config and expression matrix
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
Anno = STR2Region()

print(f"Expression matrix: {STR_BiasMat.shape}")

# %%
# Load SPARK exome-wide significant ASD genes (Zhou et al. 2022)
Spark_Meta_2stage = pd.read_excel(
    "../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
    skiprows=2, sheet_name="Table S7"
)
Spark_Meta_2stage = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] != "."]
Spark_Meta_ExomeWide = Spark_Meta_2stage[
    Spark_Meta_2stage["pDenovoWEST_Meta"] <= 1.3e-6
]
print(f"Exome-wide significant genes: {Spark_Meta_ExomeWide.shape[0]}")
print(f"  Total LoF: {Spark_Meta_ExomeWide['AutismMerged_LoF'].sum()}")
print(f"  Total Dmis: {Spark_Meta_ExomeWide['AutismMerged_Dmis_REVEL0.5'].sum()}")

# %%
# Load background mutation rate (BGMR) for mutability correction
bgmr_path = config["data_files"]["Denovo_mut_rate"]
BGMR = pd.read_csv(bgmr_path, sep=None, engine='python', index_col=0)

if "GeneName" in BGMR.columns:
    BGMR["entrez_id"] = BGMR["GeneName"].map(GeneSymbol2Entrez)
    BGMR = BGMR[~BGMR["entrez_id"].isna()].copy()
    BGMR["entrez_id"] = BGMR["entrez_id"].astype(int)
    BGMR = BGMR.set_index("entrez_id")
print(f"BGMR: {BGMR.shape[0]} genes with mutation rates")

# %%
# Compute gene weights — two versions:
# 1. With mutability correction (BGMR): subtracts expected mutation counts
# 2. Without correction (BGMR=None): uses raw observed counts only

_, Agg_gene_bgmr = SPARK_Gene_Weights(
    Spark_Meta_ExomeWide, BGMR,
    out="../dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.bgmr.csv"
)
ASD_STR_Bias_bgmr = MouseSTR_AvgZ_Weighted(STR_BiasMat, Agg_gene_bgmr)

_, Agg_gene2MutN = SPARK_Gene_Weights(
    Spark_Meta_ExomeWide, None,
    out="../dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.v2.csv"
)
ASD_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, Agg_gene2MutN)

print(f"Genes with mutability correction: {len(Agg_gene_bgmr)}")
print(f"Genes without correction:         {len(Agg_gene2MutN)}")

# %% [markdown]
# ## 2. Compare with Reference Bias

# %%
# Load reference bias (from FDR-corrected analysis)
ASD_Z2_ref = pd.read_csv(
    "../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR"
)

# Scatter: recomputed vs reference
BiasCorrelation(
    ASD_STR_Bias, ASD_Z2_ref,
    name1="Recomputed ASD Bias", name2="Reference (FDR)", dpi=200
)

# %%
# Effect of mutability correction
plot_structure_bias_correlation(
    ASD_Z2_ref, ASD_STR_Bias_bgmr,
    label_a='Mutation Bias\nZhou et al. 61 ASD genes',
    label_b='Mutation Bias (Mutability Corrected)\nZhou et al. 61 ASD genes',
)

# %% [markdown]
# ## 3. Mutation Bootstrap
#
# Resample mutations at the individual mutation level (preserving total counts)
# to generate 1000 bootstrap replicates. Two modes:
# - **Weighted**: probability proportional to observed mutation counts per gene
# - **Uniform**: equal probability across genes

# %%
# Select gene-level columns for bootstrap
Spark_Meta_EW_Genes = Spark_Meta_ExomeWide[[
    "GeneID", "EntrezID", "HGNC", "ExACpLI", "LOEUF",
    "AutismMerged_LoF", "AutismMerged_Dmis_REVEL0.5", "pDenovoWEST_Meta"
]]
print(f"Bootstrapping {len(Spark_Meta_EW_Genes)} genes, 1000 iterations each...")

# %%
# Generate bootstrap replicates
N_BOOT = 1000
boot_DFs_weights = bootstrap_gene_mutations(Spark_Meta_EW_Genes, N_BOOT, weighted=True)
boot_DFs_uniform = bootstrap_gene_mutations(Spark_Meta_EW_Genes, N_BOOT, weighted=False)
print(f"Generated {N_BOOT} weighted + {N_BOOT} uniform bootstrap replicates")

# %%
# Compute bias for each bootstrap replicate (with caching)
BOOT_CACHE_W = "../results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling"
BOOT_CACHE_U = "../results/Bootstrap_bias/Spark_ExomeWide/Uniform_Resampling"

# Check if cached results exist
cached_w = os.path.exists(os.path.join(BOOT_CACHE_W, "Spark_ExomeWide.GeneWeight.boot0.csv"))
cached_u = os.path.exists(os.path.join(BOOT_CACHE_U, "Spark_ExomeWide.GeneWeight.boot0.csv"))

if cached_w:
    print("Loading cached weighted bootstrap bias...")
    boot_bias_list_weights = []
    for i in range(N_BOOT):
        df = pd.read_csv(os.path.join(BOOT_CACHE_W, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"), index_col=0)
        boot_bias_list_weights.append(df)
else:
    print("Computing weighted bootstrap bias (this takes a few minutes)...")
    os.makedirs(BOOT_CACHE_W, exist_ok=True)
    boot_bias_list_weights = []
    for i, DF in enumerate(boot_DFs_weights):
        _, boot_gw = SPARK_Gene_Weights(DF, BGMR, Bmis=False)
        boot_bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, boot_gw)
        boot_bias_list_weights.append(boot_bias)
        boot_bias.to_csv(os.path.join(BOOT_CACHE_W, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"))

if cached_u:
    print("Loading cached uniform bootstrap bias...")
    boot_bias_list_uniform = []
    for i in range(N_BOOT):
        df = pd.read_csv(os.path.join(BOOT_CACHE_U, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"), index_col=0)
        boot_bias_list_uniform.append(df)
else:
    print("Computing uniform bootstrap bias (this takes a few minutes)...")
    os.makedirs(BOOT_CACHE_U, exist_ok=True)
    boot_bias_list_uniform = []
    for i, DF in enumerate(boot_DFs_uniform):
        _, boot_gw = SPARK_Gene_Weights(DF, BGMR)
        boot_bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, boot_gw)
        boot_bias_list_uniform.append(boot_bias)
        boot_bias.to_csv(os.path.join(BOOT_CACHE_U, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"))

print(f"Loaded {len(boot_bias_list_weights)} weighted, {len(boot_bias_list_uniform)} uniform bootstrap biases")

# %% [markdown]
# ## 4. CCS Profiles with Bootstrap Confidence Intervals
#
# Compute Circuit Connectivity Scores (CCS) for the original ASD bias
# and all bootstrap replicates, then plot with confidence intervals.

# %%
# Load connectivity scoring matrices and sibling null CCS profiles
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
# CCS profile: recomputed vs reference
topNs = list(range(200, 5, -1))

profiles = {
    "Spark 61 (recomputed)": ASD_STR_Bias,
    "Spark 61 (reference)": ASD_Z2_ref,
}

circuit_scores = compute_circuit_scores_for_profiles(profiles, topNs, info_mats)

fig = plot_circuit_connectivity_scores_multi(
    topNs, circuit_scores, cont_distance_dict, xlim=(0, 121)
)
plt.show()

# %%
# Compute CCS for original ASD bias (mean line for bootstrap CI plots)
CCS_CACHE = "../results/Bootstrap_bias/Spark_ExomeWide/bootstrap_CCS.npz"

if os.path.exists(CCS_CACHE):
    print("Loading cached bootstrap CCS scores...")
    cached = np.load(CCS_CACHE)
    mean_circuit_scores = {k: cached[f"mean_{k}"] for k in info_mats}
    boot_circuit_scores_weights = {k: cached[f"boot_{k}"] for k in info_mats}
else:
    print("Computing CCS for original ASD bias...")
    mean_circuit_scores = {}
    str_ranks = ASD_STR_Bias.sort_values("EFFECT", ascending=False).index.values
    for conn_type, info_mat in info_mats.items():
        scores = [ScoreCircuit_SI_Joint(str_ranks[:topN], info_mat) for topN in topNs]
        mean_circuit_scores[conn_type] = np.array(scores)

    print(f"Computing CCS for {N_BOOT} bootstrap samples (3 conn types × 195 topNs)...")
    boot_circuit_scores_weights = {ct: [] for ct in info_mats}
    for boot_idx, boot_bias in enumerate(boot_bias_list_weights):
        if (boot_idx + 1) % 100 == 0:
            print(f"  Bootstrap {boot_idx + 1}/{N_BOOT}")
        str_ranks = boot_bias.sort_values("EFFECT", ascending=False).index.values
        for conn_type, info_mat in info_mats.items():
            scores = [ScoreCircuit_SI_Joint(str_ranks[:topN], info_mat) for topN in topNs]
            boot_circuit_scores_weights[conn_type].append(scores)

    for ct in boot_circuit_scores_weights:
        boot_circuit_scores_weights[ct] = np.array(boot_circuit_scores_weights[ct])
        print(f"  {ct}: {boot_circuit_scores_weights[ct].shape}")

    # Cache results
    os.makedirs(os.path.dirname(CCS_CACHE), exist_ok=True)
    save_dict = {}
    for ct in info_mats:
        save_dict[f"mean_{ct}"] = mean_circuit_scores[ct]
        save_dict[f"boot_{ct}"] = boot_circuit_scores_weights[ct]
    np.savez(CCS_CACHE, **save_dict)
    print(f"Cached to {CCS_CACHE}")

# %%
# Plot: CCS with 95% bootstrap CI + sibling IQR
fig = plot_circuit_scores_with_bootstrap_ci(
    topNs=topNs,
    mean_scores=mean_circuit_scores,
    boot_scores=boot_circuit_scores_weights,
    cont_distance_dict=cont_distance_dict,
    ci_type='percentile',
    percentile_range=95,
    viz_style='shade',
    show_asd_ci=True,
    show_sib_ci=True,
    xlim=(0, 121),
)
plt.show()

# %%
# Plot: CCS with sibling IQR only (no bootstrap CI)
fig = plot_circuit_scores_with_bootstrap_ci(
    topNs=topNs,
    mean_scores=mean_circuit_scores,
    boot_scores=boot_circuit_scores_weights,
    cont_distance_dict=cont_distance_dict,
    show_asd_ci=False,
    show_sib_ci=True,
    xlim=(0, 121),
)
plt.show()

# %%
# Plot: CCS with bootstrap CI only (no sibling IQR)
fig = plot_circuit_scores_with_bootstrap_ci(
    topNs=topNs,
    mean_scores=mean_circuit_scores,
    boot_scores=boot_circuit_scores_weights,
    cont_distance_dict=cont_distance_dict,
    show_asd_ci=True,
    show_sib_ci=False,
    xlim=(0, 121),
)
plt.show()

# %% [markdown]
# ## Summary
#
# | Output | Description |
# |--------|-------------|
# | `Spark_Meta_EWS.GeneWeight.v2.csv` | Gene weights (no mutability correction) |
# | `Spark_Meta_EWS.GeneWeight.bgmr.csv` | Gene weights (mutability corrected) |
# | `Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling/` | 1000 bootstrap bias CSVs |
# | `Bootstrap_bias/Spark_ExomeWide/Uniform_Resampling/` | 1000 bootstrap bias CSVs |
# | `Bootstrap_bias/Spark_ExomeWide/bootstrap_CCS.npz` | Cached CCS for bootstrap |
