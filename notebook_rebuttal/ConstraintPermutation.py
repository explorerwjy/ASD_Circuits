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

# %% [markdown]
# # Permutation Test: ASD vs Constrained Gene Pool
#
# Tests whether ASD structure bias is significantly different from what would
# be expected by randomly sampling genes from the constrained gene pool
# (LOEUF top 10%).
#
# **Analyses:**
# 1. Batch permutation (10K) with caching
# 2. Per-structure null tests (e.g., Nucleus accumbens, Caudoputamen)
# 3. Correlation null distribution (ASD and DDD vs random constrained subsets)

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# %%
# Load ASD bias and gene weights
Spark_ASD_STR_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col=0)
Spark_ASD_STR_Bias["Region"] = Spark_ASD_STR_Bias["REGION"]
ASD_GW = Fil2Dict(ProjDIR + "dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw")
ASD_GENES = list(ASD_GW.keys())

# Load DDD bias (exclude ASD genes)
DDD_GW = Fil2Dict(config["gene_sets"]["DDD_293"]["geneweights"])
DDD_GW_filt_ASD = {k: v for k, v in DDD_GW.items() if k not in ASD_GENES}
DDD_rmASD_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, DDD_GW_filt_ASD)
DDD_rmASD_STR_Bias["Region"] = [STR_Anno.get(s, "Unknown") for s in DDD_rmASD_STR_Bias.index]

# %%
# Load gnomAD v4 constraint data
gnomad4 = pd.read_csv("/home/jw3514/Work/data/gnomad/gnomad.v4.0.constraint_metrics.tsv", sep="\t")
gnomad4 = gnomad4[(gnomad4["transcript"].str.contains('ENST'))]
gnomad4 = gnomad4[gnomad4["mane_select"] == True]
for i, row in gnomad4.iterrows():
    gnomad4.loc[i, "Entrez"] = int(GeneSymbol2Entrez.get(row["gene"], 0))

# %% [markdown]
# ## 1. Batch permutation (with caching)

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
# ## 2. Per-structure null tests

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
# ## 3. Correlation null distribution

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
