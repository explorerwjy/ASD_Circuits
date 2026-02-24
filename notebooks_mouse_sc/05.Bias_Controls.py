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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05. Bias Controls — Sibling-Null P-values for Cell-Type Bias
#
# Compute permutation-based p-values and FDR q-values for ASD cell-type bias
# by comparing observed bias against a sibling-null distribution (10,000 permutations
# with gene-level mutation probability weighting).
#
# **Input**:
# - Cell-type bias (from 02): `dat/Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.csv`
# - Sibling null distributions (10K CSVs, legacy): `SubSampleSib_w_GeneProb/Cluster_V3_UMI_Z2_DN_V2_zclip3/`
#
# **Output**: `dat/Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.addP.csv`
# - Adds columns: EFFECT2 (mean-adjusted), Pvalue, Z_Match, Z_Pvalue, qvalues, qvalues_ZP
#
# **Note**: The sibling null data (10K CSVs, ~30 GB) is not stored locally. This notebook
# documents the methodology and validates the existing output. To regenerate, place
# null distributions at the legacy path and re-run the commented generation code in Section 2.

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. Methodology
#
# For each of the ~5,000 cell-type clusters, we compare the observed ASD bias
# (weighted average Z2 score) against a null distribution derived from 10,000
# sibling-null permutations.
#
# **Sibling-null model**: For each permutation, gene weights are drawn from
# unaffected siblings using mutation-probability weighting (`GeneProb_LGD_Dmis.csv`),
# then cell-type bias is computed identically to the observed ASD bias.
#
# **P-value computation**:
# - **Permutation P**: Fraction of null values ≥ observed: `P = (count + 1) / (N + 1)`
# - **Z-score**: `Z = (observed - mean(null)) / std(null)`
# - **Z P-value**: `P_Z = 1 - Φ(|Z|)` (one-sided normal survival)
# - **FDR correction**: Benjamini-Hochberg (α=0.1, "indep" method) applied to both
#   permutation P and Z P-values → `qvalues` and `qvalues_ZP`
#
# ```python
# # --- Generation code (requires sibling null data) ---
# # SibBiasDFs = LoadCTRLBiasDFs(CTRL_DIR, n_samples=10000)
# # for CT, row in bias_df.iterrows():
# #     null_biases = np.array([df.loc[CT, "EFFECT"] for df in SibBiasDFs])
# #     Z, P, _ = GetPermutationP(null_biases, row["EFFECT"])
# #     bias_df.loc[CT, "EFFECT2"] = row["EFFECT"] - np.mean(null_biases)
# #     bias_df.loc[CT, "Pvalue"] = P
# #     bias_df.loc[CT, "Z_Match"] = Z
# #     bias_df.loc[CT, "Z_Pvalue"] = scipy.stats.norm.sf(abs(Z))
# # _, qvalues = fdrcorrection(bias_df["Pvalue"].values, alpha=0.1, method="i")
# # bias_df["qvalues"] = qvalues
# ```

# %% [markdown]
# ## 2. Load Existing Results

# %%
addP_path = f"../{config['data_files']['ct_bias_addp']}"
CT_Bias = pd.read_csv(addP_path, index_col=0)
print(f"Cell-type bias with p-values: {CT_Bias.shape[0]} clusters")
print(f"Columns: {list(CT_Bias.columns)}")

# %% [markdown]
# ## 3. Summary Statistics

# %%
n_total = len(CT_Bias)
n_sig_005 = (CT_Bias["qvalues"] < 0.05).sum()
n_sig_010 = (CT_Bias["qvalues"] < 0.10).sum()
n_sigZ_005 = (CT_Bias["qvalues_ZP"] < 0.05).sum()
n_sigZ_010 = (CT_Bias["qvalues_ZP"] < 0.10).sum()

print(f"Total clusters: {n_total}")
print(f"\nPermutation P (qvalues):")
print(f"  FDR < 0.05: {n_sig_005}")
print(f"  FDR < 0.10: {n_sig_010}")
print(f"\nZ-score P (qvalues_ZP):")
print(f"  FDR < 0.05: {n_sigZ_005}")
print(f"  FDR < 0.10: {n_sigZ_010}")

# %%
print(f"\nTop 10 clusters by EFFECT:")
print(CT_Bias[["EFFECT", "Pvalue", "qvalues", "class_id_label"]].head(10).to_string())

# %% [markdown]
# ## 4. FDR-Significant Clusters by Cell Class

# %%
sig_clusters = CT_Bias[CT_Bias["qvalues"] < 0.10]
class_counts = sig_clusters["class_id_label"].value_counts()

print(f"FDR < 0.10 significant clusters by class (total = {len(sig_clusters)}):")
for cls, count in class_counts.items():
    total_in_class = (CT_Bias["class_id_label"] == cls).sum()
    print(f"  {cls}: {count}/{total_in_class}")

# %% [markdown]
# ## 5. QQ Plot of P-values by Cell Class

# %%
# 7 highlight classes with distinct colors; everything else as gray "Other"
highlight_classes = [
    "01 IT-ET Glut",
    "02 NP-CT-L6b Glut",
    "06 CTX-CGE GABA",
    "09 CNU-LGE GABA",
    "11 CNU-HYa GABA",
    "13 CNU-HYa Glut",
    "18 TH Glut",
]
highlight_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Plot "Other" first (gray, behind)
other_mask = ~CT_Bias["class_id_label"].isin(highlight_classes)
pvals = np.sort(CT_Bias.loc[other_mask, "Pvalue"].values)
n = len(pvals)
expected = -np.log10(np.arange(1, n + 1) / (n + 1))
observed = -np.log10(pvals)
ax.scatter(np.sort(expected)[::-1], observed, s=4, alpha=0.3, color="0.6", label=f"Other (n={n})", zorder=1)

# Plot each highlighted class
for cls, color in zip(highlight_classes, highlight_colors):
    mask = CT_Bias["class_id_label"] == cls
    pvals = np.sort(CT_Bias.loc[mask, "Pvalue"].values)
    n = len(pvals)
    if n == 0:
        continue
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(pvals)
    ax.scatter(np.sort(expected)[::-1], observed, s=8, alpha=0.7, color=color, label=f"{cls} (n={n})", zorder=2)

# FDR threshold lines
for fdr_thresh, ls in [(0.05, "-"), (0.10, "--")]:
    mask_fdr = CT_Bias["qvalues"] < fdr_thresh
    if mask_fdr.any():
        p_cutoff = CT_Bias.loc[mask_fdr, "Pvalue"].max()
        ax.axhline(-np.log10(p_cutoff), color="0.3", ls=ls, lw=0.8, alpha=0.7, zorder=0)
        ax.text(ax.get_xlim()[1] * 0.02, -np.log10(p_cutoff) + 0.05,
                f"FDR {fdr_thresh:.0%}", fontsize=7, color="0.3", va="bottom")

# Diagonal reference
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, max_val], [0, max_val], "k--", lw=0.5, alpha=0.5)
ax.set_xlabel("Expected -log10(p)")
ax.set_ylabel("Observed -log10(p)")
ax.set_title("QQ Plot: Sibling-Null P-values by Cell Class")
ax.legend(fontsize=7, loc="lower right", markerscale=1.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Bias Distribution: Significant vs Non-Significant

# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

sig = CT_Bias["qvalues"] < 0.10
ax.hist(CT_Bias.loc[~sig, "EFFECT"], bins=50, alpha=0.5, color="grey",
        label=f"Not sig (n={(~sig).sum()})", density=True)
ax.hist(CT_Bias.loc[sig, "EFFECT"], bins=30, alpha=0.7, color="steelblue",
        label=f"FDR < 0.10 (n={sig.sum()})", density=True)
ax.set_xlabel("EFFECT (weighted avg Z2)")
ax.set_ylabel("Density")
ax.set_title("Cell-Type Bias Distribution")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
