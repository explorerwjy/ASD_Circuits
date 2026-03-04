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
# # 04. MERFISH Structure-Level Bias
#
# Compute ASD mutation bias at the brain structure level using MERFISH spatial
# transcriptomics data, and validate by comparing with ISH-derived bias.
#
# Gene weights are denoised using V2-V3 single-cell reproducibility (V2_V3_CT_Corr²).
# Best validation: Cell Mean (CM) Pearson r = 0.77 vs ISH.
#
# **Input**: MERFISH Z2 expression matrices (from 03.MERFISH_Preprocessing), DN gene weights
#
# **Output**: Structure-level bias files in `dat/Bias/STR/`
# - `ASD.MERFISH_Allen.CM.ISHMatch.Z2.csv` — Cell Mean (r=0.77 vs ISH)
# - `ASD.MERFISH_Allen.VM.ISHMatch.Z2.csv` — Volume Mean
# - `ASD.MERFISH_Allen.NM.ISHMatch.Z2.csv` — Neuron Mean
# - `ASD.MERFISH_Allen.NVM.ISHMatch.Z2.csv` — Neuron Volume Mean

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. Load MERFISH Z2 Expression Matrices
#
# Four structure-level aggregation methods, all ISH-expression-matched Z2:
# - **Cell Mean (CM)**: Average expression across all cells per structure
# - **Volume Mean (VM)**: Expression weighted by cell volume per structure
# - **Neuron Mean (NM)**: Average expression across neuronal cells only
# - **Neuron Volume Mean (NVM)**: Neuron expression weighted by volume

# %%
z2_files = {
    "CM": config["data_files"]["merfish_z2_cell_mean"],
    "VM": config["data_files"]["merfish_z2_vol_mean"],
    "NM": config["data_files"]["merfish_z2_neur_mean"],
    "NVM": config["data_files"]["merfish_z2_neur_vol_mean"],
}

Z2_mats = {}
for label, path in z2_files.items():
    df = pd.read_csv(f"../{path}", index_col=0)
    # Convert space-separated column names to underscore-separated (for str2reg)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    Z2_mats[label] = df
    print(f"{label}: {df.shape[0]} genes x {df.shape[1]} structures")

# %% [markdown]
# ## 2. Load Gene Weights and Compute Bias
#
# Gene weights are denoised using V2-V3 single-cell cross-cell-type correlation
# (weight_DN = weight_ISH × V2_V3_CT_Corr²). This downweights genes with poor
# reproducibility across independent scRNA-seq experiments.

# %%
ASD_GW = Fil2Dict(f"../{config['data_files']['asd_gene_weights_dn']}")
print(f"ASD gene weights (DN): {len(ASD_GW)} genes")

# %%
BIAS_DIR = "dat/Bias/STR"
os.makedirs(BIAS_DIR, exist_ok=True)

output_names = {
    "CM": "ASD.MERFISH_Allen.CM.ISHMatch.Z2.csv",
    "VM": "ASD.MERFISH_Allen.VM.ISHMatch.Z2.csv",
    "NM": "ASD.MERFISH_Allen.NM.ISHMatch.Z2.csv",
    "NVM": "ASD.MERFISH_Allen.NVM.ISHMatch.Z2.csv",
}

bias_results = {}
for label, z2_mat in Z2_mats.items():
    outpath = f"{BIAS_DIR}/{output_names[label]}"
    bias_df = MouseSTR_AvgZ_Weighted(z2_mat, ASD_GW, csv_fil=outpath)
    bias_results[label] = bias_df
    print(f"{label}: top EFFECT = {bias_df['EFFECT'].iloc[0]:.4f} at {bias_df.index[0]}")

# %% [markdown]
# ## 3. Validate Against ISH Bias

# %%
ASD_ISH_Bias = pd.read_csv(f"../{config['data_files']['str_bias_fdr']}", index_col=0)
# Merge Subiculum parts
if "Subiculum_dorsal_part" in ASD_ISH_Bias.index:
    sub_d = ASD_ISH_Bias.loc["Subiculum_dorsal_part", "EFFECT"]
    sub_v = ASD_ISH_Bias.loc["Subiculum_ventral_part", "EFFECT"]
    ASD_ISH_Bias.loc["Subiculum"] = [(sub_d + sub_v) / 2, "Hippocampus",
                                      214, 1, 0, 1]
    ASD_ISH_Bias = ASD_ISH_Bias.drop(
        ["Subiculum_dorsal_part", "Subiculum_ventral_part"])

# %%
print("MERFISH vs ISH bias correlation:")
print(f"{'Method':<6} {'Pearson r':>10} {'Spearman r':>12} {'Top-50 overlap':>16}")
print("-" * 48)
for label, bias_df in bias_results.items():
    shared = bias_df.index.intersection(ASD_ISH_Bias.index)
    r_p, _ = pearsonr(ASD_ISH_Bias.loc[shared, "EFFECT"],
                       bias_df.loc[shared, "EFFECT"])
    r_s, _ = spearmanr(ASD_ISH_Bias.loc[shared, "EFFECT"],
                        bias_df.loc[shared, "EFFECT"])
    top50_ish = set(ASD_ISH_Bias.head(50).index)
    top50_mf = set(bias_df.head(50).index)
    overlap = len(top50_ish & top50_mf)
    print(f"{label:<6} {r_p:>10.4f} {r_s:>12.4f} {overlap:>16}")

# %% [markdown]
# ## 4. Compare Aggregation Methods

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
fig.patch.set_alpha(0)

comparisons = [("CM", "VM"), ("NM", "NVM"), ("CM", "NM")]
for ax, (a, b) in zip(axes, comparisons):
    ax.patch.set_alpha(0)
    shared = bias_results[a].index.intersection(bias_results[b].index)
    x = bias_results[a].loc[shared, "EFFECT"]
    y = bias_results[b].loc[shared, "EFFECT"]
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=8, alpha=0.5, c="steelblue")
    ax.set_xlabel(f"{a} bias")
    ax.set_ylabel(f"{b} bias")
    ax.set_title(f"r = {r:.3f}")
    ax.axhline(0, c="grey", ls="--", lw=0.5)
    ax.axvline(0, c="grey", ls="--", lw=0.5)

plt.suptitle("MERFISH Structure Bias: Aggregation Comparison", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. MERFISH vs ISH Scatter — All Methods
#
# Compare all four MERFISH aggregation methods against ISH bias.

# %%
method_labels = {"CM": "Cell Mean", "VM": "Volume Mean",
                 "NM": "Neuron Mean", "NVM": "Neuron Vol Mean"}
region_colors = {
    "Isocortex": "#1f77b4", "Hippocampus": "#2ca02c",
    "Thalamus": "#ff7f0e", "Hypothalamus": "#d62728",
    "Striatum": "#9467bd", "Midbrain": "#8c564b",
    "Cerebellum": "#e377c2", "Medulla": "#7f7f7f",
    "Pons": "#bcbd22", "Pallidum": "#17becf",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
fig.patch.set_alpha(0)

for ax, (key, full_name) in zip(axes.flat, method_labels.items()):
    ax.patch.set_alpha(0)
    bias_df = bias_results[key]
    shared = bias_df.index.intersection(ASD_ISH_Bias.index)
    x = ASD_ISH_Bias.loc[shared, "EFFECT"]
    y = bias_df.loc[shared, "EFFECT"]
    r, p = pearsonr(x, y)
    regions = ASD_ISH_Bias.loc[shared, "REGION"]

    for reg in regions.unique():
        mask = regions == reg
        color = region_colors.get(reg, "grey")
        ax.scatter(x[mask], y[mask], s=15, alpha=0.6, c=color, label=reg)

    ax.set_xlabel("ISH Z2 bias")
    ax.set_ylabel(f"MERFISH {key} Z2 bias")
    ax.set_title(f"{full_name} — r = {r:.3f}")
    ax.axhline(0, c="grey", ls="--", lw=0.5)
    ax.axvline(0, c="grey", ls="--", lw=0.5)

# Single legend from first axis
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.show()
