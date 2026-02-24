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
# # 06. Bias Display — Spatial Visualization of Cell-Type ASD Bias
#
# Map cluster-level ASD bias onto individual MERFISH cells and visualize
# on brain section coordinates, showing spatial distribution of ASD mutation bias.
#
# **Input**:
# - MERFISH annotation (from 03): `dat/MERFISH/MERFISH.ISH_Annot.parquet`
# - Cell-type bias with p-values (from 05): `dat/Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.addP.csv`
#
# **Output**: `dat/MERFISH/MERFISH.cells.ASD.Bias.Anno.parquet`
# - MERFISH cells annotated with `ASD.Bias` (raw) and `ASD.Bias.adj` (mean-adjusted)

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. Load & Annotate MERFISH Cells with ASD Bias
#
# For each MERFISH cell, assign the ASD bias of its cluster (from the addP file).
# Cache the result as a parquet file for fast reloading.

# %%
MERFISH_BIAS_FILE = f"../{config['data_files']['merfish_cells_bias_annotated']}"

if os.path.exists(MERFISH_BIAS_FILE):
    MERFISH = pd.read_parquet(MERFISH_BIAS_FILE)
    print(f"Loaded cached annotated MERFISH: {MERFISH.shape[0]:,} cells")
else:
    # Load MERFISH annotation
    merfish_parquet = f"../{config['data_files']['merfish_annotation_parquet']}"
    MERFISH = pd.read_parquet(merfish_parquet)

    # Load cluster-level bias
    addP_path = f"../{config['data_files']['ct_bias_addp']}"
    CT_Bias = pd.read_csv(addP_path, index_col=0)

    # Map cluster bias to each cell
    bias_map = CT_Bias["EFFECT"].to_dict()
    bias_adj_map = CT_Bias["EFFECT2"].to_dict()
    MERFISH["ASD.Bias"] = MERFISH["cluster"].map(bias_map).fillna(0)
    MERFISH["ASD.Bias.adj"] = MERFISH["cluster"].map(bias_adj_map).fillna(0)

    # Ensure numeric coordinates
    MERFISH["x_reconstructed"] = pd.to_numeric(MERFISH["x_reconstructed"], errors="coerce")
    MERFISH["y_reconstructed"] = pd.to_numeric(MERFISH["y_reconstructed"], errors="coerce")

    MERFISH.to_parquet(MERFISH_BIAS_FILE)
    print(f"Created annotated MERFISH: {MERFISH.shape[0]:,} cells, saved to {MERFISH_BIAS_FILE}")

# %%
print(f"ASD.Bias range: [{MERFISH['ASD.Bias'].min():.3f}, {MERFISH['ASD.Bias'].max():.3f}]")
print(f"Cells with non-zero bias: {(MERFISH['ASD.Bias'] != 0).sum():,} / {len(MERFISH):,}")

# %% [markdown]
# ## 2. Spatial Plotting Function

# %%
from alpha_shapes import Alpha_Shaper
from alpha_shapes.boundary import get_boundaries


def plot_section_bias(section_df, structures, title, vmin=-0.5, vmax=0.5, dpi=300,
                      show_labels=True):
    """Plot ASD bias on a single brain section with structure boundaries."""
    fig, ax = plt.subplots(dpi=dpi, figsize=(10, 8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    sc = ax.scatter(
        section_df["x_reconstructed"],
        section_df["y_reconstructed"],
        c=section_df["ASD.Bias"],
        cmap="coolwarm", s=0.5, alpha=0.7, edgecolor="none",
        vmin=vmin, vmax=vmax,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("ASD Bias", fontsize=14)

    # Draw structure boundaries and labels
    for reg in structures:
        sub = section_df[section_df["parcellation_structure"] == reg]
        if len(sub) < 10:
            continue
        points = sub[["x_reconstructed", "y_reconstructed"]].values
        try:
            shaper = Alpha_Shaper(points)
            alpha_opt, _ = shaper.optimize()
            alpha_shape = shaper.get_shape(alpha=alpha_opt * 0.6)
            for bound in get_boundaries(alpha_shape):
                ext = bound._exterior
                ax.plot(ext[:, 0], ext[:, 1], color="black", ls="--", lw=1.2)
            if show_labels:
                left_mask = points[:, 0] < np.median(points[:, 0])
                center = points[left_mask].mean(axis=0) if left_mask.any() else points.mean(axis=0)
                if np.isnan(center).any():
                    center = points.mean(axis=0)
                ax.text(center[0] - 0.3, center[1], reg, fontsize=8, fontweight="bold")
        except Exception:
            pass

    ax.set_xlabel("X Reconstructed")
    ax.set_ylabel("Y Reconstructed")
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# %% [markdown]
# ## 3. Representative Brain Sections
#
# Show six sections highlighting key brain regions:
# - **Section 38**: Thalamus (MD), hippocampus (CA1, DG)
# - **Section 51**: Striatum (ACB, CP), cortex (MOp, SSp)
# - **Section 36**: Amygdala (BLA, LA, MEA), temporal cortex
# - **Section 56**: Prefrontal cortex (PL, ILA, MOs)
# - **Section 15**: Midbrain (VTA, IC), hindbrain (PB, PRNc)
# - **Section 14**: Cerebellum (CENT, CUL, SIM, AN) — negative control

# %%
sections = {
    "C57BL6J-638850.38": {
        "title": "Section 38 — Thalamus & Hippocampus",
        "structures": ["LP", "MD", "CP", "CA1", "SSs", "VISa", "DG", "MEA", "PIR", "RSPv", "RE"],
    },
    "C57BL6J-638850.51": {
        "title": "Section 51 — Striatum & Cortex",
        "structures": ["CP", "SSp-m", "ACB", "MOp", "OT", "PIR", "MOs", "SSp-ul", "LSr",
                        "ACAv", "ACAd", "SSs", "GU", "AId"],
    },
    "C57BL6J-638850.36": {
        "title": "Section 36 — Amygdala & Temporal Cortex",
        "structures": ["DG", "CA1", "RSPv", "MEA", "AUDp", "AUDv", "AUDd", "VISam", "SSs",
                        "TEa", "VISrl", "CA3", "PIR", "PF", "LA", "RSPd", "BLA", "BMA"],
    },
    "C57BL6J-638850.56": {
        "title": "Section 56 — Prefrontal Cortex",
        "structures": ["MOp", "MOs", "PIR", "PL", "ILA", "AId", "ORBl", "AON", "ACAd", "TT"],
    },
    "C57BL6J-638850.15": {
        "title": "Section 15 — Midbrain & Hindbrain (VTA)",
        "structures": ["IC", "PRNc", "VCO", "PB", "PSV", "PCG", "PARN", "MV", "GRN",
                        "AN", "CENT", "SIM", "PFL", "CUL", "FL"],
    },
    "C57BL6J-638850.14": {
        "title": "Section 14 — Cerebellum (Negative Control)",
        "structures": ["CENT", "CUL", "SIM", "AN", "PFL", "FL", "MV", "GRN", "VCO",
                        "PARN", "PCG", "DCO"],
    },
}

for section_label, info in sections.items():
    section_df = MERFISH[MERFISH["brain_section_label"] == section_label]
    print(f"\n{info['title']}: {len(section_df):,} cells")
    fig = plot_section_bias(section_df, info["structures"], info["title"])
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 4. Regional Bias Distribution

# %%
select_regions = ["Isocortex", "STR", "HPF", "TH", "HY", "MB", "OLF", "PAL",
                  "CTXsp", "P", "MY", "CB"]

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

region_data = MERFISH[MERFISH["parcellation_division"].isin(select_regions)]
# Compute mean bias per region for ordering
region_means = region_data.groupby("parcellation_division")["ASD.Bias"].mean()
region_order = region_means.sort_values(ascending=False).index.tolist()

sns.violinplot(data=region_data, x="parcellation_division", y="ASD.Bias",
               order=region_order, cut=0, inner="quartile", ax=ax)
ax.axhline(0, color="grey", ls="--", lw=0.5)
ax.set_xlabel("Brain Division")
ax.set_ylabel("ASD Bias (cell-level)")
ax.set_title("Cell-Level ASD Bias by Brain Region")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
