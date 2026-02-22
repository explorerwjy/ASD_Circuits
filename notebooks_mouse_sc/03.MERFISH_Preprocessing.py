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
# # 03. MERFISH Preprocessing
#
# Map MERFISH (Multiplexed Error-Robust Fluorescence In Situ Hybridization) cells
# from the Allen Brain Cell Atlas to ISH brain structures used in the connectome.
#
# **Pipeline** (already run, output exists):
# 1. Load CCF v3 parcellation ontology
# 2. Map each MERFISH cell's parcellation structure/substructure to an ISH structure
# 3. Save annotated cell metadata with `ISH_STR` column
#
# **Input**: Raw Allen Brain Cell Atlas MERFISH cell metadata + CCF v3 ontology
#
# **Output**: `dat/MERFISH/MERFISH.ISH_Annot.csv` (~3.7M cells, 39 columns)

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. CCF v3 Ontology and ISH Structure List
#
# The CCF v3 ontology maps MERFISH parcellation abbreviations to ISH brain structure
# names. The ISH structure list (213 structures) comes from the structure-level bias
# analysis.

# %%
CCF_V3_ontology = pd.read_csv(f"../{config['data_files']['merfish_ccf_ontology']}", index_col=0)
print(f"CCF v3 ontology: {CCF_V3_ontology.shape[0]} structures")
CCF_V3_ontology.head(3)

# %%
ASD_STR_Bias = pd.read_csv(f"../{config['data_files']['str_bias_fdr']}", index_col=0)
Structures_ISH = [s.replace("_", " ") for s in ASD_STR_Bias.index.values]
print(f"ISH structures: {len(Structures_ISH)}")

# %% [markdown]
# ## 2. ISH Structure Mapping Function
#
# `Add_ISH_STR` maps each MERFISH cell to an ISH brain structure:
# 1. Look up the cell's `parcellation_structure` in CCF v3 → get `CleanName`
# 2. If that name is in the ISH structure list, use it
# 3. Otherwise try `parcellation_substructure` (catches e.g. AM, IC, MG, MDRN)
# 4. Special cases: VISa/VISrl → "Posterior parietal association areas", Subiculum
# 5. All other cells → "Not in Connectome"

# %%
def Add_ISH_STR(cell_meta, CCF_V3_ontology, Structures_ISH):
    """Map MERFISH cells to ISH brain structures via CCF v3 ontology.

    Parameters
    ----------
    cell_meta : DataFrame
        MERFISH cell metadata with 'parcellation_structure' and
        'parcellation_substructure' columns.
    CCF_V3_ontology : DataFrame
        CCF v3 ontology indexed by abbreviation with 'CleanName' column.
    Structures_ISH : list
        ISH structure names (space-separated) from the connectome.

    Returns
    -------
    DataFrame with added 'ISH_STR' and 'ISH_STR2' columns.
    """
    for i, row in cell_meta.iterrows():
        _str = row["parcellation_structure"]
        _substr = row["parcellation_substructure"]
        name_str = CCF_V3_ontology.loc[_str, "CleanName"] if _str in CCF_V3_ontology.index.values else "None"
        name_substr = CCF_V3_ontology.loc[_substr, "CleanName"] if _substr in CCF_V3_ontology.index.values else "None"

        if name_str in Structures_ISH:
            ISH_STR = name_str
        elif name_substr in Structures_ISH:
            ISH_STR = name_substr
        elif _str in ["VISa", "VISrl"]:
            ISH_STR = "Posterior parietal association areas"
        elif name_str == "Subiculum":
            ISH_STR = "Subiculum"
        else:
            ISH_STR = "Not in Connectome"

        cell_meta.loc[i, "ISH_STR"] = ISH_STR
        cell_meta.loc[i, "ISH_STR2"] = "_".join(ISH_STR.split())
    return cell_meta


# %% [markdown]
# ## 3. Load and Validate MERFISH Annotation
#
# The annotation was produced by running `Add_ISH_STR` on all Allen Brain Cell Atlas
# MERFISH cells (4 Zhuang Lab datasets, ABCA-1 through ABCA-4, combined).

# %%
merfish_csv = f"../{config['data_files']['merfish_annotation']}"
merfish_parquet = f"../{config['data_files']['merfish_annotation_parquet']}"

if os.path.exists(merfish_parquet):
    MERFISH = pd.read_parquet(merfish_parquet)
    print(f"Loaded from parquet: {merfish_parquet}")
else:
    MERFISH = pd.read_csv(merfish_csv)
    print(f"Loaded from CSV: {merfish_csv}")
print(f"Shape: {MERFISH.shape}")

# %%
# Basic validation
assert "ISH_STR" in MERFISH.columns, "Missing ISH_STR column"
n_mapped = (MERFISH["ISH_STR"] != "Not in Connectome").sum()
n_total = len(MERFISH)
print(f"Total cells: {n_total:,}")
print(f"Mapped to ISH structures: {n_mapped:,} ({100*n_mapped/n_total:.1f}%)")
print(f"Not in connectome: {n_total - n_mapped:,} ({100*(n_total-n_mapped)/n_total:.1f}%)")

# %%
# Cells per ISH structure
str_counts = MERFISH[MERFISH["ISH_STR"] != "Not in Connectome"]["ISH_STR"].value_counts()
print(f"ISH structures with cells: {len(str_counts)}")
print(f"\nTop 20 structures by cell count:")
str_counts.head(20)

# %%
# Cell class distribution
class_counts = MERFISH["class"].value_counts()
print("Cell class distribution:")
class_counts

# %% [markdown]
# ## 4. Save Parquet Version

# %%
if not os.path.exists(merfish_parquet):
    # Fix mixed-type columns (cluster_alias has both int and str values)
    for col in MERFISH.select_dtypes(include=["object"]).columns:
        MERFISH[col] = MERFISH[col].astype(str)
    MERFISH.to_parquet(merfish_parquet, index=False)
    print(f"Saved parquet: {merfish_parquet}")
    parquet_size = os.path.getsize(merfish_parquet) / 1e6
    csv_size = os.path.getsize(merfish_csv) / 1e6
    print(f"CSV: {csv_size:.0f} MB → Parquet: {parquet_size:.0f} MB")
else:
    print(f"Parquet already exists: {merfish_parquet}")
