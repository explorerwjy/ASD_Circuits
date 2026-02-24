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
# Prepare MERFISH data for structure-level analysis:
# 1. Map MERFISH cells to ISH brain structures (via CCF v3 ontology)
# 2. Compute Z1 matrices (Allen + Zhuang MERFISH)
# 3. Assemble Z2 matrices from pre-computed splits
# 4. Compute expression matching quantiles

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
from CellType_PSY import *

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

# %% [markdown]
# ## 5. Allen MERFISH Z1 Matrices
#
# Compute Z1-normalized expression matrices from raw UMI counts for both
# all-cell and neuron-only aggregations (cell-mean and volume-weighted).

# %%
# Allen MERFISH — all cells
MERFISH_CellMeanExp = pd.read_csv(f"../{config['data_files']['merfish_cell_mean_umi']}", index_col=0)
MERFISH_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_vol_mean_umi']}", index_col=0)
print(f"Allen MERFISH Cell-mean: {MERFISH_CellMeanExp.shape}")
print(f"Allen MERFISH Vol-mean:  {MERFISH_VolMeanExp.shape}")

MERFISH_CellMean_Z1 = Z1Conversion(MERFISH_CellMeanExp, "../dat/MERFISH/STR_Cell_Mean_Z1Mat.csv")
MERFISH_VolMean_Z1 = Z1Conversion(MERFISH_VolMeanExp, "../dat/MERFISH/STR_Vol_Mean_Z1Mat.csv")

MERFISH_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Cell_Mean_Z1Mat.clip.csv")
MERFISH_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Vol_Mean_Z1Mat.clip.csv")

# %%
# Allen MERFISH — neuron-only
MERFISH_NEU_MeanExp = pd.read_csv(f"../{config['data_files']['merfish_neur_mean_umi']}", index_col=0)
MERFISH_NEU_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_neur_vol_mean_umi']}", index_col=0)
print(f"Allen MERFISH Neuron Cell-mean: {MERFISH_NEU_MeanExp.shape}")
print(f"Allen MERFISH Neuron Vol-mean:  {MERFISH_NEU_VolMeanExp.shape}")

MERFISH_NEU_Z1 = Z1Conversion(MERFISH_NEU_MeanExp, "../dat/MERFISH/STR_NEU_Mean_Z1Mat.csv")
MERFISH_NEU_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Mean_Z1Mat.clip.csv")

MERFISH_NEU_Vol_Z1 = Z1Conversion(MERFISH_NEU_VolMeanExp, "../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.csv")
MERFISH_NEU_Vol_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.clip.csv")

# %% [markdown]
# ## 6. Zhuang/MIT MERFISH Z1 Matrices
#
# Same Z1 normalization for the Zhuang Lab MERFISH dataset (if available).

# %%
if os.path.exists(f"../{config['data_files']['merfish_zhuang_cell_mean_umi']}"):
    Zhuang_CellMeanExp = pd.read_csv(f"../{config['data_files']['merfish_zhuang_cell_mean_umi']}", index_col=0)
    Zhuang_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_zhuang_vol_mean_umi']}", index_col=0)
    print(f"Zhuang MERFISH Cell-mean: {Zhuang_CellMeanExp.shape}")
    print(f"Zhuang MERFISH Vol-mean:  {Zhuang_VolMeanExp.shape}")

    Zhuang_CellMean_Z1 = Z1Conversion(Zhuang_CellMeanExp, "../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.csv")
    Zhuang_VolMean_Z1 = Z1Conversion(Zhuang_VolMeanExp, "../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.csv")

    Zhuang_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.clip.csv")
    Zhuang_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.clip.csv")
else:
    print("Zhuang MERFISH data not found — skipping")

# %% [markdown]
# ## 7. MERFISH Z2 Matrices (from pre-computed splits)
#
# Z2 normalization is ISH expression-matched. The Z2 computation was run externally
# and split across multiple CSV files. Here we reassemble them into single matrices.

# %%
Z2_SPLIT_BASE = config["data_files"]["z2_split_base"]


def assemble_z2_splits(split_dir, outpath):
    """Concatenate Z2 split CSVs into a single matrix."""
    if not os.path.isdir(split_dir):
        print(f"  SKIP (not found): {split_dir}")
        return None
    dfs = []
    for f in sorted(os.listdir(split_dir)):
        dfs.append(pd.read_csv(os.path.join(split_dir, f), index_col=0))
    z2 = pd.concat(dfs)
    z2.to_csv(outpath)
    print(f"  {outpath}: {z2.shape}")
    return z2


# Allen MERFISH Z2
print("Allen MERFISH Z2:")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_CellMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_Cell_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_VolMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_Vol_Mean_Z2Mat_ISHMatch.csv")

# Allen MERFISH — neuron-only Z2
print("Allen MERFISH Neuron Z2:")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_NEU_Mean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_NEUR_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_NEU_Vol_Mean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_NEUR_Vol_Mean_Z2Mat_ISHMatch.csv")

# Zhuang MERFISH Z2
print("Zhuang MERFISH Z2:")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_MIT_CellMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH_Zhuang/STR_Cell_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_MIT_VolMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH_Zhuang/STR_Vol_Mean_Z2Mat_ISHMatch.csv")

# %% [markdown]
# ## 8. MERFISH Expression Matching Quantiles
#
# Compute per-gene expression quantiles across all MERFISH clusters. These quantiles
# are used for ISH expression-level matching during Z2 normalization.

# %%
ClusterExpDF = pd.read_csv(f"../{config['data_files']['cluster_mean_log_umi_csv']}", index_col=0)
MERFISH_STRAnn = pd.read_csv(f"../{config['data_files']['merfish_annotation']}")

Total_Exp_Genes = np.zeros(ClusterExpDF.shape[0])
matched_clusters = 0
for _, row in MERFISH_STRAnn.iterrows():
    cluster = row.get("cluster")
    if cluster is not None and cluster in ClusterExpDF.columns:
        Total_Exp_Genes += ClusterExpDF[cluster].values
        matched_clusters += 1
print(f"Matched {matched_clusters} MERFISH entries to clusters")

WB_ExpDF = pd.DataFrame(Total_Exp_Genes, index=ClusterExpDF.index, columns=["TotalExp"])
WB_ExpDF = WB_ExpDF.sort_values("TotalExp")
WB_ExpDF["Rank"] = range(1, len(WB_ExpDF) + 1)
WB_ExpDF["quantile"] = WB_ExpDF["Rank"] / len(WB_ExpDF)
WB_ExpDF.to_csv("../dat/MERFISH/MouseMERFISHGeneMatchQuantile.csv")
print(f"Saved expression quantiles: {WB_ExpDF.shape}")
