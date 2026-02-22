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
#     display_name: Python 3 (gencic)
#     language: python
#     name: gencic
# ---

# %% [markdown]
# # Preprocessing: Z1 and Z2 Expression Specificity Matrices
#
# This notebook builds Z1 (gene-level z-score) and Z2 (expression-matched z-score)
# matrices for:
# 1. **Allen Brain Cell Atlas clusters** — using `cluster_MeanLogUMI` from Stage 1
# 2. **Allen MERFISH spatial data** — cell-mean and volume-mean variants
# 3. **Zhuang/MIT MERFISH** — second independent MERFISH dataset
#
# **Cluster Z2** is now produced by `scripts/build_celltype_z2_matrix.py` (stages 2-3).
# This notebook handles Z1 for reference and all MERFISH Z1/Z2 processing.

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
sys.path.insert(1, '../src')
from CellType_PSY import *

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %% [markdown]
# ## 1. Cluster-Level Z1 Matrices
#
# Input: `dat/SC_UMI_Mats/cluster_MeanLogUMI.csv` (17938 genes x 5312 clusters)
# produced by `scripts/build_celltype_z2_matrix.py --stage1`.
#
# The Z2 matrix (`Cluster_Z2Mat_ISHMatch.z1clip3.parquet`) is also produced by
# that script (stages 2-3). We keep Z1 CSV outputs here for compatibility.

# %%
ClusterExpDF = pd.read_csv("../dat/SC_UMI_Mats/cluster_MeanLogUMI.csv", index_col=0)
print(f"Loaded cluster expression: {ClusterExpDF.shape}")

# Z1 conversion (z-score each gene across clusters)
ClusterZ1 = Z1Conversion(ClusterExpDF, "../dat/SC_UMI_Mats/cluster_V3_Z1Mat.csv")

# Clip at +/-3 (used for Z2 calculation)
ClusterZ1_clip3 = ClusterZ1.clip(upper=3, lower=-3)
ClusterZ1_clip3.to_csv("../dat/SC_UMI_Mats/cluster_V3_Z1Mat.clip3.csv")
print(f"Saved Z1 and Z1.clip3: {ClusterZ1.shape}")

# %% [markdown]
# ## 2. Allen MERFISH Z1 Matrices
#
# MERFISH provides spatially-resolved gene expression. We compute Z1 for:
# - Cell-mean and volume-mean (all cells)
# - Neuron-only cell-mean and volume-mean

# %%
# Allen MERFISH — all cells
MERFISH_CellMeanExp = pd.read_csv("../dat/MERFISH/STR_Cell_Mean_DF.UMI.csv", index_col=0)
MERFISH_VolMeanExp = pd.read_csv("../dat/MERFISH/STR_Vol_Mean_DF.UMI.csv", index_col=0)
print(f"Allen MERFISH Cell-mean: {MERFISH_CellMeanExp.shape}")
print(f"Allen MERFISH Vol-mean:  {MERFISH_VolMeanExp.shape}")

MERFISH_CellMean_Z1 = Z1Conversion(MERFISH_CellMeanExp, "../dat/MERFISH/STR_Cell_Mean_Z1Mat.csv")
MERFISH_VolMean_Z1 = Z1Conversion(MERFISH_VolMeanExp, "../dat/MERFISH/STR_Vol_Mean_Z1Mat.csv")

MERFISH_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Cell_Mean_Z1Mat.clip.csv")
MERFISH_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Vol_Mean_Z1Mat.clip.csv")

# %%
# Allen MERFISH — neuron-only
MERFISH_NEU_MeanExp = pd.read_csv("../dat/MERFISH/STR_NEU_Mean_DF.UMI.csv", index_col=0)
MERFISH_NEU_VolMeanExp = pd.read_csv("../dat/MERFISH/STR_NEU_Vol_Mean_DF.UMI.csv", index_col=0)
print(f"Allen MERFISH Neuron Cell-mean: {MERFISH_NEU_MeanExp.shape}")
print(f"Allen MERFISH Neuron Vol-mean:  {MERFISH_NEU_VolMeanExp.shape}")

MERFISH_NEU_Z1 = Z1Conversion(MERFISH_NEU_MeanExp, "../dat/MERFISH/STR_NEU_Mean_Z1Mat.csv")
MERFISH_NEU_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Mean_Z1Mat.clip.csv")

MERFISH_NEU_Vol_Z1 = Z1Conversion(MERFISH_NEU_VolMeanExp, "../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.csv")
MERFISH_NEU_Vol_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.clip.csv")

# %% [markdown]
# ## 3. Zhuang/MIT MERFISH Z1 Matrices

# %%
if os.path.exists("../dat/MERFISH_Zhuang/STR_Cell_Mean_DF.UMI.csv"):
    Zhuang_CellMeanExp = pd.read_csv("../dat/MERFISH_Zhuang/STR_Cell_Mean_DF.UMI.csv", index_col=0)
    Zhuang_VolMeanExp = pd.read_csv("../dat/MERFISH_Zhuang/STR_Vol_Mean_DF.UMI.csv", index_col=0)
    print(f"Zhuang MERFISH Cell-mean: {Zhuang_CellMeanExp.shape}")
    print(f"Zhuang MERFISH Vol-mean:  {Zhuang_VolMeanExp.shape}")

    Zhuang_CellMean_Z1 = Z1Conversion(Zhuang_CellMeanExp, "../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.csv")
    Zhuang_VolMean_Z1 = Z1Conversion(Zhuang_VolMeanExp, "../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.csv")

    Zhuang_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.clip.csv")
    Zhuang_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.clip.csv")
else:
    print("Zhuang MERFISH data not found — skipping")

# %% [markdown]
# ## 4. MERFISH Z2 Matrices (from pre-computed split files)
#
# Z2 split files were pre-computed from Z1 matrices using ISH expression-matched
# gene sets. We assemble them into complete Z2 matrices here.

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
# ## 5. MERFISH Expression Matching Quantiles
#
# Compute per-gene total expression across MERFISH-annotated clusters,
# for use in expression-matched null comparisons.

# %%
MERFISH_STRAnn = pd.read_csv("../dat/MERFISH/MERFISH.ISH_Annot.csv")

Total_Exp_Genes = np.zeros(ClusterExpDF.shape[0])
matched_clusters = 0
for _, row in MERFISH_STRAnn.iterrows():
    cluster = row["cluster"] if "cluster" in row.index else None
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
