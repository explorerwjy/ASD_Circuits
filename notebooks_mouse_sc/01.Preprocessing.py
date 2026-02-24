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
# # Preprocessing: Cluster-Level Z1 and Z2 Expression Specificity Matrices
#
# This notebook builds Z1 (gene-level z-score) and Z2 (expression-matched z-score)
# matrices for **Allen Brain Cell Atlas clusters** (5312 clusters, 17938 genes).
#
# **Pipeline:**
# 1. Load cluster expression matrix (from `scripts/build_celltype_z2_matrix.py --stage1`)
# 2. Z1 normalization (z-score each gene across clusters, clip to +/-3)
# 3. Z2 calculation (ISH expression-matched normalization, parallelized)
# 4. Save output to `dat/BiasMatrices/`

# %% [markdown]
# ## Section 0: Setup

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import yaml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from CellType_PSY import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## Section 1: Load Cluster Expression Matrix
#
# Input: `cluster_MeanLogUMI.csv` (17938 genes x 5312 clusters)
# produced by `scripts/build_celltype_z2_matrix.py --stage1`.

# %%
ClusterExpDF = pd.read_csv(f"../{config['data_files']['cluster_mean_log_umi_csv']}", index_col=0)
print(f"Loaded cluster expression: {ClusterExpDF.shape}")

# %% [markdown]
# ## Section 2: Z1 Normalization
#
# Z-score each gene (row) across clusters (columns), then clip to +/-3.
# No CSV output -- kept in memory for Z2 computation.

# %%
ClusterZ1 = Z1Conversion(ClusterExpDF)
ClusterZ1_clip3 = ClusterZ1.clip(upper=3, lower=-3)
print(f"Z1 matrix: {ClusterZ1_clip3.shape}, range: [{ClusterZ1_clip3.min().min():.1f}, {ClusterZ1_clip3.max().max():.1f}]")

# %% [markdown]
# ## Section 3: Z2 Calculation (ISH Expression-Matched)
#
# For each gene, load its ISH expression-matched gene set (up to 1000 genes with
# similar whole-brain expression levels). Compute Z2 as the z-score of that gene's
# Z1 values relative to the mean and std of its matched set.
#
# Parallelized across gene chunks using joblib (10 workers).

# %%
ISH_MATCH_DIR = "/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"


def _load_match_genes(entrez_id, match_dir):
    """Load expression-matched gene list for a given Entrez ID."""
    fpath = os.path.join(match_dir, f"{entrez_id}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        genes = [int(df.columns[0])] + [int(x) for x in df.iloc[:, 0].values]
        return genes
    except Exception:
        return None


def _z2_gene_chunk(gene_indices, z1_values, z1_index, z1_columns,
                   match_dir, index_to_entrez, entrez_to_row, max_genes=1000):
    """Compute Z2 for a chunk of genes."""
    result_indices = []
    result_rows = []
    for idx in gene_indices:
        if idx not in index_to_entrez:
            continue
        entrez = index_to_entrez[idx]
        match_genes = _load_match_genes(entrez, match_dir)
        if match_genes is None:
            continue
        # Filter to Z1-present genes first, THEN limit to 1000 (matches legacy order)
        match_rows = [entrez_to_row[g] for g in match_genes if g in entrez_to_row][:max_genes]
        if len(match_rows) < 2:
            continue
        gene_row = z1_values[entrez_to_row[entrez]]
        match_vals = z1_values[match_rows]
        match_mean = np.nanmean(match_vals, axis=0)
        match_std = np.nanstd(match_vals, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            z2_row = (gene_row - match_mean) / match_std
        z2_row[~np.isfinite(z2_row)] = np.nan
        result_indices.append(entrez)
        result_rows.append(z2_row)
    return result_indices, result_rows


# Build lookup tables
z1_values = ClusterZ1_clip3.values
z1_index = ClusterZ1_clip3.index.values
entrez_to_row = {int(z1_index[i]): i for i in range(len(z1_index))}
index_to_entrez = {i: int(z1_index[i]) for i in range(len(z1_index))}

# Split into chunks and run in parallel
chunk_size = 500
n_genes = len(z1_index)
chunks = [range(i, min(i + chunk_size, n_genes)) for i in range(0, n_genes, chunk_size)]

results = Parallel(n_jobs=10, verbose=5)(
    delayed(_z2_gene_chunk)(
        list(chunk), z1_values, z1_index, ClusterZ1_clip3.columns.values,
        ISH_MATCH_DIR, index_to_entrez, entrez_to_row
    )
    for chunk in chunks
)

# Assemble Z2 matrix
all_indices, all_rows = [], []
for indices, rows in results:
    all_indices.extend(indices)
    all_rows.extend(rows)

ClusterZ2 = pd.DataFrame(
    data=np.array(all_rows),
    index=np.array(all_indices),
    columns=ClusterZ1_clip3.columns
)
ClusterZ2.index.name = None
print(f"Z2 matrix: {ClusterZ2.shape}")
print(f"NaN count: {ClusterZ2.isna().sum().sum()}")

# %% [markdown]
# ## Section 4: Save Output

# %%
OUT_PATH = "../dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet"
ClusterZ2.to_parquet(OUT_PATH)
print(f"Saved: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
