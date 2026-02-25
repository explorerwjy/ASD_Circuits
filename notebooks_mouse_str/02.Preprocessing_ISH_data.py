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
import subprocess
import numpy as np
import pandas as pd

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import ZscoreConverting

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")

# %%
# Load config for data paths
with open(os.path.join(ProjDIR, "config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

BIAS_DIR = os.path.join(ProjDIR, "dat/BiasMatrices")
ALLEN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp")
os.makedirs(BIAS_DIR, exist_ok=True)

# %% [markdown]
# # 02. Preprocessing ISH Expression Data
#
# This notebook transforms the expression energy matrix into the Z2
# expression-matched z-score matrix used for all structure-level bias analyses.
#
# ## Pipeline
#
# 1. **Load** Jon's log2+QN expression matrix (R-generated, 17,208 genes × 213 structures)
# 2. **Z1** — per-gene z-score across structures
# 3. **Expression features** — quantile ranks for expression matching
# 4. **Z2** — expression-matched z-score (parallelized via `script_compute_Z2.py`)
#
# **Why Jon's R QN?** Python's quantile normalization differs from R's
# `preprocessCore::normalize.quantiles` in tie-breaking (r=0.993 between them).
# This difference propagates through Z1→Z2→bias and shifts the CCS local peak
# away from size 46, the circuit size used in the paper.
# Using Jon's R-generated QN preserves the exact CCS profile.
# See `notebook_validation/Validate_ISH_Z2_Pipeline` for the comparison.
#
# **Input**: `dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv`
#
# **Output**: `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet`

# %% [markdown]
# ## 1. Load Expression Matrix

# %%
# Jon's log2+QN expression matrix (R-generated)
# Raw expression: arithmetic mean of ISH expression energy across sections per gene
# Then: log2(1+x) → R quantile normalization (preprocessCore::normalize.quantiles)
jon_qn_path = os.path.join(ProjDIR, config["data_files"]["jon_exp_log2_qn"])
ExpMat = pd.read_csv(jon_qn_path, index_col="ROW")
print(f"Expression matrix (log2+QN): {ExpMat.shape}")
print(f"  Value range: [{np.nanmin(ExpMat.values):.4f}, {np.nanmax(ExpMat.values):.4f}]")
print(f"  NaN fraction: {ExpMat.isna().sum().sum() / ExpMat.size:.4f}")

# %% [markdown]
# ## 2. Compute Z1 Matrix (Per-Gene Z-Score)
#
# For each gene, z-score its expression values across the 213 structures.
# This normalizes each gene to have mean=0, std=1 across structures.

# %%
z1_path = os.path.join(BIAS_DIR, "AllenMouseBrain_Z1.parquet")

z1_data = []
z1_genes = []
for gene in ExpMat.index:
    z1 = ZscoreConverting(ExpMat.loc[gene].values)
    if not np.all(np.isnan(z1)):
        z1_data.append(z1)
        z1_genes.append(gene)

Z1Mat = pd.DataFrame(
    data=np.array(z1_data),
    index=z1_genes,
    columns=ExpMat.columns
)
Z1Mat.index.name = None

Z1Mat.to_parquet(z1_path)
print(f"Z1 matrix: {Z1Mat.shape}")
print(f"  NaN count: {Z1Mat.isna().sum().sum()}")
print(f"  Range: [{Z1Mat.min().min():.3f}, {Z1Mat.max().max():.3f}]")

# %% [markdown]
# ## 3. Expression Feature Table
#
# Build expression features for the Z2 computation script.
# Root expression = mean expression across all 213 structures.

# %%
exp_features_path = os.path.join(ALLEN_DIR, "ExpMatchFeatures.csv")

root_exp = ExpMat.loc[Z1Mat.index].mean(axis=1, skipna=True)
exp_features = pd.DataFrame({"Genes": Z1Mat.index, "EXP": root_exp.values})
exp_features = exp_features.dropna(subset=["EXP"]).reset_index(drop=True)
exp_features = exp_features.sort_values("EXP", ascending=True).reset_index(drop=True)
exp_features["Rank"] = exp_features.index + 1
exp_features["quantile"] = exp_features["Rank"] / len(exp_features)
exp_features = exp_features.set_index("Genes")

exp_features.to_csv(exp_features_path)
print(f"Expression features: {exp_features.shape[0]} genes")
print(f"  Saved: {exp_features_path}")

# %% [markdown]
# ## 4. Compute Z2 Matrix (Expression-Matched Z-Score)
#
# For each gene *g* and structure *s*:
#
# $$Z2(g, s) = \frac{Z1(g, s) - \text{mean}(Z1(\text{matched}, s))}{\text{std}(Z1(\text{matched}, s))}$$
#
# Uses legacy expression match files (10,000 samples with replacement per gene,
# ±5% quantile window). These are the same match files used to produce the
# original published Z2.
#
# Computation is parallelized via `scripts/script_compute_Z2.py`.

# %%
z2_path = os.path.join(BIAS_DIR, "AllenMouseBrain_Z2bias.parquet")
MATCH_DIR = os.path.join(ProjDIR, config["data_files"]["legacy_match_dir"])

if os.path.exists(z2_path) and os.path.getmtime(z2_path) > os.path.getmtime(z1_path):
    print(f"Loading cached Z2 from {z2_path}")
    Z2Mat = pd.read_parquet(z2_path)
else:
    print(f"Computing Z2 via scripts/script_compute_Z2.py ...")
    print(f"  Match dir: {MATCH_DIR}")
    cmd = [
        sys.executable,
        os.path.join(ProjDIR, "scripts/script_compute_Z2.py"),
        "--z1", z1_path,
        "--exp-features", exp_features_path,
        "--output", z2_path,
        "--n-jobs", "10",
        "--also-csv",
        "--match-dir", MATCH_DIR,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Z2 computation failed with exit code {result.returncode}")

    Z2Mat = pd.read_parquet(z2_path)

print(f"Z2 matrix: {Z2Mat.shape}")
print(f"  NaN count: {Z2Mat.isna().sum().sum()}")
print(f"  Range: [{Z2Mat.min().min():.4f}, {Z2Mat.max().max():.4f}]")

# %% [markdown]
# ## 5. Cell Composition-Normalized Matrices
#
# Produce Z2 matrices normalized by neuronal density and neuron-to-glia ratio.
# These are used as confound controls in notebook 04 (Section 8).
#
# **Source**: Cell Atlas for the Mouse Brain (Erö et al. 2018, Frontiers in Neuroinformatics).
# Provides neuron/glia cell density (cells/mm³) per brain structure.
#
# **Normalization**:
# - **Neuron density**: `expression / neuron_density × 10⁵` — expression per neuron
# - **Neuro-to-glia ratio**: `expression / (neuron_density / glia_density)` — expression adjusted for cell composition
#
# Both are then Z1-scored per gene and Z2-matched using the same expression features.

# %%
import re

def modify_str(x):
    """Standardize cell atlas region names to match Allen ISH structure names."""
    x = re.sub("[()]", "", x)
    x = re.sub("-", "_", x)
    x = re.sub("reunions", "reuniens", x)
    x = "_".join(x.split(" "))
    return x

# Load raw (pre-log, pre-QN) expression for normalization
jon_raw_path = os.path.join(ProjDIR, config["data_files"]["jon_exp_raw"])
ExpRaw = pd.read_csv(jon_raw_path, index_col="ROW")
print(f"Raw expression: {ExpRaw.shape}, range [{ExpRaw.min().min():.2f}, {ExpRaw.max().max():.2f}]")

# Load cell composition densities
cell_comp_path = os.path.join(ProjDIR, "dat/cell_composition/Cell_Atlas_for_the_Mouse_brain_2.csv")
cell_comp = pd.read_csv(cell_comp_path, index_col="Regions")
cell_comp.index = [modify_str(x) for x in cell_comp.index]

# Filter to our 213 structures
cell_comp_213 = cell_comp.loc[ExpRaw.columns]
print(f"Cell composition: {cell_comp_213.shape[0]} structures matched")

# %%
# Neuron density normalization: expression / neuron_density * 1e5
neuron_density = cell_comp_213["Neurons [mm-3]"]
neuroden_norm = ExpRaw.div(neuron_density, axis=1) * 1e5

# Neuro-to-glia ratio normalization: expression / (neuron/glia ratio)
ng_ratio = cell_comp_213["Neurons [mm-3]"] / cell_comp_213["Glia [mm-3]"]
neuro2glia_norm = ExpRaw.div(ng_ratio, axis=1)

print(f"NeuroDen norm range: [{neuroden_norm.min().min():.2f}, {neuroden_norm.max().max():.2f}]")
print(f"Neuro2Glia norm range: [{neuro2glia_norm.min().min():.2f}, {neuro2glia_norm.max().max():.2f}]")

# %%
# Z1 conversion (per-gene z-score across structures)
def z1_convert_matrix(mat):
    z1_data, z1_genes = [], []
    for gene in mat.index:
        z1 = ZscoreConverting(mat.loc[gene].values)
        if not np.all(np.isnan(z1)):
            z1_data.append(z1)
            z1_genes.append(gene)
    return pd.DataFrame(data=np.array(z1_data), index=z1_genes, columns=mat.columns)

NeuroDen_Z1 = z1_convert_matrix(neuroden_norm)
Neuro2Glia_Z1 = z1_convert_matrix(neuro2glia_norm)

neuroden_z1_path = os.path.join(BIAS_DIR, "NeuroDensityNorm_Z1.parquet")
neuro2glia_z1_path = os.path.join(BIAS_DIR, "Neuro2GliaNorm_Z1.parquet")
NeuroDen_Z1.to_parquet(neuroden_z1_path)
Neuro2Glia_Z1.to_parquet(neuro2glia_z1_path)
print(f"NeuroDen Z1: {NeuroDen_Z1.shape}")
print(f"Neuro2Glia Z1: {Neuro2Glia_Z1.shape}")

# %%
# Z2 computation for both cell-composition-normalized matrices
# Uses the same expression features and match files as the standard Z2
for label, z1_in, z1_out_path in [
    ("NeuroDensityNorm", neuroden_z1_path, os.path.join(BIAS_DIR, "NeuroDensityNorm_Z2.parquet")),
    ("Neuro2GliaNorm", neuro2glia_z1_path, os.path.join(BIAS_DIR, "Neuro2GliaNorm_Z2.parquet")),
]:
    if os.path.exists(z1_out_path) and os.path.getmtime(z1_out_path) > os.path.getmtime(z1_in):
        print(f"{label} Z2: loading cached {z1_out_path}")
    else:
        print(f"{label} Z2: computing via script_compute_Z2.py ...")
        cmd = [
            sys.executable,
            os.path.join(ProjDIR, "scripts/script_compute_Z2.py"),
            "--z1", z1_in,
            "--exp-features", exp_features_path,
            "--output", z1_out_path,
            "--n-jobs", "10",
            "--match-dir", MATCH_DIR,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout[-200:] if result.stdout else "")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"{label} Z2 computation failed")

    z2_check = pd.read_parquet(z1_out_path)
    print(f"  {label} Z2: {z2_check.shape}, NaN={z2_check.isna().sum().sum()}")

# %% [markdown]
# ## Summary
#
# | Step | Output | Shape | Path |
# |------|--------|-------|------|
# | Z1 | Per-gene z-score | 17,208 × 213 | `dat/BiasMatrices/AllenMouseBrain_Z1.parquet` |
# | Z2 | Exp-matched z-score | 17,208 × 213 | `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` |
# | NeuroDen Z2 | Neuron density-normalized | 17,208 × 213 | `dat/BiasMatrices/NeuroDensityNorm_Z2.parquet` |
# | Neuro2Glia Z2 | Neuro-to-glia ratio-normalized | 17,208 × 213 | `dat/BiasMatrices/Neuro2GliaNorm_Z2.parquet` |
#
# **Input**: Jon's log2+QN expression (`dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv`)
#
# **Match files**: Legacy expression match files (`dat/allen-mouse-exp/ExpMatch_Legacy/`)
#
# **Note**: Python QN was tested but shifts the CCS local peak away from size 46.
# See `notebook_validation/Validate_ISH_Z2_Pipeline` for details.
