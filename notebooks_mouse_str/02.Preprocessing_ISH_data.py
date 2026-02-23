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
import json
import subprocess
import numpy as np
import pandas as pd

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    modify_str, LoadList, ZscoreConverting, quantileNormalize_withNA
)

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")
print(f"Working directory: {os.getcwd()}")

# %%
# Load config for data paths
with open(os.path.join(ProjDIR, "config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

ISH_DIR = config["data_files"]["ish_expression_dir"]
BIAS_DIR = os.path.join(ProjDIR, "dat/BiasMatrices")
ALLEN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp")
os.makedirs(BIAS_DIR, exist_ok=True)

print(f"ISH expression dir: {ISH_DIR}")
print(f"Bias matrices dir:  {BIAS_DIR}")
print(f"Allen metadata dir: {ALLEN_DIR}")

# %% [markdown]
# # 02. Preprocessing ISH Data
#
# This notebook processes raw Allen ISH expression data into three output matrices:
#
# 1. **Expression Level Matrix** (Gene × 213 STR) — log2(1+x) of mean expression energy
# 2. **Z1 Matrix** — per-gene z-score of expression across structures
# 3. **Z2 Matrix** — expression-matched z-score (controls for baseline expression level)
#
# All matrices are indexed by **human Entrez gene ID** and have 213 brain structure columns.
#
# The Z2 computation is parallelized via `scripts/script_compute_Z2.py`.

# %% [markdown]
# ## 1. Load Gene Mappings (from Notebook 01)

# %%
# Load human-mouse gene mapping and section ID assignments
with open(os.path.join(ALLEN_DIR, "human2mouse.0420.json"), "r") as f:
    Human2Mouse_Genes = json.load(f)
with open(os.path.join(ALLEN_DIR, "mouse2sectionID.0420.json"), "r") as f:
    Mouse2Human_Genes = {int(k): v for k, v in json.load(f).items()}

print(f"Human genes with mouse orthologs: {len(Human2Mouse_Genes)}")
print(f"Mouse genes with section IDs:     {len(Mouse2Human_Genes)}")

# %%
# Load 213 selected structures with atlas IDs
STR_Meta = pd.read_csv(os.path.join(ALLEN_DIR, "allen_brain_atlas_structures.csv"))
STR_Meta.dropna(inplace=True, subset=["atlas_id"])
STR_Meta["atlas_id"] = STR_Meta["atlas_id"].astype(int)
STR_Meta = STR_Meta.set_index("atlas_id")
STR_Meta["Name2"] = STR_Meta["safe_name"].apply(modify_str)

Selected_STRs = LoadList(os.path.join(ALLEN_DIR, "Structures.txt"))
STR_Meta_2 = STR_Meta[STR_Meta["Name2"].isin(Selected_STRs)].sort_values("Name2")

# Map structure name → structure_id for fast lookup
STR_names = STR_Meta_2["Name2"].values
STR_ids = STR_Meta_2["id"].values
print(f"Selected structures: {len(STR_Meta_2)} (expected 213)")

# %% [markdown]
# ## 2. Build Expression Level Matrix
#
# For each human gene:
# 1. Find mouse orthologs → Allen ISH section IDs
# 2. Read each ISH CSV, extract expression energy for each of 213 structures
# 3. Average across all experiments, apply log2(1+x) transform
#
# **This reads ~18,000 ISH CSV files. Cached after first run.**

# %%
exp_mat_path = os.path.join(BIAS_DIR, "AllenMouseBrain_ExpLevel.parquet")

if os.path.exists(exp_mat_path):
    print(f"Loading cached ExpMat from {exp_mat_path}")
    ExpMat = pd.read_parquet(exp_mat_path)
    print(f"ExpMat shape: {ExpMat.shape}")
else:
    # Collect per-gene section IDs
    gene_sections = {}  # human_entrez -> list of section_ids
    for entrez_str, v in Human2Mouse_Genes.items():
        entrez = int(entrez_str) if isinstance(entrez_str, str) else entrez_str
        section_ids = []
        for m_symbol, m_entrez in v["mouseHomo"]:
            if m_entrez in Mouse2Human_Genes:
                section_ids.extend(Mouse2Human_Genes[m_entrez]["allen_section_data_set_id"])
        if len(section_ids) > 0:
            gene_sections[entrez] = section_ids

    print(f"Human genes with ISH sections: {len(gene_sections)}")
    print(f"Total ISH reads needed: {sum(len(s) for s in gene_sections.values())}")

    # Read all ISH CSVs and build expression matrix
    All_Genes = []
    All_ExpEnergy = []

    for i, (entrez, sections) in enumerate(gene_sections.items()):
        g_All_dat = []
        for section_id in sections:
            csv_path = os.path.join(ISH_DIR, f"{section_id}.csv")
            if not os.path.exists(csv_path):
                continue
            dat_df = pd.read_csv(csv_path)
            # Extract expression energy for each structure
            dat = []
            for str_id in STR_ids:
                match = dat_df[dat_df["structure_id"] == str_id]
                if len(match) > 0:
                    dat.append(np.log2(1 + match["expression_energy"].values[0]))
                else:
                    dat.append(np.nan)
            g_All_dat.append(dat)

        if len(g_All_dat) == 0:
            continue

        g_All_dat = np.array(g_All_dat)
        g_avg = np.nanmean(g_All_dat, axis=0)
        All_Genes.append(entrez)
        All_ExpEnergy.append(g_avg)

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(gene_sections)} genes...")

    ExpMat = pd.DataFrame(
        data=np.array(All_ExpEnergy),
        index=All_Genes,
        columns=STR_names
    )
    ExpMat.index.name = None

    # Save
    ExpMat.to_parquet(exp_mat_path)
    ExpMat.to_csv(
        os.path.join(ALLEN_DIR, "ExpLevel.csv.gz"),
        compression="gzip"
    )
    print(f"Saved ExpMat: {ExpMat.shape}")

# %%
# Summary
n_valid = ExpMat.dropna(how="all").shape[0]
n_nan = ExpMat.isna().sum().sum()
print(f"ExpMat shape:      {ExpMat.shape}")
print(f"Genes with data:   {n_valid}")
print(f"Total NaN entries: {n_nan}")
print(f"Value range:       [{ExpMat.min().min():.3f}, {ExpMat.max().max():.3f}]")
ExpMat.iloc[:3, :5]

# %% [markdown]
# ## 3. Expression Matching
#
# Z2 controls for the relationship between baseline expression level and
# z-score magnitude. To do this, each gene is matched to ~1,000 genes with
# similar overall expression levels.
#
# **Algorithm** (uniform kernel matching):
# 1. Compute **root expression** = mean log2(1+x) expression across all 213 structures
# 2. Rank all genes by root expression, compute quantile (0→1)
# 3. For each gene, find all genes within a **±5% quantile window**
# 4. **Sample 10,000 genes with replacement** from that window (uniform kernel)
# 5. Deduplicate → keep up to 1,000 unique matched genes
# 6. Z2[g,s] = (Z1[g,s] − mean(Z1[matched,s])) / std(Z1[matched,s])
#
# The expression feature table and matching are saved for reproducibility.
# Match files are saved to `dat/allen-mouse-exp/ExpMatch/` (one file per gene).

# %%
# 3a. Compute expression feature table (root expression + quantile per gene)
exp_features_path = os.path.join(ALLEN_DIR, "ExpMatchFeatures.csv")

root_exp = ExpMat.mean(axis=1, skipna=True)
exp_features = pd.DataFrame({"Genes": ExpMat.index, "EXP": root_exp.values})
exp_features = exp_features.dropna(subset=["EXP"]).reset_index(drop=True)
exp_features = exp_features.sort_values("EXP", ascending=True).reset_index(drop=True)
exp_features["Rank"] = exp_features.index + 1
exp_features["quantile"] = exp_features["Rank"] / len(exp_features)
exp_features = exp_features.set_index("Genes")

exp_features.to_csv(exp_features_path)
print(f"Expression features: {exp_features.shape[0]} genes")
print(f"Expression range: [{exp_features['EXP'].min():.3f}, {exp_features['EXP'].max():.3f}]")
print(f"Saved to: {exp_features_path}")

# %%
# 3b. Generate expression match files (one per gene)
# Each file contains matched gene IDs (within ±5% quantile, sampled with replacement)
MATCH_DIR = os.path.join(ALLEN_DIR, "ExpMatch")
MATCH_SEED = 42
MATCH_SAMPLE_SIZE = 10000
MATCH_INTERVAL = 0.05

existing_matches = len(os.listdir(MATCH_DIR)) if os.path.exists(MATCH_DIR) else 0

if existing_matches >= len(exp_features) - 10:
    print(f"Expression match files already exist ({existing_matches} files in {MATCH_DIR})")
else:
    os.makedirs(MATCH_DIR, exist_ok=True)
    rng = np.random.default_rng(MATCH_SEED)
    quantiles = exp_features["quantile"].values
    gene_ids = exp_features.index.values

    for i, gene in enumerate(gene_ids):
        q = quantiles[i]
        q_min = max(0, q - MATCH_INTERVAL)
        q_max = min(1, q + MATCH_INTERVAL)

        mask = (quantiles >= q_min) & (quantiles <= q_max)
        mask[i] = False
        interval_genes = gene_ids[mask]

        if len(interval_genes) == 0:
            continue

        matched = rng.choice(interval_genes, size=MATCH_SAMPLE_SIZE, replace=True)
        with open(os.path.join(MATCH_DIR, f"{gene}.csv"), "w") as f:
            f.write("\n".join(str(g) for g in matched))

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{len(gene_ids)} match files...")

    n_files = len(os.listdir(MATCH_DIR))
    print(f"Generated {n_files} expression match files in {MATCH_DIR}")
    print(f"  seed={MATCH_SEED}, sample_size={MATCH_SAMPLE_SIZE}, interval=±{MATCH_INTERVAL}")

# %% [markdown]
# ## 4. Compute Z1 Matrix (Per-Gene Z-Score)
#
# For each gene, z-score its expression values across the 213 structures.
# This normalizes each gene to have mean=0, std=1 across structures.

# %%
z1_path = os.path.join(BIAS_DIR, "AllenMouseBrain_Z1.parquet")

z1_data = []
for gene_idx in ExpMat.index:
    row = ExpMat.loc[gene_idx].values
    z1_data.append(ZscoreConverting(row))

Z1Mat = pd.DataFrame(
    data=np.array(z1_data),
    index=ExpMat.index,
    columns=ExpMat.columns
)
Z1Mat.index.name = None

Z1Mat.to_parquet(z1_path)
Z1Mat.to_csv(
    os.path.join(ALLEN_DIR, "ExpLevel.log2.Zscore.csv.gz"),
    compression="gzip"
)
print(f"Z1 matrix saved: {Z1Mat.shape}")
print(f"Z1 NaN count: {Z1Mat.isna().sum().sum()}")
print(f"Z1 range: [{Z1Mat.min().min():.3f}, {Z1Mat.max().max():.3f}]")

# %% [markdown]
# ## 5. Compute Z2 Matrix (Expression-Matched Z-Score)
#
# For each gene *g* and structure *s*:
#
# $$Z2(g, s) = \frac{Z1(g, s) - \text{mean}(Z1(\text{matched}, s))}{\text{std}(Z1(\text{matched}, s))}$$
#
# **Expression matching**: For each gene, sample 10,000 genes (with replacement)
# from a ±5% quantile window around its root expression level (uniform kernel).
#
# Computation is parallelized via `scripts/script_compute_Z2.py`.

# %%
z2_path = os.path.join(BIAS_DIR, "AllenMouseBrain_Z2bias.parquet")
z2_csv_path = os.path.join(ALLEN_DIR, "AllenMouseBrain_Z2bias.csv.gz")

if os.path.exists(z2_path) and os.path.getmtime(z2_path) > os.path.getmtime(z1_path):
    print(f"Loading cached Z2 from {z2_path}")
    Z2Mat = pd.read_parquet(z2_path)
    print(f"Z2 shape: {Z2Mat.shape}")
else:
    print("Computing Z2 via scripts/script_compute_Z2.py ...")
    cmd = [
        sys.executable,
        os.path.join(ProjDIR, "scripts/script_compute_Z2.py"),
        "--z1", z1_path,
        "--exp-features", exp_features_path,
        "--output", z2_path,
        "--seed", "42",
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
    print(f"Z2 matrix loaded: {Z2Mat.shape}")

# %%
print(f"Z2 summary:")
print(f"  Shape:     {Z2Mat.shape}")
print(f"  NaN count: {Z2Mat.isna().sum().sum()}")
print(f"  Range:     [{Z2Mat.min().min():.4f}, {Z2Mat.max().max():.4f}]")

# %% [markdown]
# ## 6. Comparison with Reference Z2
#
# Compare the newly computed Z2 matrix against the old reference to verify
# consistency. Due to different random seeds in expression matching, values
# will not be identical but should be highly correlated.

# %%
from scipy.stats import pearsonr, spearmanr

# Try backup parquet first (saved before overwrite), then csv.gz
ref_parquet = os.path.join(BIAS_DIR, "AllenMouseBrain_Z2bias.parquet.bak")
ref_csv = os.path.join(ALLEN_DIR, "AllenMouseBrain_Z2bias.csv.gz")

if os.path.exists(ref_parquet):
    Z2_ref = pd.read_parquet(ref_parquet)
    print(f"Reference Z2 (parquet backup) shape: {Z2_ref.shape}")
elif os.path.exists(ref_csv):
    Z2_ref = pd.read_csv(ref_csv, index_col=0)
    print(f"Reference Z2 (csv.gz) shape: {Z2_ref.shape}")
else:
    Z2_ref = None
    print("No reference Z2 found for comparison")
if Z2_ref is not None:
    print(f"New Z2 shape: {Z2Mat.shape}")

# %%
if Z2_ref is not None:
    # Align on common genes and structures
    common_g = Z2Mat.index.intersection(Z2_ref.index)
    common_s = Z2Mat.columns.intersection(Z2_ref.columns)
    print(f"Common genes:      {len(common_g)}")
    print(f"Common structures: {len(common_s)}")

    z2_new = Z2Mat.loc[common_g, common_s].values.flatten()
    z2_old = Z2_ref.loc[common_g, common_s].values.flatten()

    # Remove NaN pairs
    valid = ~(np.isnan(z2_new) | np.isnan(z2_old))
    z2_new_v = z2_new[valid]
    z2_old_v = z2_old[valid]

    r_pearson, p_pearson = pearsonr(z2_new_v, z2_old_v)
    r_spearman, p_spearman = spearmanr(z2_new_v, z2_old_v)
    mae = np.mean(np.abs(z2_new_v - z2_old_v))
    max_diff = np.max(np.abs(z2_new_v - z2_old_v))

    print(f"\nZ2 Comparison (new vs old reference):")
    print(f"  Valid value pairs: {len(z2_new_v)}")
    print(f"  Pearson r:         {r_pearson:.6f}")
    print(f"  Spearman r:        {r_spearman:.6f}")
    print(f"  Mean abs error:    {mae:.4f}")
    print(f"  Max abs error:     {max_diff:.4f}")

# %%
if Z2_ref is not None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_alpha(0)

    # Scatter plot
    ax = axes[0]
    idx = np.random.default_rng(0).choice(len(z2_new_v), size=min(50000, len(z2_new_v)), replace=False)
    ax.scatter(z2_old_v[idx], z2_new_v[idx], s=0.5, alpha=0.3)
    ax.plot([-8, 20], [-8, 20], "r--", lw=0.5)
    ax.set_xlabel("Reference Z2")
    ax.set_ylabel("New Z2")
    ax.set_title(f"Z2 Comparison (r={r_pearson:.4f})")
    ax.patch.set_alpha(0)

    # Per-structure correlation
    ax = axes[1]
    str_corrs = []
    for s in common_s:
        old_s = Z2_ref.loc[common_g, s].values
        new_s = Z2Mat.loc[common_g, s].values
        valid_s = ~(np.isnan(old_s) | np.isnan(new_s))
        if valid_s.sum() > 10:
            r, _ = pearsonr(old_s[valid_s], new_s[valid_s])
            str_corrs.append(r)
    ax.hist(str_corrs, bins=30, edgecolor="black")
    ax.axvline(np.mean(str_corrs), color="red", linestyle="--",
               label=f"mean={np.mean(str_corrs):.4f}")
    ax.set_xlabel("Pearson r (per structure)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Structure Correlation")
    ax.legend()
    ax.patch.set_alpha(0)

    # Difference distribution
    ax = axes[2]
    diffs = z2_new_v - z2_old_v
    ax.hist(diffs, bins=100, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Z2_new - Z2_old")
    ax.set_ylabel("Count")
    ax.set_title(f"Difference Distribution (MAE={mae:.4f})")
    ax.patch.set_alpha(0)

    plt.tight_layout()
    os.makedirs(os.path.join(ProjDIR, "results/figures"), exist_ok=True)
    plt.savefig(
        os.path.join(ProjDIR, "results/figures/Z2_comparison.png"),
        dpi=150, transparent=True, bbox_inches="tight"
    )
    plt.show()

# %% [markdown]
# ## 7. Check Downstream Impact
#
# Compute ASD bias using both the new and old Z2 matrices to verify that
# downstream results are consistent.

# %%
if Z2_ref is not None:
    from ASD_Circuits import MouseSTR_AvgZ_Weighted, Fil2Dict

    # Load ASD gene weights
    gw_path = os.path.join(ProjDIR, "dat/Genetics/GeneWeights/Spark_Meta_EWS.GeneWeight.csv")
    Gene2Weights = Fil2Dict(gw_path)

    # Compute bias with new Z2
    bias_new = MouseSTR_AvgZ_Weighted(Z2Mat, Gene2Weights)
    bias_old = MouseSTR_AvgZ_Weighted(Z2_ref, Gene2Weights)

    # Compare
    common_str = bias_new.index.intersection(bias_old.index)
    r_bias, _ = pearsonr(
        bias_new.loc[common_str, "EFFECT"].values,
        bias_old.loc[common_str, "EFFECT"].values
    )
    bias_diff = (bias_new.loc[common_str, "EFFECT"] - bias_old.loc[common_str, "EFFECT"]).abs()

    print(f"ASD Bias Comparison (new vs old Z2):")
    print(f"  Structures:       {len(common_str)}")
    print(f"  Pearson r:        {r_bias:.6f}")
    print(f"  Mean abs diff:    {bias_diff.mean():.6f}")
    print(f"  Max abs diff:     {bias_diff.max():.6f}")

    # Show top structures
    top_new = bias_new.sort_values("EFFECT", ascending=False).head(10)
    top_old = bias_old.sort_values("EFFECT", ascending=False).head(10)
    print(f"\nTop 10 by new Z2:  {top_new.index.tolist()}")
    print(f"Top 10 by old Z2:  {top_old.index.tolist()}")
    print(f"Overlap:           {len(set(top_new.index) & set(top_old.index))}/10")

# %% [markdown]
# ## Summary
#
# | Output | Shape | Path |
# |--------|-------|------|
# | Expression Level | Gene × 213 STR | `dat/BiasMatrices/AllenMouseBrain_ExpLevel.parquet` |
# | Z1 (z-score) | Gene × 213 STR | `dat/BiasMatrices/AllenMouseBrain_Z1.parquet` |
# | Z2 (exp-matched) | Gene × 213 STR | `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` |
