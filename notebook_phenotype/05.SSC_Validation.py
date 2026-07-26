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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)

ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted, Mut2GeneDF, Filt_LGD_Mis,
)

os.chdir(os.path.join(ProjDIR, "notebook_phenotype"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 05. SSC Validation
#
# Replicate SPARK phenotype-brain bias correlations in the SSC cohort
# (gold-standard clinician-administered instruments: ADI-R, ADOS CSS).
#
# **Sections:**
# 1. Cross-cohort replication of shared phenotype-brain correlations
# 2. SSC-unique instruments (ADI-R subscales, SRS teacher report)
# 3. ADOS Calibrated Severity Score analysis
#
# **Outputs:** `results/phenotype/ssc_validation/`

# %% [markdown]
# ## Setup

# %%
# Load master phenotype table and bias matrix
master = pd.read_parquet(os.path.join(ProjDIR, "results/phenotype/mutation_phenotype_master.parquet"))
bias_mat = pd.read_parquet(os.path.join(ProjDIR, "results/phenotype/cache/subject_structure_bias_matrix.parquet"))

print(f"Master table: {master.shape[0]} subjects ({master['cohort'].value_counts().to_dict()})")
print(f"Bias matrix: {bias_mat.shape[0]} subjects x {bias_mat.shape[1]} structures")

# %%
# Split into SPARK and SSC subsets (restricted to subjects with bias data)
bias_iids = set(bias_mat.index)

spark_master = master[(master["cohort"] == "SPARK") & (master["IID"].isin(bias_iids))].copy()
ssc_master = master[(master["cohort"] == "SSC") & (master["IID"].isin(bias_iids))].copy()

spark_bias = bias_mat.loc[bias_mat.index.isin(spark_master["IID"])]
ssc_bias = bias_mat.loc[bias_mat.index.isin(ssc_master["IID"])]

print(f"SPARK with bias data: {len(spark_master)} subjects")
print(f"SSC with bias data: {len(ssc_master)} subjects")

# %%
# Load region annotations for coloring
str2reg = STR2Region()
structures = bias_mat.columns.tolist()

# Output directory
OUT_DIR = os.path.join(ProjDIR, "results/phenotype/ssc_validation")
FIG_DIR = os.path.join(ProjDIR, "results/phenotype/figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## Section 1: Replication of Core Phenotype-Brain Correlations
#
# For phenotypes measured in both SPARK and SSC, compute per-structure
# Spearman rho (phenotype vs. brain bias across subjects) separately in each
# cohort, then assess how well the 213-dimensional rho vectors agree.

# %%
# Shared phenotypes to compare
SHARED_PHENOTYPES = {
    "rbsr_total": "RBS-R Total",
    "dcdq_total": "DCDQ Total",
    "vine_abc": "Vineland ABC",
    "srs_total_t": "SRS Total T",
    "iq_fsiq": "Full-Scale IQ",
}


def compute_structure_rho(pheno_df, bias_df, pheno_col):
    """Compute Spearman rho between a phenotype and bias at each structure.

    Parameters
    ----------
    pheno_df : DataFrame
        Must contain 'IID' and *pheno_col* columns.
    bias_df : DataFrame
        Subjects (index) x structures (columns).
    pheno_col : str
        Column name in *pheno_df*.

    Returns
    -------
    pd.Series
        Spearman rho indexed by structure name.  NaN where n < 5.
    """
    # Align subjects with valid phenotype data
    valid = pheno_df[["IID", pheno_col]].dropna(subset=[pheno_col])
    common_iids = sorted(set(valid["IID"]) & set(bias_df.index))
    if len(common_iids) < 5:
        return pd.Series(np.nan, index=bias_df.columns, name=pheno_col)

    pheno_vals = valid.set_index("IID").loc[common_iids, pheno_col].values
    bias_vals = bias_df.loc[common_iids].values  # (n_subjects, n_structures)

    rhos = np.full(bias_vals.shape[1], np.nan)
    for j in range(bias_vals.shape[1]):
        col = bias_vals[:, j]
        mask = ~np.isnan(col) & ~np.isnan(pheno_vals)
        if mask.sum() >= 5:
            rhos[j], _ = spearmanr(pheno_vals[mask], col[mask])

    return pd.Series(rhos, index=bias_df.columns, name=pheno_col)


# %%
# Compute per-structure rho vectors for each cohort
spark_rhos = {}
ssc_rhos = {}

for pheno_key, pheno_label in SHARED_PHENOTYPES.items():
    spark_rhos[pheno_key] = compute_structure_rho(spark_master, spark_bias, pheno_key)
    ssc_rhos[pheno_key] = compute_structure_rho(ssc_master, ssc_bias, pheno_key)

    n_spark = spark_master[pheno_key].notna().sum()
    n_ssc = ssc_master[pheno_key].notna().sum()
    print(f"{pheno_label:20s}  SPARK n={n_spark:3d}  SSC n={n_ssc:3d}")

# %%
# Cross-cohort correlation: SPARK rho vs SSC rho
cross_cohort_results = []

for pheno_key, pheno_label in SHARED_PHENOTYPES.items():
    sr = spark_rhos[pheno_key]
    ss = ssc_rhos[pheno_key]
    mask = sr.notna() & ss.notna()
    if mask.sum() >= 5:
        r, p = spearmanr(sr[mask], ss[mask])
    else:
        r, p = np.nan, np.nan
    cross_cohort_results.append({
        "Phenotype": pheno_label,
        "pheno_key": pheno_key,
        "r_cross": r,
        "p_cross": p,
        "n_structures": int(mask.sum()),
    })

cross_df = pd.DataFrame(cross_cohort_results)
print("\nCross-cohort replication (SPARK rho vs SSC rho):")
print(cross_df[["Phenotype", "r_cross", "p_cross", "n_structures"]].to_string(index=False))

# %%
# Save cross-cohort results
cross_df.to_csv(os.path.join(OUT_DIR, "cross_cohort_replication.csv"), index=False)

# %%
# Scatter plots: SPARK rho vs SSC rho for each shared phenotype
region_palette = {
    "Isocortex": "#1f77b4",
    "OLF": "#ff7f0e",
    "HPF": "#2ca02c",
    "CTXsp": "#d62728",
    "STR": "#9467bd",
    "PAL": "#8c564b",
    "TH": "#e377c2",
    "HY": "#7f7f7f",
    "MB": "#bcbd22",
    "P": "#17becf",
    "MY": "#aec7e8",
    "CB": "#ffbb78",
}

n_pheno = len(SHARED_PHENOTYPES)
fig, axes = plt.subplots(1, n_pheno, figsize=(5 * n_pheno, 4.5))
if n_pheno == 1:
    axes = [axes]

for ax, (pheno_key, pheno_label) in zip(axes, SHARED_PHENOTYPES.items()):
    sr = spark_rhos[pheno_key]
    ss = ssc_rhos[pheno_key]
    mask = sr.notna() & ss.notna()
    structs = sr.index[mask]

    colors = [region_palette.get(str2reg.get(s, ""), "#cccccc") for s in structs]
    ax.scatter(sr[mask], ss[mask], c=colors, s=12, alpha=0.7, edgecolors="none")

    # Cross-cohort r annotation
    row = cross_df[cross_df["pheno_key"] == pheno_key].iloc[0]
    ax.set_title(f"{pheno_label}\nr = {row['r_cross']:.3f}, p = {row['p_cross']:.2e}", fontsize=10)
    ax.set_xlabel("SPARK rho", fontsize=9)
    ax.set_ylabel("SSC rho", fontsize=9)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")

    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)

# Region legend
handles = [plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=6, label=r)
           for r, c in region_palette.items()]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.patch.set_alpha(0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ssc_replication_scatter.pdf"),
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR}/ssc_replication_scatter.pdf")

# %% [markdown]
# ## Section 2: SSC-Unique Instruments
#
# ### 2a. ADI-R Subscales
# The Autism Diagnostic Interview-Revised (ADI-R) is a clinician-administered
# parent interview.  SSC provides three domain totals:
# - **A**: Social interaction (reciprocal social interaction)
# - **B verbal**: Communication (verbal)
# - **C**: Restricted, repetitive behaviours (RRB)
#
# ### 2b. SRS Teacher Report
# Compare parent-rated SRS (already in master) with teacher-rated SRS.

# %%
# --- 2a. ADI-R subscales from ssc_core_descriptive ---
SSC_PROBAND_DIR = os.path.join(
    ProjDIR,
    "dat/Phenotype/SSC_Phenotype_Dataset/SSC_V15_Phenotype_DATA/Proband_Data",
)

ssc_core = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "ssc_core_descriptive.csv"))
ssc_core = ssc_core[ssc_core["individual"].isin(ssc_master["IID"])]

adir_cols = {
    "individual": "IID",
    "adi_r_soc_a_total": "adir_social",
    "adi_r_b_comm_verbal_total": "adir_comm_verbal",
    "adi_r_rrb_c_total": "adir_rrb",
}
adir_df = ssc_core[list(adir_cols.keys())].rename(columns=adir_cols).copy()

# Coerce to numeric
for col in ["adir_social", "adir_comm_verbal", "adir_rrb"]:
    adir_df[col] = pd.to_numeric(adir_df[col], errors="coerce")

print("ADI-R subscales (SSC subjects with bias data):")
for col in ["adir_social", "adir_comm_verbal", "adir_rrb"]:
    n = adir_df[col].notna().sum()
    print(f"  {col}: n={n}, mean={adir_df[col].mean():.1f}, sd={adir_df[col].std():.1f}")

# %%
# Compute structure-level correlations for ADI-R subscales
ADIR_PHENOTYPES = {
    "adir_social": "ADI-R Social (A)",
    "adir_comm_verbal": "ADI-R Comm Verbal (B)",
    "adir_rrb": "ADI-R RRB (C)",
}

adir_rho_results = {}
for pheno_key, pheno_label in ADIR_PHENOTYPES.items():
    rho_vec = compute_structure_rho(adir_df, ssc_bias, pheno_key)
    adir_rho_results[pheno_key] = rho_vec
    n_valid = adir_df[pheno_key].notna().sum()
    median_rho = np.nanmedian(rho_vec)
    print(f"{pheno_label:30s}  n={n_valid:3d}  median rho={median_rho:.4f}")

# Save ADI-R rho vectors
adir_rho_df = pd.DataFrame(adir_rho_results)
adir_rho_df.index.name = "structure"
adir_rho_df.to_csv(os.path.join(OUT_DIR, "adir_structure_rho.csv"))
print(f"\nSaved: {OUT_DIR}/adir_structure_rho.csv")

# %%
# --- 2b. SRS Teacher Report ---
srs_teacher_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "srs_teacher.csv"))
srs_teacher_raw = srs_teacher_raw[srs_teacher_raw["individual"].isin(ssc_master["IID"])]

srs_teacher_cols = {
    "individual": "IID",
    "t_score": "srs_teacher_total_t",
    "awareness": "srs_teacher_awareness_raw",
    "cognition": "srs_teacher_cognition_raw",
    "communication": "srs_teacher_communication_raw",
    "mannerisms": "srs_teacher_mannerisms_raw",
    "motivation": "srs_teacher_motivation_raw",
}
srs_teacher = srs_teacher_raw[list(srs_teacher_cols.keys())].rename(columns=srs_teacher_cols)

# Coerce to numeric
for col in srs_teacher.columns:
    if col != "IID":
        srs_teacher[col] = pd.to_numeric(srs_teacher[col], errors="coerce")

# De-duplicate: keep first per subject
srs_teacher = srs_teacher.drop_duplicates(subset="IID", keep="first")

print(f"\nSRS Teacher: {len(srs_teacher)} SSC subjects with bias data")
print(f"  Total T-score: n={srs_teacher['srs_teacher_total_t'].notna().sum()}, "
      f"mean={srs_teacher['srs_teacher_total_t'].mean():.1f}")

# %%
# Compare parent- vs teacher-rated SRS correlations with brain bias
# Parent SRS total T is already in ssc_master as srs_total_t
parent_rho = compute_structure_rho(ssc_master, ssc_bias, "srs_total_t")
teacher_rho = compute_structure_rho(srs_teacher, ssc_bias, "srs_teacher_total_t")

mask = parent_rho.notna() & teacher_rho.notna()
if mask.sum() >= 5:
    r_pt, p_pt = spearmanr(parent_rho[mask], teacher_rho[mask])
else:
    r_pt, p_pt = np.nan, np.nan

n_parent = ssc_master["srs_total_t"].notna().sum()
n_teacher = srs_teacher["srs_teacher_total_t"].notna().sum()

print(f"\nParent SRS-brain rho vector (n={n_parent}): median = {np.nanmedian(parent_rho):.4f}")
print(f"Teacher SRS-brain rho vector (n={n_teacher}): median = {np.nanmedian(teacher_rho):.4f}")
print(f"Parent vs Teacher rho-vector correlation: r = {r_pt:.3f}, p = {p_pt:.2e}")

# %%
# Scatter: parent vs teacher SRS-brain rho
fig, ax = plt.subplots(figsize=(5, 5))

structs = parent_rho.index[mask]
colors = [region_palette.get(str2reg.get(s, ""), "#cccccc") for s in structs]
ax.scatter(parent_rho[mask], teacher_rho[mask], c=colors, s=12, alpha=0.7, edgecolors="none")

ax.set_xlabel("Parent SRS -- brain bias rho", fontsize=10)
ax.set_ylabel("Teacher SRS -- brain bias rho", fontsize=10)
ax.set_title(f"Parent vs Teacher SRS\nr = {r_pt:.3f}, p = {p_pt:.2e}", fontsize=11)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal", adjustable="box")

fig.patch.set_alpha(0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ssc_parent_vs_teacher_srs.pdf"),
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR}/ssc_parent_vs_teacher_srs.pdf")

# %%
# Save SRS comparison
srs_compare_df = pd.DataFrame({
    "structure": structures,
    "parent_rho": parent_rho.values,
    "teacher_rho": teacher_rho.values,
})
srs_compare_df.to_csv(os.path.join(OUT_DIR, "srs_parent_vs_teacher_rho.csv"), index=False)
print(f"Saved: {OUT_DIR}/srs_parent_vs_teacher_rho.csv")

# %% [markdown]
# ## Section 3: ADOS Calibrated Severity Score
#
# The ADOS CSS is the gold-standard clinician-administered autism severity
# measure.  It is available for SSC subjects in `ssc_core_descriptive.csv`.

# %%
# Load ADOS CSS
ados_cols = {
    "individual": "IID",
    "ados_css": "ados_css",
    "ados_social_affect": "ados_sa",
    "ados_restricted_repetitive": "ados_rrb",
}
ados_df = ssc_core[list(ados_cols.keys())].rename(columns=ados_cols).copy()

for col in ["ados_css", "ados_sa", "ados_rrb"]:
    ados_df[col] = pd.to_numeric(ados_df[col], errors="coerce")

print("ADOS scores (SSC subjects with bias data):")
for col in ["ados_css", "ados_sa", "ados_rrb"]:
    n = ados_df[col].notna().sum()
    print(f"  {col}: n={n}, mean={ados_df[col].mean():.1f}, sd={ados_df[col].std():.1f}")

# %%
# Compute structure-level correlations for ADOS measures
ADOS_PHENOTYPES = {
    "ados_css": "ADOS CSS",
    "ados_sa": "ADOS Social Affect",
    "ados_rrb": "ADOS RRB",
}

ados_rho_results = {}
for pheno_key, pheno_label in ADOS_PHENOTYPES.items():
    rho_vec = compute_structure_rho(ados_df, ssc_bias, pheno_key)
    ados_rho_results[pheno_key] = rho_vec
    n_valid = ados_df[pheno_key].notna().sum()
    median_rho = np.nanmedian(rho_vec)
    mean_rho = np.nanmean(rho_vec)
    print(f"{pheno_label:25s}  n={n_valid:3d}  median rho={median_rho:.4f}  mean rho={mean_rho:.4f}")

# %%
# Compare ADOS CSS rho with SPARK phenotype rhos (if replication phenotypes
# track the same brain regions as ADOS, it supports construct validity)
ados_css_rho = ados_rho_results["ados_css"]

print("\nCorrelation of ADOS CSS rho vector with SPARK rho vectors:")
for pheno_key, pheno_label in SHARED_PHENOTYPES.items():
    sr = spark_rhos[pheno_key]
    mask_a = ados_css_rho.notna() & sr.notna()
    if mask_a.sum() >= 5:
        r_a, p_a = spearmanr(ados_css_rho[mask_a], sr[mask_a])
    else:
        r_a, p_a = np.nan, np.nan
    print(f"  ADOS CSS vs SPARK {pheno_label:15s}: r = {r_a:.3f}, p = {p_a:.2e}")

# %%
# Scatter plot: ADOS rho vectors
n_ados = len(ADOS_PHENOTYPES)
fig, axes = plt.subplots(1, n_ados, figsize=(5 * n_ados, 4.5))
if n_ados == 1:
    axes = [axes]

for ax, (pheno_key, pheno_label) in zip(axes, ADOS_PHENOTYPES.items()):
    rho_vec = ados_rho_results[pheno_key]
    valid_mask = rho_vec.notna()
    structs = rho_vec.index[valid_mask]
    vals = rho_vec[valid_mask].values

    colors = [region_palette.get(str2reg.get(s, ""), "#cccccc") for s in structs]
    ranks = np.argsort(np.argsort(-vals))  # rank by rho (highest = 0)

    ax.scatter(ranks, vals, c=colors, s=12, alpha=0.7, edgecolors="none")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    n_valid = ados_df[pheno_key].notna().sum()
    ax.set_title(f"{pheno_label} (n={n_valid})\nmedian rho = {np.nanmedian(vals):.4f}", fontsize=10)
    ax.set_xlabel("Structure rank", fontsize=9)
    ax.set_ylabel("Spearman rho", fontsize=9)
    ax.tick_params(labelsize=8)

handles = [plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=6, label=r)
           for r, c in region_palette.items()]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.patch.set_alpha(0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ssc_ados_structure_rho.pdf"),
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR}/ssc_ados_structure_rho.pdf")

# %%
# Save ADOS results
ados_rho_df = pd.DataFrame(ados_rho_results)
ados_rho_df.index.name = "structure"
ados_rho_df.to_csv(os.path.join(OUT_DIR, "ados_structure_rho.csv"))
print(f"Saved: {OUT_DIR}/ados_structure_rho.csv")

# %%
# Save all per-structure rho vectors (SPARK + SSC shared phenotypes)
all_rhos = {}
for pheno_key in SHARED_PHENOTYPES:
    all_rhos[f"spark_{pheno_key}"] = spark_rhos[pheno_key]
    all_rhos[f"ssc_{pheno_key}"] = ssc_rhos[pheno_key]

all_rho_df = pd.DataFrame(all_rhos)
all_rho_df.index.name = "structure"
all_rho_df.to_csv(os.path.join(OUT_DIR, "shared_phenotype_structure_rho.csv"))
print(f"Saved: {OUT_DIR}/shared_phenotype_structure_rho.csv")

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("SSC VALIDATION SUMMARY")
print("=" * 60)

print("\n1. Cross-cohort replication (SPARK rho vs SSC rho):")
for _, row in cross_df.iterrows():
    print(f"   {row['Phenotype']:20s}  r = {row['r_cross']:+.3f}  (p = {row['p_cross']:.2e})")

print("\n2. ADI-R subscales (SSC only):")
for pheno_key, pheno_label in ADIR_PHENOTYPES.items():
    rho = adir_rho_results[pheno_key]
    print(f"   {pheno_label:30s}  median rho = {np.nanmedian(rho):+.4f}")

print(f"\n3. SRS Parent vs Teacher:")
print(f"   Rho-vector correlation: r = {r_pt:.3f} (p = {p_pt:.2e})")

print("\n4. ADOS CSS (gold-standard severity):")
for pheno_key, pheno_label in ADOS_PHENOTYPES.items():
    rho = ados_rho_results[pheno_key]
    print(f"   {pheno_label:25s}  median rho = {np.nanmedian(rho):+.4f}")

print("\nOutput files saved to:", OUT_DIR)
