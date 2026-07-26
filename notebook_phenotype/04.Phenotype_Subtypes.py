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
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests

SEED = 42
np.random.seed(SEED)

ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted

os.chdir(os.path.join(ProjDIR, "notebook_phenotype"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 04. Phenotype Subtypes
#
# PCA on a phenotype summary matrix to identify latent dimensions of clinical
# heterogeneity, then map each principal component to brain-structure bias.
#
# **Inputs:**
# - `results/phenotype/mutation_phenotype_master.parquet` (NB01)
# - `results/phenotype/subject_gene_weights.pkl` (NB01)
# - `results/phenotype/cache/subject_structure_bias_matrix.parquet` (NB03)
# - `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet`
#
# **Outputs:**
# - `results/phenotype/figs/pca_loadings.pdf`
# - `results/phenotype/figs/pc_brain_heatmap.pdf`
# - `results/phenotype/subtypes/` (PCA results, PC-brain correlations)

# %% [markdown]
# ## Setup: Load Data

# %%
# Load master phenotype table
master = pd.read_parquet(
    os.path.join(ProjDIR, "results/phenotype/mutation_phenotype_master.parquet")
)
print(f"Master table: {master.shape[0]} subjects x {master.shape[1]} columns")
print(f"  SPARK: {(master['cohort'] == 'SPARK').sum()}")
print(f"  SSC: {(master['cohort'] == 'SSC').sum()}")

# %%
# Load per-subject gene weights
with open(os.path.join(ProjDIR, "results/phenotype/subject_gene_weights.pkl"), "rb") as f:
    subject_gene_weights = pk.load(f)
print(f"Gene weights for {len(subject_gene_weights)} subjects")

# %%
# Load subject-structure bias matrix (subjects x 213 structures)
bias_mat = pd.read_parquet(
    os.path.join(ProjDIR, "results/phenotype/cache/subject_structure_bias_matrix.parquet")
)
print(f"Subject-structure bias matrix: {bias_mat.shape}")

# %%
# Load expression Z2 matrix and region annotations
ExpZ2Mat = pd.read_parquet(
    os.path.join(ProjDIR, "dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet")
)
str2reg = STR2Region()
print(f"Expression matrix: {ExpZ2Mat.shape[0]} genes x {ExpZ2Mat.shape[1]} structures")

# %% [markdown]
# ## Section 1: Build Phenotype Matrix

# %%
# Select summary scores across 6 clinical domains
pheno_cols = [
    "rbsr_total",          # Repetitive behavior (RBS-R overall)
    "dcdq_total",          # Motor coordination (DCDQ total)
    "vine_abc",            # Adaptive behavior (Vineland ABC)
    "srs_total_t",         # Social responsiveness (SRS total T-score)
    "iq_fsiq",             # Cognitive ability (full-scale IQ)
    "milestone_words_mos", # Language development (age first words, months)
]

pheno_labels = {
    "rbsr_total": "RBS-R Total",
    "dcdq_total": "DCDQ Total",
    "vine_abc": "Vineland ABC",
    "srs_total_t": "SRS Total T",
    "iq_fsiq": "FSIQ",
    "milestone_words_mos": "Words (mos)",
}

# %%
# Require at least 3 non-null phenotype measures per subject
pheno_sub = master[["IID", "cohort"] + pheno_cols].copy()
n_valid = pheno_sub[pheno_cols].notna().sum(axis=1)
pheno_sub = pheno_sub[n_valid >= 3].reset_index(drop=True)

print(f"Subjects with >=3 phenotypes: {len(pheno_sub)} / {len(master)}")
print(f"  SPARK: {(pheno_sub['cohort'] == 'SPARK').sum()}")
print(f"  SSC: {(pheno_sub['cohort'] == 'SSC').sum()}")
print()

# Per-phenotype coverage
for col in pheno_cols:
    n = pheno_sub[col].notna().sum()
    print(f"  {pheno_labels[col]:20s}: {n:4d} / {len(pheno_sub)} ({100*n/len(pheno_sub):.1f}%)")

# %% [markdown]
# ## Section 2: PCA on Phenotype Matrix

# %%
# Impute missing values with median, then standardize
X_raw = pheno_sub[pheno_cols].values

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"Phenotype matrix after imputation/scaling: {X_scaled.shape}")

# %%
# Run PCA with all components
pca = PCA(n_components=len(pheno_cols), random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    cum = pca.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1}: {ev:.4f}  (cumulative: {cum:.4f})")

# %%
# Heatmap of top 4 PC loadings
n_pcs_show = min(4, len(pheno_cols))
loadings = pd.DataFrame(
    pca.components_[:n_pcs_show].T,
    index=[pheno_labels[c] for c in pheno_cols],
    columns=[f"PC{i+1}" for i in range(n_pcs_show)],
)

fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
ax.patch.set_alpha(0)
sns.heatmap(
    loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("PCA Loadings (Top 4 PCs)", fontsize=12)
ax.set_ylabel("")
plt.tight_layout()

out_fig = os.path.join(ProjDIR, "results/phenotype/figs/pca_loadings.pdf")
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
fig.savefig(out_fig, transparent=True, dpi=300, bbox_inches="tight")
print(f"Saved: {out_fig}")
plt.show()

# %% [markdown]
# ## Section 3: Map PCs to Brain-Structure Bias

# %%
# Align subjects between phenotype PCA and bias matrix
common_iids = sorted(set(pheno_sub["IID"]) & set(bias_mat.index))
print(f"Subjects in both phenotype PCA and bias matrix: {len(common_iids)}")

# Build aligned matrices
pca_scores = pd.DataFrame(
    X_pca, index=pheno_sub["IID"].values,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
)
pca_aligned = pca_scores.loc[common_iids]
bias_aligned = bias_mat.loc[common_iids]

# %%
# Spearman correlation: each PC vs each structure's bias across subjects
n_pcs_map = min(4, X_pca.shape[1])
structures = bias_aligned.columns.tolist()

corr_matrix = np.zeros((n_pcs_map, len(structures)))
pval_matrix = np.zeros((n_pcs_map, len(structures)))

for i in range(n_pcs_map):
    pc_vals = pca_aligned[f"PC{i+1}"].values
    for j, st in enumerate(structures):
        bias_vals = bias_aligned[st].values
        # Drop pairs where bias is NaN
        mask = ~np.isnan(bias_vals)
        if mask.sum() < 10:
            corr_matrix[i, j] = np.nan
            pval_matrix[i, j] = 1.0
            continue
        rho, pv = spearmanr(pc_vals[mask], bias_vals[mask])
        corr_matrix[i, j] = rho
        pval_matrix[i, j] = pv

corr_df = pd.DataFrame(
    corr_matrix, index=[f"PC{i+1}" for i in range(n_pcs_map)], columns=structures
)
pval_df = pd.DataFrame(
    pval_matrix, index=[f"PC{i+1}" for i in range(n_pcs_map)], columns=structures
)

# %%
# FDR correction (per PC, across structures)
fdr_df = pval_df.copy()
for pc in fdr_df.index:
    pvals = pval_df.loc[pc].values
    valid = ~np.isnan(pvals)
    if valid.sum() > 0:
        _, qvals, _, _ = multipletests(pvals[valid], method="fdr_bh")
        fdr_vals = np.full(len(pvals), np.nan)
        fdr_vals[valid] = qvals
        fdr_df.loc[pc] = fdr_vals

n_sig = (fdr_df < 0.05).sum(axis=1)
print("Structures with q < 0.05 per PC:")
for pc in fdr_df.index:
    print(f"  {pc}: {n_sig[pc]} / {len(structures)}")

# %%
# Order structures by brain region for the heatmap
REGIONS_seq = [
    "Isocortex", "Olfactory_areas", "Cortical_subplate", "Hippocampus",
    "Amygdala", "Striatum", "Pallidum", "Thalamus", "Hypothalamus",
    "Midbrain", "Pons", "Medulla", "Cerebellum",
]

REGION_COLORS = {
    "Isocortex": "#268ad5", "Olfactory_areas": "#5ab4ac",
    "Cortical_subplate": "#7ac3fa", "Hippocampus": "#2c9d39",
    "Amygdala": "#742eb5", "Striatum": "#ed8921",
    "Thalamus": "#e82315", "Hypothalamus": "#c27ba0",
    "Midbrain": "#f6b26b", "Pallidum": "#2ECC71",
    "Cerebellum": "#8B4513", "Medulla": "#708090",
    "Pons": "#A0522D",
}

# Sort structures by region order
struct_regions = pd.Series({s: str2reg.get(s, "Other") for s in structures})
ordered_structs = []
for reg in REGIONS_seq:
    reg_structs = [s for s in structures if struct_regions[s] == reg]
    ordered_structs.extend(sorted(reg_structs))
# Append any structures not in the defined regions
remaining = [s for s in structures if s not in ordered_structs]
ordered_structs.extend(sorted(remaining))

# %%
# PC-brain heatmap
corr_ordered = corr_df[ordered_structs]
fdr_ordered = fdr_df[ordered_structs]

# Region color bar
region_bar_colors = [
    REGION_COLORS.get(struct_regions.get(s, "Other"), "#cccccc")
    for s in ordered_structs
]

fig, axes = plt.subplots(
    2, 1, figsize=(18, 4), facecolor="none",
    gridspec_kw={"height_ratios": [0.15, 1], "hspace": 0.05},
    sharex=True,
)

# Top: region color bar
ax_bar = axes[0]
ax_bar.patch.set_alpha(0)
for i, color in enumerate(region_bar_colors):
    ax_bar.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.8)
ax_bar.set_xlim(-0.5, len(ordered_structs) - 0.5)
ax_bar.set_yticks([])
ax_bar.set_title("PC-Brain Structure Correlation (Spearman)", fontsize=13)

# Bottom: heatmap
ax_hm = axes[1]
ax_hm.patch.set_alpha(0)
vmax = max(0.3, np.nanmax(np.abs(corr_ordered.values)))
im = ax_hm.imshow(
    corr_ordered.values, aspect="auto", cmap="RdBu_r",
    vmin=-vmax, vmax=vmax, interpolation="none",
)
ax_hm.set_yticks(range(n_pcs_map))
ax_hm.set_yticklabels(corr_ordered.index, fontsize=11)
ax_hm.set_xticks([])
ax_hm.set_xlabel(f"{len(ordered_structs)} brain structures (ordered by region)")

# Mark significant entries
for i in range(n_pcs_map):
    for j in range(len(ordered_structs)):
        if fdr_ordered.iloc[i, j] < 0.05:
            ax_hm.text(j, i, "*", ha="center", va="center", fontsize=6, color="k")

# Colorbar
cbar = fig.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02)
cbar.set_label("Spearman rho", fontsize=10)

# Region legend
import matplotlib.patches as mpatches
present_regions = [r for r in REGIONS_seq if r in struct_regions.values]
legend_handles = [
    mpatches.Patch(color=REGION_COLORS[r], label=r.replace("_", " "))
    for r in present_regions
]
ax_bar.legend(
    handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    fontsize=7, ncol=1, frameon=False,
)

plt.tight_layout()
out_fig2 = os.path.join(ProjDIR, "results/phenotype/figs/pc_brain_heatmap.pdf")
fig.savefig(out_fig2, transparent=True, dpi=300, bbox_inches="tight")
print(f"Saved: {out_fig2}")
plt.show()

# %% [markdown]
# ## Section 4: Robustness -- SPARK vs SSC Cohort Comparison

# %%
# Run PCA separately for SPARK and SSC, compare loadings to full PCA
cohorts = {"SPARK": "SPARK", "SSC": "SSC"}
cohort_loadings = {}

for label, cohort_val in cohorts.items():
    idx = pheno_sub["cohort"] == cohort_val
    n_cohort = idx.sum()
    if n_cohort < 20:
        print(f"  {label}: only {n_cohort} subjects, skipping PCA")
        continue

    X_c = pheno_sub.loc[idx, pheno_cols].values
    X_c_imp = SimpleImputer(strategy="median").fit_transform(X_c)
    X_c_sc = StandardScaler().fit_transform(X_c_imp)

    pca_c = PCA(n_components=len(pheno_cols), random_state=SEED)
    pca_c.fit(X_c_sc)

    cohort_loadings[label] = pca_c.components_

    print(f"{label} (n={n_cohort}):")
    for i, ev in enumerate(pca_c.explained_variance_ratio_[:4]):
        print(f"  PC{i+1}: {ev:.4f}")

# %%
# Compare cohort loadings to full-sample loadings (correlation per PC)
# Note: PC sign is arbitrary; use absolute correlation
print("\nLoading correlation with full-sample PCA (|r|):")
for label, comp in cohort_loadings.items():
    n_pcs_cmp = min(4, comp.shape[0])
    corrs = []
    for i in range(n_pcs_cmp):
        r = np.abs(np.corrcoef(pca.components_[i], comp[i])[0, 1])
        corrs.append(r)
    print(f"  {label}: " + ", ".join(f"PC{i+1}={c:.3f}" for i, c in enumerate(corrs)))

# %%
# Visualization: side-by-side loading comparison
if len(cohort_loadings) == 2:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="none")
    titles = ["Full Sample", "SPARK", "SSC"]
    all_comps = [pca.components_] + [cohort_loadings[k] for k in ["SPARK", "SSC"]]

    for ax, title, comp in zip(axes, titles, all_comps):
        ax.patch.set_alpha(0)
        n_show = min(4, comp.shape[0])
        ld = pd.DataFrame(
            comp[:n_show].T,
            index=[pheno_labels[c] for c in pheno_cols],
            columns=[f"PC{i+1}" for i in range(n_show)],
        )
        sns.heatmap(
            ld, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, cbar=False,
            vmin=-1, vmax=1,
        )
        ax.set_title(title, fontsize=11)
        if ax != axes[0]:
            ax.set_ylabel("")

    plt.suptitle("PCA Loadings: Full vs Cohort-Specific", fontsize=13, y=1.02)
    plt.tight_layout()

    out_fig3 = os.path.join(ProjDIR, "results/phenotype/figs/pca_loadings_cohort_comparison.pdf")
    fig.savefig(out_fig3, transparent=True, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_fig3}")
    plt.show()

# %% [markdown]
# ## Section 5: Save Results

# %%
out_dir = os.path.join(ProjDIR, "results/phenotype/subtypes")
os.makedirs(out_dir, exist_ok=True)

# PCA scores per subject
pca_scores_df = pheno_sub[["IID", "cohort"]].copy()
for i in range(X_pca.shape[1]):
    pca_scores_df[f"PC{i+1}"] = X_pca[:, i]
pca_scores_df.to_csv(os.path.join(out_dir, "pca_scores.csv"), index=False)
print(f"Saved: {out_dir}/pca_scores.csv ({len(pca_scores_df)} subjects)")

# PCA loadings
loadings_full = pd.DataFrame(
    pca.components_.T,
    index=pheno_cols,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
)
loadings_full.to_csv(os.path.join(out_dir, "pca_loadings.csv"))
print(f"Saved: {out_dir}/pca_loadings.csv")

# Explained variance
var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
})
var_df.to_csv(os.path.join(out_dir, "pca_variance.csv"), index=False)
print(f"Saved: {out_dir}/pca_variance.csv")

# PC-brain correlations
corr_df.to_csv(os.path.join(out_dir, "pc_brain_spearman_rho.csv"))
pval_df.to_csv(os.path.join(out_dir, "pc_brain_pvalues.csv"))
fdr_df.to_csv(os.path.join(out_dir, "pc_brain_fdr_qvalues.csv"))
print(f"Saved: {out_dir}/pc_brain_spearman_rho.csv")
print(f"Saved: {out_dir}/pc_brain_pvalues.csv")
print(f"Saved: {out_dir}/pc_brain_fdr_qvalues.csv")

print("\nDone.")
