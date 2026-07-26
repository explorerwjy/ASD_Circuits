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
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed

ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted
from plot import bh_fdr, REGION_COLORS

os.chdir(os.path.join(ProjDIR, "notebook_phenotype"))

SEED = 42
N_PERM = 10_000
N_JOBS = 10

# Load config
with open(os.path.join(ProjDIR, "config/config.yaml")) as f:
    config = yaml.safe_load(f)

print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 03. Phenotype--Brain Mapping
#
# Continuous phenotype--brain correlations: for each clinical phenotype
# dimension, compute Spearman correlations between the phenotype score and
# per-subject structure bias across 213 brain structures. Permutation-based
# p-values (10,000 permutations) with BH FDR correction.
#
# **Inputs:**
# - `results/phenotype/mutation_phenotype_master.parquet` (from NB01)
# - `results/phenotype/subject_gene_weights.pkl` (from NB01)
# - Expression Z2 matrix (from config)
#
# **Outputs:**
# - `results/phenotype/cache/subject_structure_bias_matrix.parquet`
# - `results/phenotype/cache/continuous_corr_{pheno}.parquet` per phenotype
# - `results/phenotype/figs/phenotype_structure_heatmap.pdf`
# - `results/phenotype/continuous/phenotype_structure_summary.csv`

# %% [markdown]
# ## 1. Load Data

# %%
# Master table
master = pd.read_parquet(
    os.path.join(ProjDIR, "results/phenotype/mutation_phenotype_master.parquet")
)
print(f"Master table: {master.shape[0]} subjects x {master.shape[1]} columns")

# Per-subject gene weights
with open(os.path.join(ProjDIR, "results/phenotype/subject_gene_weights.pkl"), "rb") as f:
    subject_gene_weights = pk.load(f)
print(f"Gene weights for {len(subject_gene_weights)} subjects")

# Expression Z2 matrix (genes x structures)
expr_path = os.path.join(ProjDIR, config["analysis_types"]["STR_ISH"]["expr_matrix"])
ExpZ2 = pd.read_parquet(expr_path)
print(f"Expression matrix: {ExpZ2.shape[0]} genes x {ExpZ2.shape[1]} structures")

# Structure-to-region mapping
str2reg = STR2Region()

# %% [markdown]
# ## 2. Per-Subject Structure Bias Matrix

# %%
cache_dir = os.path.join(ProjDIR, "results/phenotype/cache")
os.makedirs(cache_dir, exist_ok=True)

bias_cache = os.path.join(cache_dir, "subject_structure_bias_matrix.parquet")

if os.path.exists(bias_cache):
    print(f"Loading cached bias matrix from {bias_cache}")
    subject_bias_df = pd.read_parquet(bias_cache)
    print(f"Loaded: {subject_bias_df.shape[0]} subjects x {subject_bias_df.shape[1]} structures")
else:
    print("Computing per-subject structure bias vectors ...")
    structures = ExpZ2.columns.tolist()
    iids = list(master["IID"].values)

    # Pre-filter subjects with at least 1 gene weight
    valid_iids = [iid for iid in iids if len(subject_gene_weights.get(iid, {})) > 0]
    print(f"  Subjects with gene weights: {len(valid_iids)} / {len(iids)}")

    # Compute bias for each subject -- MouseSTR_AvgZ_Weighted returns a
    # DataFrame with EFFECT column indexed by structure (sorted by EFFECT).
    # We need a consistent structure order, so we reindex.
    rows = {}
    for i, iid in enumerate(valid_iids):
        gw = subject_gene_weights[iid]
        bias_df = MouseSTR_AvgZ_Weighted(ExpZ2, gw)
        rows[iid] = bias_df["EFFECT"].reindex(structures)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(valid_iids)} subjects done")

    subject_bias_df = pd.DataFrame(rows).T  # (N_subjects x 213 structures)
    subject_bias_df.index.name = "IID"
    print(f"Bias matrix: {subject_bias_df.shape}")

    subject_bias_df.to_parquet(bias_cache)
    print(f"Cached to {bias_cache}")

# %% [markdown]
# ## 3. Phenotype--Structure Correlations (Spearman + Permutation)

# %%
# Define phenotype columns and display labels
phenotype_cols = {
    # RBS-R (Repetitive Behavior Scale-Revised)
    "rbsr_total": "RBS-R Total",
    "rbsr_stereotyped": "RBS-R Stereotyped",
    "rbsr_selfinjury": "RBS-R Self-Injury",
    "rbsr_compulsive": "RBS-R Compulsive",
    "rbsr_ritualistic": "RBS-R Ritualistic",
    "rbsr_sameness": "RBS-R Sameness",
    "rbsr_restricted": "RBS-R Restricted",
    # DCDQ (Developmental Coordination Disorder Questionnaire)
    "dcdq_total": "DCDQ Total",
    "dcdq_control": "DCDQ Motor Control",
    "dcdq_fine": "DCDQ Fine Motor",
    "dcdq_general": "DCDQ Coordination",
    # Vineland Adaptive Behavior
    "vine_abc": "Vineland ABC",
    "vine_comm": "Vineland Communication",
    "vine_dls": "Vineland Daily Living",
    "vine_social": "Vineland Social",
    "vine_motor": "Vineland Motor",
    # SRS (Social Responsiveness Scale)
    "srs_total_t": "SRS Total T",
    "srs_rrb_t": "SRS RRB T",
    "srs_awr_t": "SRS Awareness T",
    "srs_soccog_t": "SRS Social Cognition T",
    "srs_com_t": "SRS Communication T",
    "srs_mot_t": "SRS Motivation T",
    # IQ
    "iq_fsiq": "FSIQ",
    "iq_viq": "Verbal IQ",
    "iq_nviq": "Nonverbal IQ",
    # Developmental milestones (age in months -- higher = later = worse)
    "milestone_words_mos": "Age First Words (mos)",
    "milestone_phrases_mos": "Age Phrases (mos)",
    "milestone_walk_mos": "Age Walking (mos)",
    "milestone_sat_mos": "Age Sitting (mos)",
}

print(f"Phenotype dimensions: {len(phenotype_cols)}")
for col, label in phenotype_cols.items():
    n_valid = master[col].notna().sum() if col in master.columns else 0
    print(f"  {label:35s}  n={n_valid}")


# %%
def compute_phenotype_structure_corr(pheno_col, master, subject_bias_df,
                                     n_perm=N_PERM, seed=SEED):
    """Spearman correlation between one phenotype and all structures,
    with permutation p-values and BH FDR correction.

    Optimization: rank-transform once, then use Pearson on ranks for the
    permutation loop (equivalent to Spearman but ~10x faster).

    Returns a DataFrame with columns:
        rho, pval_perm, qval_fdr  (indexed by structure)
    """
    # Align subjects: must have both phenotype and bias (no NaN in bias)
    valid_iids = master.loc[master[pheno_col].notna(), "IID"].values
    valid_iids = [iid for iid in valid_iids if iid in subject_bias_df.index]
    # Drop subjects with any NaN in their bias vector (e.g., genes not in expression matrix)
    bias_sub = subject_bias_df.loc[valid_iids]
    nan_rows = bias_sub.isna().any(axis=1)
    if nan_rows.any():
        valid_iids = [iid for iid in valid_iids if not nan_rows.loc[iid]]
    if len(valid_iids) < 10:
        print(f"  WARNING: {pheno_col} has only {len(valid_iids)} valid subjects, skipping")
        return None

    pheno_vals = master.set_index("IID").loc[valid_iids, pheno_col].values.astype(float)
    bias_mat = subject_bias_df.loc[valid_iids].values  # (N_subj x 213)
    structures = subject_bias_df.columns.tolist()
    n_subj, n_str = bias_mat.shape

    # Rank-transform phenotype and each structure column (for fast Spearman)
    pheno_ranks = rankdata(pheno_vals)
    bias_ranks = np.apply_along_axis(rankdata, 0, bias_mat)  # (N_subj x 213)

    # Demean ranks for fast Pearson-on-ranks
    pheno_dm = pheno_ranks - pheno_ranks.mean()
    bias_dm = bias_ranks - bias_ranks.mean(axis=0)  # broadcast over structures

    # Observed Spearman rho (= Pearson on ranks)
    denom = np.sqrt(np.sum(pheno_dm ** 2)) * np.sqrt(np.sum(bias_dm ** 2, axis=0))
    rho_obs = np.sum(pheno_dm[:, None] * bias_dm, axis=0) / denom  # (213,)

    # Permutation test: shuffle phenotype ranks, recompute correlation
    rng = np.random.default_rng(seed)
    # Count how many permutations yield |rho| >= |rho_obs| for each structure
    count_ge = np.zeros(n_str, dtype=int)

    pheno_dm_norm = np.sqrt(np.sum(pheno_dm ** 2))
    bias_dm_norms = np.sqrt(np.sum(bias_dm ** 2, axis=0))  # (213,)

    for _ in range(n_perm):
        perm_idx = rng.permutation(n_subj)
        pheno_perm_dm = pheno_dm[perm_idx]
        rho_perm = np.dot(pheno_perm_dm, bias_dm) / (pheno_dm_norm * bias_dm_norms)
        count_ge += (np.abs(rho_perm) >= np.abs(rho_obs)).astype(int)

    # Two-sided permutation p-value: add 1 to numerator and denominator
    # to avoid p=0 and for conservative estimation
    pval_perm = (count_ge + 1) / (n_perm + 1)

    # BH FDR correction across structures
    qval_fdr = bh_fdr(pval_perm)

    result = pd.DataFrame({
        "rho": rho_obs,
        "pval_perm": pval_perm,
        "qval_fdr": qval_fdr,
    }, index=structures)
    result.index.name = "Structure"

    return result


# %%
# Run correlations for all phenotypes (with caching)
all_results = {}

for pheno_col, label in phenotype_cols.items():
    if pheno_col not in master.columns:
        print(f"SKIP {label}: column not in master table")
        continue

    cache_path = os.path.join(cache_dir, f"continuous_corr_{pheno_col}.parquet")
    if os.path.exists(cache_path):
        print(f"  Loading cached: {label}")
        all_results[pheno_col] = pd.read_parquet(cache_path)
        continue

    n_valid = master[pheno_col].notna().sum()
    print(f"Computing: {label} (n={n_valid}) ...")
    result = compute_phenotype_structure_corr(pheno_col, master, subject_bias_df,
                                              n_perm=N_PERM, seed=SEED)
    if result is not None:
        result.to_parquet(cache_path)
        all_results[pheno_col] = result
        n_sig = (result["qval_fdr"] < 0.05).sum()
        print(f"  rho range: [{result['rho'].min():.3f}, {result['rho'].max():.3f}], "
              f"FDR<0.05: {n_sig}")

print(f"\nCompleted {len(all_results)} phenotypes")

# %% [markdown]
# ## 4. Phenotype x Structure Heatmap

# %%
# Build rho matrix (phenotypes x structures) and significance mask
# Order structures by region for visual grouping
structures = subject_bias_df.columns.tolist()
str_regions = pd.Series({s: str2reg.get(s, "Other") for s in structures})

REGIONS_seq = [
    "Isocortex", "Olfactory_areas", "Cortical_subplate",
    "Hippocampus", "Amygdala", "Striatum",
    "Thalamus", "Hypothalamus", "Midbrain",
    "Pallidum", "Pons", "Medulla", "Cerebellum",
]

# Sort structures by region order, then alphabetically within region
region_order_map = {r: i for i, r in enumerate(REGIONS_seq)}
str_sort_key = pd.DataFrame({
    "structure": structures,
    "region": [str2reg.get(s, "Other") for s in structures],
})
str_sort_key["region_rank"] = str_sort_key["region"].map(
    lambda r: region_order_map.get(r, 99)
)
str_sort_key = str_sort_key.sort_values(["region_rank", "structure"])
ordered_structures = str_sort_key["structure"].tolist()

# Phenotype display order (group by domain)
pheno_order = [col for col in phenotype_cols.keys() if col in all_results]
pheno_labels = [phenotype_cols[col] for col in pheno_order]

# Assemble rho matrix and significance matrix
rho_matrix = np.full((len(pheno_order), len(ordered_structures)), np.nan)
sig_matrix = np.full((len(pheno_order), len(ordered_structures)), False)

for i, pheno_col in enumerate(pheno_order):
    res = all_results[pheno_col]
    for j, structure in enumerate(ordered_structures):
        if structure in res.index:
            rho_matrix[i, j] = res.loc[structure, "rho"]
            sig_matrix[i, j] = res.loc[structure, "qval_fdr"] < 0.05

rho_df = pd.DataFrame(rho_matrix, index=pheno_labels, columns=ordered_structures)

print(f"Heatmap matrix: {rho_df.shape}")
print(f"Significant cells (FDR<0.05): {sig_matrix.sum()}")

# %%
fig_dir = os.path.join(ProjDIR, "results/phenotype/figs")
os.makedirs(fig_dir, exist_ok=True)

# Determine symmetric color limits
vmax = np.nanmax(np.abs(rho_matrix))
vmax = np.ceil(vmax * 20) / 20  # round up to nearest 0.05

fig, axes = plt.subplots(
    2, 1, figsize=(24, len(pheno_order) * 0.45 + 2),
    height_ratios=[1, len(pheno_order)],
    facecolor="none", sharex=True,
    gridspec_kw={"hspace": 0.02},
)

# --- Top axis: region color bar ---
ax_top = axes[0]
region_colors_for_bar = [
    REGION_COLORS.get(str2reg.get(s, "Other"), "#cccccc")
    for s in ordered_structures
]
for j, color in enumerate(region_colors_for_bar):
    ax_top.axvspan(j - 0.5, j + 0.5, color=color, alpha=0.8)
ax_top.set_xlim(-0.5, len(ordered_structures) - 0.5)
ax_top.set_yticks([])
ax_top.set_xticks([])
ax_top.patch.set_alpha(0)
for spine in ax_top.spines.values():
    spine.set_visible(False)

# Region legend
used_regions = []
for r in REGIONS_seq:
    if r in str_regions.values:
        used_regions.append(r)
legend_handles = [
    mpatches.Patch(color=REGION_COLORS.get(r, "#cccccc"),
                   label=r.replace("_", " "))
    for r in used_regions
]
ax_top.legend(
    handles=legend_handles, loc="upper center",
    bbox_to_anchor=(0.5, 2.5), ncol=len(used_regions),
    frameon=False, fontsize=7,
)

# --- Bottom axis: heatmap ---
ax_heat = axes[1]
im = ax_heat.imshow(
    rho_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    interpolation="nearest",
)

# Stars for FDR < 0.05
for i in range(sig_matrix.shape[0]):
    for j in range(sig_matrix.shape[1]):
        if sig_matrix[i, j]:
            ax_heat.text(j, i, "*", ha="center", va="center",
                         fontsize=6, color="black", fontweight="bold")

ax_heat.set_yticks(range(len(pheno_labels)))
ax_heat.set_yticklabels(pheno_labels, fontsize=8)
ax_heat.set_xticks([])
ax_heat.set_xlabel("Brain structures (ordered by region)", fontsize=10)
ax_heat.patch.set_alpha(0)

# Colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02, label="Spearman rho")

fig.suptitle(
    "Phenotype--Structure Correlations (per-subject ASD bias)",
    fontsize=13, y=0.98,
)

save_path = os.path.join(fig_dir, "phenotype_structure_heatmap.pdf")
fig.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()

# %% [markdown]
# ## 5. Top Structures per Phenotype

# %%
out_dir = os.path.join(ProjDIR, "results/phenotype/continuous")
os.makedirs(out_dir, exist_ok=True)

summary_rows = []
for pheno_col in pheno_order:
    label = phenotype_cols[pheno_col]
    res = all_results[pheno_col]
    n_valid = master[pheno_col].notna().sum()
    n_valid_with_bias = len(
        [iid for iid in master.loc[master[pheno_col].notna(), "IID"]
         if iid in subject_bias_df.index]
    )

    # Top 5 positive and negative
    res_sorted = res.sort_values("rho", ascending=False)
    n_fdr05 = (res["qval_fdr"] < 0.05).sum()
    n_fdr10 = (res["qval_fdr"] < 0.10).sum()

    print(f"\n{'='*60}")
    print(f"{label}  (n={n_valid_with_bias}, FDR<0.05: {n_fdr05}, FDR<0.10: {n_fdr10})")
    print(f"{'='*60}")

    if n_fdr05 > 0:
        sig_res = res[res["qval_fdr"] < 0.05].sort_values("rho", ascending=False)
        sig_res["region"] = sig_res.index.map(str2reg)
        print(f"\n  Significant structures (FDR<0.05):")
        for s, row in sig_res.iterrows():
            direction = "+" if row["rho"] > 0 else "-"
            print(f"    {direction} {s:40s}  rho={row['rho']:+.3f}  "
                  f"q={row['qval_fdr']:.4f}  [{row['region']}]")
    else:
        # Show top 3 regardless
        print(f"\n  Top 3 (no FDR<0.05 structures):")
        for s, row in res_sorted.head(3).iterrows():
            print(f"    {s:40s}  rho={row['rho']:+.3f}  q={row['qval_fdr']:.4f}")

    # Collect summary stats per phenotype
    summary_rows.append({
        "phenotype": pheno_col,
        "label": label,
        "n_subjects": n_valid_with_bias,
        "max_rho": res["rho"].max(),
        "min_rho": res["rho"].min(),
        "median_abs_rho": np.median(np.abs(res["rho"])),
        "n_fdr05": n_fdr05,
        "n_fdr10": n_fdr10,
        "top_structure_pos": res_sorted.index[0],
        "top_rho_pos": res_sorted.iloc[0]["rho"],
        "top_structure_neg": res_sorted.index[-1],
        "top_rho_neg": res_sorted.iloc[-1]["rho"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(out_dir, "phenotype_structure_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")
print(f"\n{summary_df.to_string(index=False)}")

# %% [markdown]
# ## 6. Confound Check: Partial Correlations

# %%
# Partial correlations controlling for n_mutations, sex, cohort.
# Strategy: rank-transform all variables, regress out confounds via OLS,
# then correlate residuals.

def partial_spearman(pheno_vals, bias_vals, confound_mat):
    """Spearman partial correlation via rank-residualization.

    1. Rank-transform phenotype, bias, and confounds.
    2. Regress confounds out of phenotype ranks and bias ranks via OLS.
    3. Pearson correlation on residuals = partial Spearman rho.

    Returns (rho_partial, pval) where pval is from the Pearson test on residuals.
    """
    from scipy.stats import pearsonr

    # Rank-transform
    pheno_r = rankdata(pheno_vals)
    bias_r = rankdata(bias_vals)
    conf_r = np.apply_along_axis(rankdata, 0, confound_mat)

    # Add intercept
    X = np.column_stack([np.ones(len(pheno_r)), conf_r])

    # Residualize phenotype
    beta_p = np.linalg.lstsq(X, pheno_r, rcond=None)[0]
    resid_pheno = pheno_r - X @ beta_p

    # Residualize bias
    beta_b = np.linalg.lstsq(X, bias_r, rcond=None)[0]
    resid_bias = bias_r - X @ beta_b

    rho, pval = pearsonr(resid_pheno, resid_bias)
    return rho, pval


# %%
# Select phenotypes with enough subjects for confound analysis
# Need: sex, cohort, n_mutations all non-null
# Encode sex and cohort as numeric
master_conf = master.copy()
master_conf["sex_num"] = (master_conf["sex"] == "Male").astype(float)
master_conf["cohort_num"] = (master_conf["cohort"] == "SPARK").astype(float)

# Focus on phenotypes with >= 50 subjects after confound filtering
top_phenotypes = [col for col in pheno_order
                  if master_conf[[col, "n_mutations", "sex_num", "cohort_num"]].dropna().shape[0] >= 50]

print(f"Phenotypes with >= 50 subjects for partial correlation: {len(top_phenotypes)}")

partial_results = {}
for pheno_col in top_phenotypes:
    label = phenotype_cols[pheno_col]
    # Get complete cases
    cols_needed = [pheno_col, "n_mutations", "sex_num", "cohort_num", "IID"]
    sub = master_conf[cols_needed].dropna()
    valid_iids = [iid for iid in sub["IID"].values if iid in subject_bias_df.index]
    sub = sub[sub["IID"].isin(valid_iids)].set_index("IID")

    if len(sub) < 50:
        continue

    pheno_vals = sub[pheno_col].values
    confound_mat = sub[["n_mutations", "sex_num", "cohort_num"]].values
    bias_mat_sub = subject_bias_df.loc[sub.index].values

    raw_rhos = all_results[pheno_col].reindex(subject_bias_df.columns)["rho"].values
    partial_rhos = np.zeros(bias_mat_sub.shape[1])
    partial_pvals = np.zeros(bias_mat_sub.shape[1])

    for j in range(bias_mat_sub.shape[1]):
        bv = bias_mat_sub[:, j]
        mask = np.isfinite(bv)
        if mask.sum() < 20:
            partial_rhos[j] = np.nan
            partial_pvals[j] = np.nan
            continue
        rho_p, pval_p = partial_spearman(pheno_vals[mask], bv[mask], confound_mat[mask])
        partial_rhos[j] = rho_p
        partial_pvals[j] = pval_p

    valid_mask = np.isfinite(partial_pvals)
    partial_qvals = np.full_like(partial_pvals, np.nan)
    if valid_mask.any():
        partial_qvals[valid_mask] = bh_fdr(partial_pvals[valid_mask])

    partial_df = pd.DataFrame({
        "rho_raw": raw_rhos,
        "rho_partial": partial_rhos,
        "pval_partial": partial_pvals,
        "qval_partial": partial_qvals,
    }, index=subject_bias_df.columns)
    partial_df.index.name = "Structure"

    partial_results[pheno_col] = partial_df
    both_valid = np.isfinite(raw_rhos) & np.isfinite(partial_rhos)
    r_corr = np.corrcoef(raw_rhos[both_valid], partial_rhos[both_valid])[0, 1] if both_valid.sum() > 2 else np.nan
    n_sig_raw = (all_results[pheno_col]["qval_fdr"] < 0.05).sum()
    n_sig_part = np.nansum(partial_qvals < 0.05)
    print(f"  {label:35s}  n={len(sub):>3d}  "
          f"raw_vs_partial r={r_corr:.3f}  "
          f"FDR<0.05: {n_sig_raw}->{n_sig_part}")

# %%
# Scatter: raw rho vs partial rho for selected phenotypes
n_plots = min(len(partial_results), 6)
plot_phenos = list(partial_results.keys())[:n_plots]

if n_plots > 0:
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             facecolor="none")
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, pheno_col in enumerate(plot_phenos):
        ax = axes[idx]
        ax.patch.set_alpha(0)
        pdf = partial_results[pheno_col].dropna(subset=["rho_raw", "rho_partial"])
        ax.scatter(pdf["rho_raw"], pdf["rho_partial"], s=8, alpha=0.5,
                   color="#268ad5", edgecolors="none")
        # Identity line
        if len(pdf) > 0:
            lim = max(abs(pdf["rho_raw"].max()), abs(pdf["rho_partial"].max()),
                      abs(pdf["rho_raw"].min()), abs(pdf["rho_partial"].min()))
        else:
            lim = 0.1
        lim = max(lim * 1.1, 0.01)  # avoid zero-range
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Raw Spearman rho", fontsize=9)
        ax.set_ylabel("Partial Spearman rho", fontsize=9)
        if len(pdf) > 2:
            r_corr = np.corrcoef(pdf["rho_raw"], pdf["rho_partial"])[0, 1]
        else:
            r_corr = np.nan
        ax.set_title(f"{phenotype_cols[pheno_col]}\nr={r_corr:.3f}", fontsize=10)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Raw vs Partial Spearman rho (controlling for n_mutations, sex, cohort)",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    save_path = os.path.join(fig_dir, "raw_vs_partial_rho_scatter.pdf")
    fig.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
else:
    print("No phenotypes with >= 50 subjects for partial correlation scatter.")

# %%
print("\n--- Notebook 03 complete ---")
print(f"Results in: {os.path.join(ProjDIR, 'results/phenotype/')}")
