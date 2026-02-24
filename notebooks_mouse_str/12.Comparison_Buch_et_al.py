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

# %% [markdown]
# # 12. Cross-Species Validation: GENCIC Mouse vs Human fMRI
# Comparing ASD risk gene enrichment with functional connectivity abnormalities from Buch et al. (Nature Neuroscience 2024).

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress, pearsonr, rankdata
from adjustText import adjust_text

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import *

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ## 1. Data Loading

# %%
# Mouse GENCIC data
gencic_data = pd.read_csv('../results/GENCIC_MouseSTRBias.csv')
print(f"GENCIC: {gencic_data.shape[0]} structures")

# Buch et al. fMRI connectivity matrices
BUCH_DIR = "../dat/Buch_et_al"
Fig2A = pd.read_csv(os.path.join(BUCH_DIR, 'Fig2a_ccu1.csv'))
Fig2A.columns = Fig2A.columns.tolist()
Fig2A.index = Fig2A.columns

Fig2B = pd.read_csv(os.path.join(BUCH_DIR, 'Fig2b_ccu2.csv'))
Fig2B.columns = Fig2B.columns.tolist()
Fig2B.index = Fig2B.columns

Fig2C = pd.read_csv(os.path.join(BUCH_DIR, 'Fig2c_ccu3.csv'))
Fig2C.columns = Fig2C.columns.tolist()
Fig2C.index = Fig2C.columns

Fig2D = pd.read_csv(os.path.join(BUCH_DIR, 'Fig2d_atypical_all.csv'))
Fig2D.columns = Fig2D.columns.tolist()
Fig2D.index = Fig2D.columns
print(f"fMRI: 4 connectivity matrices with {Fig2A.shape[0]} regions each")

# Cross-species mapping (3 confidence tiers)
RegionMapping = pd.read_excel(os.path.join(BUCH_DIR, 'claude_mapping_v6.xlsx')).drop(columns=["Bias"], errors='ignore')
RegionMappingT1 = RegionMapping[RegionMapping["Score"] > 0.4]
RegionMappingT2 = RegionMapping[RegionMapping["Score"] > 0.6]
print(f"Mappings: ALL={len(RegionMapping)} | T1={len(RegionMappingT1)} | T2={len(RegionMappingT2)}")

# %% [markdown]
# ## 2. Preprocessing

# %%
def aggregate_bias_by_tier(gencic_data, region_mapping):
    gencic_human = gencic_data.merge(region_mapping, left_on='Structure', right_on='Mouse_Structure', how='inner')
    bias_by_region = gencic_human.groupby('Representative_Human_Group').agg({
        'Bias': ['mean', 'median', 'max', 'std', 'count'],
        'Pvalue': lambda x: (x < 0.05).sum()
    }).round(4)
    bias_by_region.columns = ['Mean_Bias', 'Median_Bias', 'Max_Bias', 'Std_Bias', 'N_Structures', 'N_Significant']
    return bias_by_region.sort_values('Mean_Bias', ascending=False)

bias_by_human_region = aggregate_bias_by_tier(gencic_data, RegionMapping)
bias_by_human_region_T1 = aggregate_bias_by_tier(gencic_data, RegionMappingT1)
bias_by_human_region_T2 = aggregate_bias_by_tier(gencic_data, RegionMappingT2)
print(f"Aggregated bias: ALL={len(bias_by_human_region)} | T1={len(bias_by_human_region_T1)} | T2={len(bias_by_human_region_T2)} human regions")

# %% [markdown]
# ## 3. Analysis Functions

# %%
def calculate_fmri_metrics(fmri_matrix):
    metrics = pd.DataFrame(index=fmri_matrix.columns)
    metrics['Total_Strength'] = fmri_matrix.sum()
    metrics['Total_Strength_Abs'] = fmri_matrix.abs().sum()
    metrics['Mean_Strength'] = fmri_matrix.apply(lambda col: col[col != 0].mean() if (col != 0).sum() > 0 else 0)
    metrics['N_Connections'] = (fmri_matrix != 0).sum()
    return metrics

def analyze_fmri_gencic_correlation(fmri_matrix, bias_data, dataset_name, verbose=True):
    fmri_metrics = calculate_fmri_metrics(fmri_matrix)
    merged = bias_data.merge(fmri_metrics, left_index=True, right_index=True, how='inner')
    if verbose:
        print(f"\n{'='*80}\n{dataset_name}\n{'='*80}")
        print(f"Regions: {len(merged)}")
        for metric in ['Total_Strength_Abs', 'Mean_Strength']:
            valid = merged.dropna(subset=[metric, 'Mean_Bias'])
            if len(valid) > 3:
                rho, pval = spearmanr(valid['Mean_Bias'], valid[metric])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {metric:20s}: rho = {rho:7.4f}, p = {pval:.4f} {sig}")
    return merged, {}

def mergeCP(df):
    df = df.copy()
    for merge_list, new_name in [(['Caudate', 'Putamen'], 'Caudate-Putamen'), (['SMA', 'Premotor'], 'SMA-Premotor')]:
        structs = [s for s in merge_list if s in df.index]
        if len(structs) == 2:
            df.loc[new_name] = df.loc[structs].mean()
            df = df.drop(structs)
    return df

def plot_combined(Ys_dfs, title_text="", figsize=(11, 8), p_perm_pearson=None, p_perm_spearman=None):
    Xs = Ys_dfs[0]['Mean_Bias']
    Ys = np.zeros_like(Xs)
    for df in Ys_dfs:
        Ys += df['Total_Strength_Abs'].values
    Xs, Ys = np.asarray(Xs), np.asarray(Ys)
    rho, pval = spearmanr(Xs, Ys)
    linres = linregress(Xs, Ys)
    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.scatter(Xs, Ys, color='#377eb8', edgecolor='k', s=80, alpha=0.9, zorder=2)
    X_fit = np.linspace(Xs.min(), Xs.max(), 100)
    ax.plot(X_fit, linres.slope * X_fit + linres.intercept, color="#e6550d", linestyle="--", linewidth=2.2, zorder=1)
    texts = []
    for i, label in enumerate(Ys_dfs[0].index):
        texts.append(ax.text(Xs[i], Ys[i], label, fontsize=15, fontweight='bold', alpha=0.91,
                             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="gray", lw=1.0, alpha=0.65), zorder=3))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.8), expand_points=(1.3, 2.0))
    ax.set_xlabel('ASD Mutation Bias', fontsize=17, fontweight='bold')
    ax.set_ylabel('fMRI Connectivity-Symptom Association\nStrength (SA + RRB + VIQ)', fontsize=17, fontweight='bold')
    stat_text = f"$r$ = {linres.rvalue:.2f}\n$p$ = {linres.pvalue:.2f}"
    if p_perm_pearson is not None:
        stat_text += f"\n$p_{{perm}}$ = {p_perm_pearson:.4f}"
    ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, fontsize=25, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", lw=1.1, alpha=0.70))
    ax.set_xticks(ax.get_xticks())
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3, linestyle=":", zorder=0)
    plt.tight_layout()
    plt.show()
    print(f"Spearman rho={rho:.3f}, p={pval:.4f} | Linear r={linres.rvalue:.3f}, p={linres.pvalue:.4f}")
    if p_perm_pearson is not None:
        print(f"Sibling permutation: p_perm(Pearson)={p_perm_pearson:.4f}, p_perm(Spearman)={p_perm_spearman:.4f}")

# %% [markdown]
# ## 4. All Regions Analysis

# %%
merged_viq, _ = analyze_fmri_gencic_correlation(Fig2A, bias_by_human_region, "Verbal IQ")
merged_sa, _ = analyze_fmri_gencic_correlation(Fig2B, bias_by_human_region, "Social Affect")
merged_rrb, _ = analyze_fmri_gencic_correlation(Fig2C, bias_by_human_region, "RRB")

# %%
plot_combined([merged_sa], "ALL REGIONS: GENCIC Bias vs SA fMRI")
plot_combined([merged_rrb], "ALL REGIONS: GENCIC Bias vs RRB fMRI")
plot_combined([merged_viq], "ALL REGIONS: GENCIC Bias vs VIQ fMRI")

# %%
plot_combined([merged_sa, merged_rrb, merged_viq], "ALL REGIONS: GENCIC Bias vs Combined fMRI")

# %%
merged_viq_cp, merged_sa_cp, merged_rrb_cp = mergeCP(merged_viq), mergeCP(merged_sa), mergeCP(merged_rrb)
plot_combined([merged_sa_cp, merged_rrb_cp, merged_viq_cp], "ALL REGIONS (CP Merged): GENCIC vs fMRI")

# %% [markdown]
# ## 5. Tier 1 Analysis (Score > 0.4)

# %%
merged_viq_T1, _ = analyze_fmri_gencic_correlation(Fig2A, bias_by_human_region_T1, "Verbal IQ (T1)")
merged_sa_T1, _ = analyze_fmri_gencic_correlation(Fig2B, bias_by_human_region_T1, "Social Affect (T1)")
merged_rrb_T1, _ = analyze_fmri_gencic_correlation(Fig2C, bias_by_human_region_T1, "RRB (T1)")

# %%
plot_combined([merged_sa_T1, merged_rrb_T1, merged_viq_T1], "TIER 1 (Score > 0.4): GENCIC vs fMRI")

# %%
merged_viq_cp_T1, merged_sa_cp_T1, merged_rrb_cp_T1 = mergeCP(merged_viq_T1), mergeCP(merged_sa_T1), mergeCP(merged_rrb_T1)
plot_combined([merged_sa_cp_T1, merged_rrb_cp_T1, merged_viq_cp_T1], "TIER 1 - CP Merged: GENCIC vs fMRI")

# %% [markdown]
# ## 6. Tier 2 Analysis (Score > 0.6)

# %%
merged_viq_T2, _ = analyze_fmri_gencic_correlation(Fig2A, bias_by_human_region_T2, "Verbal IQ (T2)")
merged_sa_T2, _ = analyze_fmri_gencic_correlation(Fig2B, bias_by_human_region_T2, "Social Affect (T2)")
merged_rrb_T2, _ = analyze_fmri_gencic_correlation(Fig2C, bias_by_human_region_T2, "RRB (T2)")

# %% [markdown]
# ### Sibling-Based Permutation P-Values
# Standard regression p-values assume independence between brain structures.
# To address potential spatial autocorrelation, we compute empirical p-values
# from the sibling mutability null distribution (10,000 simulated bias profiles).

# %%
# Load sibling null bias (213 structures × 10,000 sims)
sib_null = pd.read_parquet('../results/Sibling_bias/Mutability_61gene/sibling_mutability_bias.parquet')

# Build T2 aggregation: group sibling structures by human region
gencic_struct_set = set(gencic_data['Structure'].values)
t2_structs_in_null = RegionMappingT2[RegionMappingT2['Mouse_Structure'].isin(gencic_struct_set & set(sib_null.index))]
region_groups_T2 = t2_structs_in_null.groupby('Representative_Human_Group')['Mouse_Structure'].apply(list).to_dict()

# Aggregate null bias to human regions (mean per region, matching aggregate_bias_by_tier)
null_regions = sorted(region_groups_T2.keys())
null_agg = np.zeros((len(null_regions), sib_null.shape[1]))
for i, region in enumerate(null_regions):
    null_agg[i] = sib_null.loc[region_groups_T2[region]].mean(axis=0).values
null_agg_df = pd.DataFrame(null_agg, index=null_regions, columns=sib_null.columns)

# Align with fMRI data (same regions as in merged_sa_T2)
common_regions = null_agg_df.index.intersection(merged_sa_T2.index)
X_null = null_agg_df.loc[common_regions].values  # (N_regions, 10000)
Y_fmri = (merged_sa_T2.loc[common_regions, 'Total_Strength_Abs'].values
         + merged_rrb_T2.loc[common_regions, 'Total_Strength_Abs'].values
         + merged_viq_T2.loc[common_regions, 'Total_Strength_Abs'].values)

# Observed correlations
X_obs = merged_sa_T2.loc[common_regions, 'Mean_Bias'].values
obs_pearson_r, obs_pearson_p = pearsonr(X_obs, Y_fmri)
obs_spearman_rho, obs_spearman_p = spearmanr(X_obs, Y_fmri)

# Vectorized Pearson null correlations
n_sims = X_null.shape[1]
Y_c = Y_fmri - Y_fmri.mean()
Y_ss = np.sum(Y_c**2)
X_means = X_null.mean(axis=0)
X_c = X_null - X_means[np.newaxis, :]
X_ss = np.sum(X_c**2, axis=0)
null_pearson_r = (Y_c @ X_c) / np.sqrt(X_ss * Y_ss)

# Spearman null correlations (rank-based)
Y_ranks = rankdata(Y_fmri)
null_spearman_rho = np.empty(n_sims)
for i in range(n_sims):
    null_spearman_rho[i] = np.corrcoef(rankdata(X_null[:, i]), Y_ranks)[0, 1]

# One-tailed permutation p-values (testing positive correlation)
p_perm_pearson_T2 = (np.sum(null_pearson_r >= obs_pearson_r) + 1) / (n_sims + 1)
p_perm_spearman_T2 = (np.sum(null_spearman_rho >= obs_spearman_rho) + 1) / (n_sims + 1)
# Two-tailed alternative:
# p_perm_pearson_T2 = (np.sum(np.abs(null_pearson_r) >= np.abs(obs_pearson_r)) + 1) / (n_sims + 1)
# p_perm_spearman_T2 = (np.sum(np.abs(null_spearman_rho) >= np.abs(obs_spearman_rho)) + 1) / (n_sims + 1)

print(f"T2 Sibling Permutation P-values (N={len(common_regions)} regions, {n_sims} sims):")
print(f"  Pearson:  r = {obs_pearson_r:.4f}, p = {obs_pearson_p:.4f}, p_perm = {p_perm_pearson_T2:.4f}")
print(f"  Spearman: rho = {obs_spearman_rho:.4f}, p = {obs_spearman_p:.4f}, p_perm = {p_perm_spearman_T2:.4f}")

# %%
plot_combined([merged_sa_T2, merged_rrb_T2, merged_viq_T2], "TIER 2 (Score > 0.6): GENCIC vs fMRI",
              p_perm_pearson=p_perm_pearson_T2, p_perm_spearman=p_perm_spearman_T2)

# %%
merged_viq_cp_T2, merged_sa_cp_T2, merged_rrb_cp_T2 = mergeCP(merged_viq_T2), mergeCP(merged_sa_T2), mergeCP(merged_rrb_T2)

# CP-merged sibling permutation p-values
null_agg_cp_df = mergeCP(null_agg_df)
common_regions_cp = null_agg_cp_df.index.intersection(merged_sa_cp_T2.index)
X_null_cp = null_agg_cp_df.loc[common_regions_cp].values
Y_fmri_cp = (merged_sa_cp_T2.loc[common_regions_cp, 'Total_Strength_Abs'].values
            + merged_rrb_cp_T2.loc[common_regions_cp, 'Total_Strength_Abs'].values
            + merged_viq_cp_T2.loc[common_regions_cp, 'Total_Strength_Abs'].values)

X_obs_cp = merged_sa_cp_T2.loc[common_regions_cp, 'Mean_Bias'].values
obs_pearson_r_cp, obs_pearson_p_cp = pearsonr(X_obs_cp, Y_fmri_cp)
obs_spearman_rho_cp, obs_spearman_p_cp = spearmanr(X_obs_cp, Y_fmri_cp)

# Vectorized Pearson null
Y_c_cp = Y_fmri_cp - Y_fmri_cp.mean()
Y_ss_cp = np.sum(Y_c_cp**2)
X_c_cp = X_null_cp - X_null_cp.mean(axis=0)[np.newaxis, :]
X_ss_cp = np.sum(X_c_cp**2, axis=0)
null_pearson_r_cp = (Y_c_cp @ X_c_cp) / np.sqrt(X_ss_cp * Y_ss_cp)

# Spearman null
Y_ranks_cp = rankdata(Y_fmri_cp)
null_spearman_rho_cp = np.empty(n_sims)
for i in range(n_sims):
    null_spearman_rho_cp[i] = np.corrcoef(rankdata(X_null_cp[:, i]), Y_ranks_cp)[0, 1]

# One-tailed permutation p-values (testing positive correlation)
p_perm_pearson_cp = (np.sum(null_pearson_r_cp >= obs_pearson_r_cp) + 1) / (n_sims + 1)
p_perm_spearman_cp = (np.sum(null_spearman_rho_cp >= obs_spearman_rho_cp) + 1) / (n_sims + 1)
# Two-tailed alternative:
# p_perm_pearson_cp = (np.sum(np.abs(null_pearson_r_cp) >= np.abs(obs_pearson_r_cp)) + 1) / (n_sims + 1)
# p_perm_spearman_cp = (np.sum(np.abs(null_spearman_rho_cp) >= np.abs(obs_spearman_rho_cp)) + 1) / (n_sims + 1)

print(f"T2 CP-Merged Sibling Permutation P-values (N={len(common_regions_cp)} regions, {n_sims} sims):")
print(f"  Pearson:  r = {obs_pearson_r_cp:.4f}, p = {obs_pearson_p_cp:.4f}, p_perm = {p_perm_pearson_cp:.4f}")
print(f"  Spearman: rho = {obs_spearman_rho_cp:.4f}, p = {obs_spearman_p_cp:.4f}, p_perm = {p_perm_spearman_cp:.4f}")

# %%
plot_combined([merged_sa_cp_T2, merged_rrb_cp_T2, merged_viq_cp_T2], "",
              p_perm_pearson=p_perm_pearson_cp, p_perm_spearman=p_perm_spearman_cp)

# %% [markdown]
# ## 7. Summary

# %%
summary = []
for tier_name, dfs in [
    ('ALL', (merged_sa, merged_rrb, merged_viq)),
    ('T1', (merged_sa_T1, merged_rrb_T1, merged_viq_T1)),
    ('T2', (merged_sa_T2, merged_rrb_T2, merged_viq_T2))
]:
    Xs = dfs[0]['Mean_Bias']
    Ys = sum(df['Total_Strength_Abs'].values for df in dfs)
    rho, pval = spearmanr(Xs, Ys)
    linres = linregress(Xs, Ys)
    summary.append({
        'Tier': tier_name, 'N': len(Xs),
        'Spearman_rho': rho, 'Spearman_p': pval,
        'Linear_r': linres.rvalue, 'Linear_p': linres.pvalue,
    })

summary_df = pd.DataFrame(summary)
print("Combined SA + RRB + VIQ Analysis")
print(summary_df.round(4).to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='none')
fig.patch.set_alpha(0)
for idx, (tier_name, dfs) in enumerate([
    ('ALL', (merged_sa, merged_rrb, merged_viq)),
    ('T1 (>0.4)', (merged_sa_T1, merged_rrb_T1, merged_viq_T1)),
    ('T2 (>0.6)', (merged_sa_T2, merged_rrb_T2, merged_viq_T2))
]):
    ax = axes[idx]
    ax.patch.set_alpha(0)
    Xs = dfs[0]['Mean_Bias']
    Ys = sum(df['Total_Strength_Abs'].values for df in dfs)
    ax.scatter(Xs, Ys, color='#377eb8', edgecolor='k', s=80, alpha=0.9, zorder=2)
    linres = linregress(Xs, Ys)
    X_fit = np.linspace(Xs.min(), Xs.max(), 100)
    ax.plot(X_fit, linres.slope * X_fit + linres.intercept, color="#e6550d", linestyle="--", linewidth=2.2)
    ax.text(0.02, 0.98, f"$r$={linres.rvalue:.3f}, $p$={linres.pvalue:.4f}\nN={len(Xs)}",
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    ax.set_xlabel('Mean_Bias', fontsize=13, fontweight='bold')
    ax.set_ylabel('fMRI Strength (SA+RRB+VIQ)', fontsize=13, fontweight='bold')
    ax.set_title(tier_name, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Atypical Connectivity (Fig2D)

# %%
analyze_fmri_gencic_correlation(Fig2D, bias_by_human_region, "Atypical (ALL)")
analyze_fmri_gencic_correlation(Fig2D, bias_by_human_region_T1, "Atypical (T1)")
analyze_fmri_gencic_correlation(Fig2D, bias_by_human_region_T2, "Atypical (T2)")

# %% [markdown]
# ## 9. Data Table

# %%
def calculate_fmri_phenotype_metrics(fmri_matrix, phenotype_name):
    metrics = pd.DataFrame(index=fmri_matrix.columns)
    metrics[f'{phenotype_name}_Total_Strength_Abs'] = fmri_matrix.abs().sum()
    return metrics

viq_metrics = calculate_fmri_phenotype_metrics(Fig2A, 'VIQ')
sa_metrics = calculate_fmri_phenotype_metrics(Fig2B, 'Social')
rrb_metrics = calculate_fmri_phenotype_metrics(Fig2C, 'RRB')

comprehensive_table = pd.DataFrame(index=RegionMapping['Representative_Human_Group'].unique())
comprehensive_table.index.name = 'Human_Region'
comprehensive_table = comprehensive_table.join(viq_metrics, how='left')
comprehensive_table = comprehensive_table.join(sa_metrics, how='left')
comprehensive_table = comprehensive_table.join(rrb_metrics, how='left')

gencic_bias_all = aggregate_bias_by_tier(gencic_data, RegionMapping)
gencic_bias_all.index.name = 'Human_Region'
comprehensive_table = comprehensive_table.join(gencic_bias_all[['Mean_Bias']], how='left')
comprehensive_table = comprehensive_table.sort_values('Mean_Bias', ascending=False, na_position='last')

output_file = os.path.join(BUCH_DIR, 'Comprehensive_fMRI_GENCIC_Table.csv')
comprehensive_table.to_csv(output_file)
print(f"Saved: {output_file} ({comprehensive_table.shape})")
print(comprehensive_table.round(4).to_string())
