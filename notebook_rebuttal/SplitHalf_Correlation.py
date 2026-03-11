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
# # Split-Half Correlation Analysis
#
# Three-way comparison of within-disease and cross-disease split-half correlations:
# - **ASD–ASD**: Split Fu et al. top 200 ASD genes → half vs half
# - **DDD–DDD**: Split DDD-ExclASD 237 genes → half vs half
# - **ASD–DDD**: ASD half vs DDD half (cross-disease)
#
# If ASD and DDD target genuinely different spatial patterns, the cross-disease
# correlation should be lower than within-disease split-halves.

# %% [markdown]
# ## Section 1: Setup & Data Loading

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import (
    Fil2Dict, MouseSTR_AvgZ_Weighted, STR2Region,
    merge_bias_datasets, LoadGeneINFO,
)
from plot import REGION_COLORS, REGIONS_SEQ

try:
    os.chdir(f"{ProjDIR}/notebook_rebuttal/")
    print(f"Current working directory: {os.getcwd()}")
except Exception as e:
    print(f"Error: {e}")

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# Load config and expression matrix
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

STR_BiasMat = pd.read_parquet(
    f"../{config['analysis_types']['STR_ISH']['expr_matrix']}"
)
STR_Anno = STR2Region()

print(f"Z2 matrix: {STR_BiasMat.shape[0]} genes × {STR_BiasMat.shape[1]} structures")

# %% [markdown]
# ### ASD gene weights — Fu et al. 2022 top 200 (TADA p-value)

# %%
N_ASD_POOL = 200
FU_FILE = "../dat/Genetics/Fu_et_al_2022.xlsx"

fu_SSCASC = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 5")
fu_SPARK = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 6")
fu_TADA_PR = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 8")
fu_TADA_PR = fu_TADA_PR.set_index("gene_gencodeV33")
fu_S11 = pd.read_excel(FU_FILE, sheet_name="Supplementary Table 11")
fu_S11 = fu_S11[fu_S11["gene_id"].notna()]
fu_S11 = fu_S11.sort_values(by="p_TADA_ASD", ascending=True)

fu_top_genes = fu_S11.head(N_ASD_POOL)["gene_gencodeV33"]


def GeneWeights_Fu2022(gene_list, prior_df, mut_dfs):
    """Compute gene weights for Fu et al. gene sets using TADA priors."""
    gene2MutN = {}
    for mut_df in mut_dfs:
        df_filt = mut_df[mut_df["gene_gencodeV33"].isin(gene_list)]
        for _, row in df_filt.iterrows():
            symbol = row["gene_gencodeV33"]
            try:
                g = GeneSymbol2Entrez[symbol]
            except KeyError:
                continue
            try:
                PR_LGD = prior_df.loc[symbol, "prior.dn.ptv"]
                PR_MisA = prior_df.loc[symbol, "prior.dn.misa"]
                PR_MisB = prior_df.loc[symbol, "prior.dn.misb"]
            except KeyError:
                continue
            weight = (
                row["dn.ptv"] * PR_LGD +
                row["dn.misb"] * PR_MisB +
                row["dn.misa"] * PR_MisA
            )
            gene2MutN[int(g)] = gene2MutN.get(int(g), 0) + weight
    return gene2MutN


ASD_Gene2W = GeneWeights_Fu2022(fu_top_genes, fu_TADA_PR, [fu_SSCASC, fu_SPARK])
ASD_Gene2W = {g: w for g, w in ASD_Gene2W.items() if g in STR_BiasMat.index and w > 0}
print(f"ASD (Fu top {N_ASD_POOL}): {len(ASD_Gene2W)} genes in Z2 with weight > 0")

# %% [markdown]
# ### DDD gene weights — DDD top 285, excluding Fu FDR<0.05 ASD genes

# %%
# Identify ASD genes to exclude: Fu et al. FDR_TADA_ASD < 0.05
fu_asd_fdr05 = fu_S11[fu_S11["FDR_TADA_ASD"] < 0.05]["gene_gencodeV33"]
fu_asd_entrez = set()
for s in fu_asd_fdr05:
    try:
        fu_asd_entrez.add(int(GeneSymbol2Entrez[s]))
    except KeyError:
        pass
print(f"Fu ASD FDR<0.05: {len(fu_asd_entrez)} genes to exclude")

# Load DDD top 285 and remove ASD-overlapping genes
DDD_Gene2W_raw = Fil2Dict("../dat/Genetics/GeneWeights/DDD.top285.gw")
DDD_Gene2W = {int(g): w for g, w in DDD_Gene2W_raw.items()
              if int(g) in STR_BiasMat.index and w > 0 and int(g) not in fu_asd_entrez}
n_excluded = sum(1 for g in DDD_Gene2W_raw if int(g) in fu_asd_entrez)
print(f"DDD top 285: {len(DDD_Gene2W_raw)} → excluded {n_excluded} ASD overlap "
      f"→ {len(DDD_Gene2W)} in Z2 with weight > 0")

# %%
# Compute full bias profiles
ASD_bias_full = MouseSTR_AvgZ_Weighted(STR_BiasMat, ASD_Gene2W)
DDD_bias_full = MouseSTR_AvgZ_Weighted(STR_BiasMat, DDD_Gene2W)

# Full ASD vs DDD correlation
common_full = ASD_bias_full.index.intersection(DDD_bias_full.index)
r_full_pearson, _ = pearsonr(
    ASD_bias_full.loc[common_full, 'EFFECT'],
    DDD_bias_full.loc[common_full, 'EFFECT']
)
r_full_spearman, _ = spearmanr(
    ASD_bias_full.loc[common_full, 'EFFECT'],
    DDD_bias_full.loc[common_full, 'EFFECT']
)
print(f"Full ASD vs DDD-ExclFuASD:  Pearson r = {r_full_pearson:.4f}, Spearman ρ = {r_full_spearman:.4f}")

# %% [markdown]
# ## Section 2: ASD vs DDD Reference Scatter

# %%
merged_asd_ddd = merge_bias_datasets(
    ASD_bias_full, DDD_bias_full, suffixes=('_ASD', '_DDD'),
    cols1=['Rank', 'EFFECT', 'REGION'],
)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=300, facecolor='none')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

for region in REGIONS_SEQ:
    mask = merged_asd_ddd['REGION'] == region
    if mask.sum() > 0:
        ax.scatter(
            merged_asd_ddd.loc[mask, 'EFFECT_DDD'],
            merged_asd_ddd.loc[mask, 'EFFECT_ASD'],
            color=REGION_COLORS[region],
            s=50, edgecolors='black', linewidth=0.4, alpha=0.8,
            label=region,
        )

x, y = merged_asd_ddd['EFFECT_DDD'], merged_asd_ddd['EFFECT_ASD']
fit = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, np.poly1d(fit)(x_line), 'r-', linewidth=2)

ax.set_xlabel('DDD-ExclFuASD EFFECT', fontsize=14)
ax.set_ylabel('ASD EFFECT (Fu top 200)', fontsize=14)
ax.set_title(f'Full ASD vs DDD-ExclFuASD (Pearson r = {r_full_pearson:.3f})', fontsize=14)
ax.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 3: Three-Way Split-Half Analysis
#
# For each iteration:
# 1. Stratified-split ASD genes → ASD-A, ASD-B
# 2. Stratified-split DDD genes → DDD-A, DDD-B
# 3. Compute three correlations: ASD-A vs ASD-B, DDD-A vs DDD-B, ASD-A vs DDD-A

# %%
rng = np.random.default_rng(seed=42)
n_splits = 100


def sorted_pool(d):
    """Return gene IDs and weights sorted by weight descending."""
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return np.array([g for g, _ in items]), np.array([w for _, w in items])


def stratified_split(n, rng):
    """Split indices into two balanced halves by rank."""
    idx_a, idx_b = [], []
    for start in range(0, n - 1, 2):
        if rng.random() < 0.5:
            idx_a.append(start)
            idx_b.append(start + 1)
        else:
            idx_a.append(start + 1)
            idx_b.append(start)
    if n % 2 == 1:
        if rng.random() < 0.5:
            idx_a.append(n - 1)
        else:
            idx_b.append(n - 1)
    return np.array(idx_a), np.array(idx_b)


asd_ids, asd_wts = sorted_pool(ASD_Gene2W)
ddd_ids, ddd_wts = sorted_pool(DDD_Gene2W)
n_asd, n_ddd = len(asd_ids), len(ddd_ids)

print(f"ASD: {n_asd} genes (split {n_asd//2} + {n_asd - n_asd//2})")
print(f"DDD: {n_ddd} genes (split {n_ddd//2} + {n_ddd - n_ddd//2})")
print(f"ASD weight range: {asd_wts.min():.2f} – {asd_wts.max():.2f}")
print(f"DDD weight range: {ddd_wts.min():.2f} – {ddd_wts.max():.2f}")

r_asd_asd = np.empty(n_splits)
r_ddd_ddd = np.empty(n_splits)
r_asd_ddd = np.empty(n_splits)

for i in range(n_splits):
    # Split ASD
    ia, ib = stratified_split(n_asd, rng)
    asd_bias_a = MouseSTR_AvgZ_Weighted(STR_BiasMat, dict(zip(asd_ids[ia], asd_wts[ia])))
    asd_bias_b = MouseSTR_AvgZ_Weighted(STR_BiasMat, dict(zip(asd_ids[ib], asd_wts[ib])))

    # Split DDD
    ja, jb = stratified_split(n_ddd, rng)
    ddd_bias_a = MouseSTR_AvgZ_Weighted(STR_BiasMat, dict(zip(ddd_ids[ja], ddd_wts[ja])))
    ddd_bias_b = MouseSTR_AvgZ_Weighted(STR_BiasMat, dict(zip(ddd_ids[jb], ddd_wts[jb])))

    # ASD-ASD
    c = asd_bias_a.index.intersection(asd_bias_b.index)
    r_asd_asd[i], _ = pearsonr(asd_bias_a.loc[c, 'EFFECT'], asd_bias_b.loc[c, 'EFFECT'])

    # DDD-DDD
    c = ddd_bias_a.index.intersection(ddd_bias_b.index)
    r_ddd_ddd[i], _ = pearsonr(ddd_bias_a.loc[c, 'EFFECT'], ddd_bias_b.loc[c, 'EFFECT'])

    # ASD half-A vs DDD half-A (cross-disease)
    c = asd_bias_a.index.intersection(ddd_bias_a.index)
    r_asd_ddd[i], _ = pearsonr(asd_bias_a.loc[c, 'EFFECT'], ddd_bias_a.loc[c, 'EFFECT'])

print(f"\nASD–ASD split-half:  mean={r_asd_asd.mean():.4f} ± {r_asd_asd.std():.4f}  "
      f"[{r_asd_asd.min():.4f}, {r_asd_asd.max():.4f}]")
print(f"DDD–DDD split-half:  mean={r_ddd_ddd.mean():.4f} ± {r_ddd_ddd.std():.4f}  "
      f"[{r_ddd_ddd.min():.4f}, {r_ddd_ddd.max():.4f}]")
print(f"ASD–DDD cross:       mean={r_asd_ddd.mean():.4f} ± {r_asd_ddd.std():.4f}  "
      f"[{r_asd_ddd.min():.4f}, {r_asd_ddd.max():.4f}]")
print(f"Full ASD vs DDD ref: r = {r_full_pearson:.4f}")

# %% [markdown]
# ## Section 4: Plots

# %% [markdown]
# ### 4.1 Three-Way Histogram

# %%
fig, ax = plt.subplots(figsize=(8, 5), dpi=300, facecolor='none')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

bins = np.linspace(
    min(r_asd_asd.min(), r_ddd_ddd.min(), r_asd_ddd.min()) - 0.02,
    max(r_asd_asd.max(), r_ddd_ddd.max(), r_asd_ddd.max()) + 0.02,
    25,
)

ax.hist(r_asd_asd, bins=bins, alpha=0.6, color='steelblue', edgecolor='white',
        label=f'ASD–ASD (mean {r_asd_asd.mean():.3f})')
ax.hist(r_ddd_ddd, bins=bins, alpha=0.6, color='darkorange', edgecolor='white',
        label=f'DDD–DDD (mean {r_ddd_ddd.mean():.3f})')
ax.hist(r_asd_ddd, bins=bins, alpha=0.6, color='mediumseagreen', edgecolor='white',
        label=f'ASD–DDD (mean {r_asd_ddd.mean():.3f})')
ax.axvline(r_full_pearson, color='red', linewidth=2.5, linestyle='--',
           label=f'Full ASD vs DDD r = {r_full_pearson:.3f}')

ax.set_xlabel('Pearson r', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Split-Half Correlations: Within-Disease vs Cross-Disease', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('../results/figs/split_half_three_way_histogram.pdf', transparent=True,
            dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.2 Paired Comparison: ASD–ASD vs ASD–DDD per Iteration

# %%
fig, ax = plt.subplots(figsize=(6, 6), dpi=300, facecolor='none')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.scatter(r_asd_asd, r_asd_ddd, color='steelblue', s=30, alpha=0.6,
           edgecolors='black', linewidth=0.3)

lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)

frac_below = (r_asd_ddd < r_asd_asd).sum() / n_splits
ax.set_xlabel('ASD–ASD split-half r', fontsize=13)
ax.set_ylabel('ASD–DDD cross r', fontsize=13)
ax.set_title(f'ASD–DDD < ASD–ASD in {frac_below*100:.0f}% of iterations', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figs/split_half_paired_comparison.pdf', transparent=True,
            dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.3 Summary Statistics

# %%
print("=" * 65)
print("Three-Way Split-Half Correlation Summary")
print("=" * 65)
print(f"  ASD source:         Fu et al. top {N_ASD_POOL} → {n_asd} in Z2 (TADA weights)")
print(f"  DDD source:         DDD top 285 excl Fu FDR<0.05 → {n_ddd} in Z2")
print(f"  N iterations:       {n_splits}")
print(f"  Split method:       Stratified (rank-paired)")
print("-" * 65)
print(f"  ASD–ASD split-half: {r_asd_asd.mean():.4f} ± {r_asd_asd.std():.4f}")
print(f"  DDD–DDD split-half: {r_ddd_ddd.mean():.4f} ± {r_ddd_ddd.std():.4f}")
print(f"  ASD–DDD cross:      {r_asd_ddd.mean():.4f} ± {r_asd_ddd.std():.4f}")
print(f"  Full ASD vs DDD:    {r_full_pearson:.4f}")
print("-" * 65)
frac = (r_asd_ddd < r_asd_asd).sum() / n_splits
print(f"  ASD–DDD < ASD–ASD:  {frac*100:.0f}% of iterations")
print("=" * 65)

# %% [markdown]
# ## Section 5: Interpretation
#
# The three-way split-half analysis compares within-disease reliability (ASD–ASD,
# DDD–DDD) to cross-disease correlation (ASD–DDD) using mutation-derived weights.
# If ASD and DDD target the same spatial pattern, ASD–DDD should overlap with
# the within-disease distributions. If they diverge, ASD–DDD should be
# systematically lower.
