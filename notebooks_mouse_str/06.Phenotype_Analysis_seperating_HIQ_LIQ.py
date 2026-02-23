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
from matplotlib.markers import TICKDOWN

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
sys.path.insert(1, os.path.join(ProjDIR, "scripts"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted,
    Dict2Fil, RegionDistributionsList,
    CountMut, Mut2GeneDF, Filt_LGD_Mis,
)
from script_phenotype_bootstrap import bootstrap_phenotype_bias
from script_phenotype_permutation import permutation_test_phenotype

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 06. Phenotype Analysis: Separating HIQ/LIQ and Male/Female
#
# This notebook stratifies ASD mutations by phenotype (IQ level, sex) and
# examines how structural bias patterns differ across subgroups.
#
# **Sections**:
# 1. Load data and define utility functions
# 2. IQ stratification — mutation counts and gene weights
# 3. IQ stratification — structural bias with bootstrap CI
# 4. Subsampling analysis: testing if LIQ pattern holds after matching sample size
# 5. Sex stratification — structural bias with bootstrap CI

# %% [markdown]
# ## 1. Load Data & Utility Functions

# %%
# Load config and expression matrix
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
ExpZ2Mat = pd.read_parquet(f"../{expr_matrix_path}")
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

print(f"Expression matrix: {ExpZ2Mat.shape}")


# %%
# --- Utility functions for phenotype analysis plots ---

def drop_fromList(_list):
    """Replace underscores with spaces in structure names."""
    return [" ".join(x.split("_")) for x in _list]


def significance_bar(start, end, height, displaystring, linewidth=1.2,
                     markersize=8, fontsize=15, color='k'):
    """Draw a significance bar between two positions."""
    plt.plot([start, end], [height] * 2, '-', color=color, lw=linewidth,
             marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    plt.text(0.5 * (start + end), height + 0.03, displaystring,
             ha='center', va='center', fontsize=fontsize)


# Region color palette (consistent across all plots)
REGION_COLORS = {
    'Isocortex': '#268ad5',
    'Olfactory_areas': '#5ab4ac',
    'Hippocampus': '#2c9d39',
    'Amygdala': '#742eb5',
    'Striatum': '#ed8921',
    'Thalamus': '#e82315',
}

# %% [markdown]
# ## 2. IQ Stratification — Mutation Counts

# %%
# Load ASC phenotype data
ASC_IQ_dat = pd.read_excel(
    "../dat/Genetics/1-s2.0-S0092867419313984-mmc4.xlsx",
    sheet_name="Phenotype"
)
ASC_IQ_dat = ASC_IQ_dat[ASC_IQ_dat["Role"] == "Proband"]
ASC_IQ_dat = ASC_IQ_dat.dropna(subset=['IQ'])
ASC_HIQ = ASC_IQ_dat[ASC_IQ_dat["IQ"] > 70]["Phenotype_ID"].values
ASC_LIQ = ASC_IQ_dat[ASC_IQ_dat["IQ"] <= 70]["Phenotype_ID"].values

# Load SPARK phenotype data
Spark_IQ_dat = pd.read_csv(
    "/home/jw3514/Work/ASD_Phenotype/Old/SPARK_Collection_Version6/core_descriptive_variables.csv"
)
Spark_IQ_dat = Spark_IQ_dat[Spark_IQ_dat["asd"] == True]
Spark_IQ_dat = Spark_IQ_dat.dropna(subset=['fsiq'])
Spark_HIQ = Spark_IQ_dat[Spark_IQ_dat["fsiq"] > 70]["subject_sp_id"].values
Spark_LIQ = Spark_IQ_dat[Spark_IQ_dat["fsiq"] <= 70]["subject_sp_id"].values

HighIQ = np.concatenate([ASC_HIQ, Spark_HIQ])
LowIQ = np.concatenate([ASC_LIQ, Spark_LIQ])

print(f"High IQ subjects: {len(HighIQ)} (ASC: {len(ASC_HIQ)}, SPARK: {len(Spark_HIQ)})")
print(f"Low IQ subjects:  {len(LowIQ)} (ASC: {len(ASC_LIQ)}, SPARK: {len(Spark_LIQ)})")

# %%
# Load SPARK de novo mutations
ASD_Discov_Muts = pd.read_csv("../dat/genes/SPARK/ASD_Discov_DNVs.txt", delimiter="\t")
ASD_Rep_Muts = pd.read_csv("../dat/genes/SPARK/ASD_Rep_DNVs.txt", delimiter="\t")
ASD_Muts = pd.concat([ASD_Discov_Muts, ASD_Rep_Muts])

# Filter to exome-wide significant genes (Zhou et al. 2022)
Spark_Meta_2stage = pd.read_excel(
    "../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
    skiprows=2, sheet_name="Table S7"
)
Spark_Meta_2stage = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] != "."]
Spark_Meta_HC = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] <= 1.3e-6]
HighConfGenes = Spark_Meta_HC["HGNC"].values
HighConfMuts = ASD_Muts[ASD_Muts["HGNC"].isin(HighConfGenes)]
HighConfMuts = Filt_LGD_Mis(HighConfMuts, Dmis=True)

# Split mutations by IQ group
HIQ_Muts = HighConfMuts[HighConfMuts["IID"].isin(HighIQ)]
LIQ_Muts = HighConfMuts[HighConfMuts["IID"].isin(LowIQ)]
print(f"\nHigh-confidence mutations: {len(HighConfMuts)}")
print(f"  High IQ: {HIQ_Muts.shape[0]}, Low IQ: {LIQ_Muts.shape[0]}")

# %%
# IQ distribution
fig, ax = plt.subplots(dpi=120)
ASC_IQ_dat["IQ"].hist(label="ASC", density=True, alpha=0.5, ax=ax)
Spark_IQ_dat["fsiq"].hist(label="SPARK", density=True, alpha=0.5, ax=ax)
ax.set_xlabel("IQ")
ax.legend()
ax.set_title("IQ Distribution")
plt.show()
print(f"ASC mean IQ: {ASC_IQ_dat['IQ'].mean():.1f}")
print(f"SPARK mean IQ: {Spark_IQ_dat['fsiq'].mean():.1f}")

# %%
# Mutation counts by type and IQ group
def classify_mutation_type(row):
    GeneEff = row["GeneEff"].split(";")[0]
    if GeneEff in ["frameshift", "splice_acceptor", "splice_donor",
                    "start_lost", "stop_gained", "stop_lost"]:
        return "LGD"
    elif GeneEff == "missense":
        revel = row["REVEL"].split(";")[0]
        if revel != ".":
            try:
                if float(revel) > 0.5:
                    return "Dmis"
            except Exception:
                pass
    return None

HIQ_IIDs = set(HIQ_Muts["IID"].values)
LIQ_IIDs = set(LIQ_Muts["IID"].values)
muts_without_iq = HighConfMuts[~HighConfMuts["IID"].isin(HIQ_IIDs.union(LIQ_IIDs))]

n_hiq, n_liq, n_no_iq = len(HIQ_Muts), len(LIQ_Muts), len(muts_without_iq)
n_hiq_lgd = len(HIQ_Muts[HIQ_Muts.apply(classify_mutation_type, axis=1) == "LGD"])
n_hiq_dmis = len(HIQ_Muts[HIQ_Muts.apply(classify_mutation_type, axis=1) == "Dmis"])
n_liq_lgd = len(LIQ_Muts[LIQ_Muts.apply(classify_mutation_type, axis=1) == "LGD"])
n_liq_dmis = len(LIQ_Muts[LIQ_Muts.apply(classify_mutation_type, axis=1) == "Dmis"])
n_noiq_lgd = len(muts_without_iq[muts_without_iq.apply(classify_mutation_type, axis=1) == "LGD"])
n_noiq_dmis = len(muts_without_iq[muts_without_iq.apply(classify_mutation_type, axis=1) == "Dmis"])

# Stacked bar plot: mutation counts by type and IQ group
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
categories = ['All', 'LGD', 'Dmis']
x = np.arange(len(categories))
width = 0.6

all_counts = [n_hiq, n_liq, n_no_iq]
lgd_counts = [n_hiq_lgd, n_liq_lgd, n_noiq_lgd]
dmis_counts = [n_hiq_dmis, n_liq_dmis, n_noiq_dmis]

for i, (counts, label) in enumerate(zip(
    [all_counts, lgd_counts, dmis_counts], categories
)):
    bottom = 0
    for j, (count, color, lbl) in enumerate(zip(
        counts,
        ['#AED6F1', '#21618C', '#D3D3D3'],
        ['High IQ (IQ > 70)', 'Low IQ (IQ <= 70)', 'No IQ Data']
    )):
        bar = ax.bar(x[i], count, width, bottom=bottom, color=color,
                     edgecolor='black', linewidth=1.5,
                     label=lbl if i == 0 else '')
        if count > 0:
            ax.text(x[i], bottom + count / 2, f'{int(count)}',
                    ha='center', va='center', fontweight='bold', fontsize=11)
        bottom += count

ax.set_ylabel('Number of Mutations', fontsize=14, fontweight='bold')
ax.set_title('Mutation Counts by Type and IQ Group', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

print(f"\nTotal: {sum(all_counts)} (HIQ: {n_hiq}, LIQ: {n_liq}, No IQ: {n_no_iq})")
print(f"LGD:   {sum(lgd_counts)} (HIQ: {n_hiq_lgd}, LIQ: {n_liq_lgd}, No IQ: {n_noiq_lgd})")
print(f"Dmis:  {sum(dmis_counts)} (HIQ: {n_hiq_dmis}, LIQ: {n_liq_dmis}, No IQ: {n_noiq_dmis})")

# %%
# Per-gene mutation breakdown by IQ group
all_genes = sorted(HighConfMuts["HGNC"].unique())
gene_counts_df = pd.DataFrame([{
    'gene': gene,
    'HIQ': len(HIQ_Muts[HIQ_Muts["HGNC"] == gene]),
    'LIQ': len(LIQ_Muts[LIQ_Muts["HGNC"] == gene]),
    'No_IQ': len(muts_without_iq[muts_without_iq["HGNC"] == gene]),
} for gene in all_genes])
gene_counts_df['total'] = gene_counts_df[['HIQ', 'LIQ', 'No_IQ']].sum(axis=1)
gene_counts_df = gene_counts_df.sort_values('total', ascending=False)

fig, ax = plt.subplots(figsize=(max(12, len(all_genes) * 0.5), 12), dpi=240)
x = np.arange(len(gene_counts_df))

ax.bar(x, gene_counts_df['No_IQ'].values, 0.8, label='No IQ Data',
       color='#D3D3D3', edgecolor='black', linewidth=1)
ax.bar(x, gene_counts_df['LIQ'].values, 0.8,
       bottom=gene_counts_df['No_IQ'].values,
       label='Low IQ (IQ <= 70)', color='#21618C', edgecolor='black', linewidth=1)
ax.bar(x, gene_counts_df['HIQ'].values, 0.8,
       bottom=gene_counts_df['No_IQ'].values + gene_counts_df['LIQ'].values,
       label='High IQ (IQ > 70)', color='#AED6F1', edgecolor='black', linewidth=1)

ax.set_xlabel('Gene', fontsize=28, fontweight='bold')
ax.set_ylabel('Number of Mutations', fontsize=28, fontweight='bold')
ax.set_title('Mutation Counts by Gene and IQ Group', fontsize=32, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(gene_counts_df['gene'].values, rotation=45, ha='right', fontsize=20)
ax.legend(loc='upper right', fontsize=22)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=20)
plt.tight_layout()
plt.show()

print(f"\nTop 10 genes by total mutations:")
print(gene_counts_df.head(10)[['gene', 'HIQ', 'LIQ', 'No_IQ', 'total']].to_string(index=False))

# %%
# Gene overlap between HIQ and LIQ
HIQ_genes = set(HIQ_Muts["HGNC"].unique())
LIQ_genes = set(LIQ_Muts["HGNC"].unique())
print(f"Unique genes in High IQ: {len(HIQ_genes)}")
print(f"Unique genes in Low IQ: {len(LIQ_genes)}")
print(f"HIQ only: {len(HIQ_genes - LIQ_genes)}, LIQ only: {len(LIQ_genes - HIQ_genes)}, "
      f"Overlap: {len(HIQ_genes & LIQ_genes)}")

# %% [markdown]
# ## 3. IQ Stratification — Structural Bias with Bootstrap CI

# %%
# Compute gene weights and structural bias for HIQ and LIQ
HIQ_GW = Mut2GeneDF(HIQ_Muts, LGD=True, Dmis=True,
                    gene_symbol_to_entrez=GeneSymbol2Entrez)
HIQ_DF = MouseSTR_AvgZ_Weighted(ExpZ2Mat, HIQ_GW,
                                 csv_fil="../dat/Unionize_bias/ASD.HIQ_spec.bias.csv")

LIQ_GW = Mut2GeneDF(LIQ_Muts, LGD=True, Dmis=True,
                    gene_symbol_to_entrez=GeneSymbol2Entrez)
LIQ_DF = MouseSTR_AvgZ_Weighted(ExpZ2Mat, LIQ_GW,
                                 csv_fil="../dat/Unionize_bias/ASD.LIQ_spec.bias.csv")

# Save gene weights
Dict2Fil(HIQ_GW, "../dat/Genetics/GeneWeights/ASD.HIQ.gw.csv")
Dict2Fil(LIQ_GW, "../dat/Genetics/GeneWeights/ASD.LIQ.gw.csv")

print(f"HIQ genes: {len(HIQ_GW)}, LIQ genes: {len(LIQ_GW)}")

# %%
# Load ASD circuit structures
ASD_CircuitsSet = pd.read_csv(
    "../results/STR_ISH/ASD.SA.Circuits.Size46.csv", index_col="idx"
)
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")
print(f"ASD circuit structures: {len(ASD_Circuits)}")
print(f"Region distribution:\n{RegionDistributionsList(ASD_Circuits)}")

CIR_REGIONS = ['Isocortex', 'Hippocampus', 'Amygdala', 'Striatum',
               'Thalamus', 'Olfactory_areas']
CIR_REGIONS_Dict = {reg: [] for reg in CIR_REGIONS}
for _str in ASD_Circuits:
    for reg in CIR_REGIONS:
        if str2reg[_str] == reg:
            CIR_REGIONS_Dict[reg].append(_str)
            break
CIR_REGIONS_Dict["Amygdala"].append("Bed_nuclei_of_the_stria_terminalis")

# %%
# Extract circuit structures for HIQ/LIQ
HIQ_Cir = HIQ_DF.loc[ASD_Circuits, :]
LIQ_Cir = LIQ_DF.loc[ASD_Circuits, :]
HIQ_Cir.loc["Bed_nuclei_of_the_stria_terminalis", "REGION"] = "Amygdala"
LIQ_Cir.loc["Bed_nuclei_of_the_stria_terminalis", "REGION"] = "Amygdala"

# %%
# Load IQ permutation p-values (cached or computed from legacy permutation files)
PERM_CACHE = "../results/Phenotype_permutation/IQ_Bias_Diff.Pvalues.csv"
LEGACY_PERM_DIR = "../dat/Unionize_bias/Permutations/IQ_permut"

IQ_PhenotypeDF = permutation_test_phenotype(
    pd.read_csv("../dat/Other/IQ_Mutation.Mar17.2023.csv"),
    "IQ", 70, ExpZ2Mat, GeneSymbol2Entrez,
    n_perm=10000, n_jobs=10, cache_path=PERM_CACHE,
    legacy_perm_dir=LEGACY_PERM_DIR,
    legacy_group1_prefix="HighIQ", legacy_group2_prefix="LowIQ",
)

# %%
# Load bootstrap bias (from parquet cache or legacy CSVs)
BOOT_CACHE = "../results/Phenotype_bootstrap"
LEGACY_BOOT = "../dat/Unionize_bias/Bootstrap"

print("Loading HIQ bootstrap bias...")
HIQ_boot = bootstrap_phenotype_bias(
    HIQ_Muts, ExpZ2Mat, GeneSymbol2Entrez,
    n_boot=1000, cache_dir=BOOT_CACHE, group_name="ASD.HIQ",
    legacy_csv_dir=f"{LEGACY_BOOT}/ASD.HIQ",
)["ALL"]

print("Loading LIQ bootstrap bias...")
LIQ_boot = bootstrap_phenotype_bias(
    LIQ_Muts, ExpZ2Mat, GeneSymbol2Entrez,
    n_boot=1000, cache_dir=BOOT_CACHE, group_name="ASD.LIQ",
    legacy_csv_dir=f"{LEGACY_BOOT}/ASD.LIQ",
)["ALL"]

print(f"Loaded HIQ {HIQ_boot.shape} + LIQ {LIQ_boot.shape} bootstrap replicates")


# %%
# Combined phenotype plot: HIQ vs LIQ per structure (dot plot with bootstrap CI)
def PlotPhenotype_dots(HIQ_Cir, LIQ_Cir, HIQ_boot, LIQ_boot,
                       IQ_PhenotypeDF, CIR_REGIONS, region_colors):
    """Plot HIQ vs LIQ bias for each circuit structure with bootstrap error bars."""
    fig = plt.figure(dpi=100, figsize=(30, 12))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    x_offset = 0
    all_ticks = []
    all_positions = []

    for REG in CIR_REGIONS:
        color = region_colors.get(REG, '#888888')
        HIQ_dat = HIQ_Cir[HIQ_Cir["REGION"] == REG]
        LIQ_dat = LIQ_Cir[LIQ_Cir["REGION"] == REG]
        if len(HIQ_dat) == 0:
            continue

        REG_STRs = HIQ_dat.index.values
        sort_idx = np.argsort(HIQ_dat["EFFECT"].values)
        X = np.arange(len(REG_STRs))

        # Bootstrap standard errors (from parquet DataFrame)
        HIQ_err = HIQ_boot.loc[REG_STRs].std(axis=1).values
        LIQ_err = LIQ_boot.loc[REG_STRs].std(axis=1).values

        dat1 = HIQ_dat["EFFECT"].values[sort_idx]
        dat2 = LIQ_dat["EFFECT"].values[sort_idx]
        err1 = HIQ_err[sort_idx]
        err2 = LIQ_err[sort_idx]

        # HIQ dots (open circles)
        ax.errorbar(X + x_offset - 0.2, dat1, yerr=err1,
                    fmt='o', markersize=10, color='none',
                    markeredgecolor=color, markeredgewidth=2,
                    ecolor=color, capsize=5)
        # LIQ dots (filled circles)
        ax.errorbar(X + x_offset + 0.2, dat2, yerr=err2,
                    fmt='o', markersize=10, color=color,
                    markeredgecolor=color, markeredgewidth=2,
                    ecolor=color, capsize=5)

        # Significance bars
        Pvalues = IQ_PhenotypeDF.loc[REG_STRs, "Pvalue"].values[sort_idx]
        for i, p in enumerate(Pvalues):
            if p >= 0.05:
                continue
            elif p < 0.001:
                ds = r'***'
            elif p < 0.01:
                ds = r'**'
            else:
                ds = r'*'
            height = 0.05 + max(dat1[i] + err1[i], dat2[i] + err2[i])
            significance_bar(X[i] + x_offset - 0.2, X[i] + x_offset + 0.2,
                             height, ds, markersize=20, fontsize=20)

        all_positions.extend(X + x_offset)
        all_ticks.extend(drop_fromList(REG_STRs[sort_idx]))
        x_offset += len(REG_STRs)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_ticks, rotation=45, ha='right', rotation_mode="anchor", fontsize=25)
    ax.set_ylabel('Mutation Bias', fontsize=30)
    ax.set_ylim(-0.2, 0.84)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=25)

    # Legends
    leg1 = fig.add_axes([0.85, 0.7, 0.1, 0.2])
    leg1.axis('off')
    h1 = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                     markeredgecolor='black', markersize=10, markeredgewidth=2)
    h2 = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                     markeredgecolor='black', markersize=10)
    leg1.legend([h1, h2], ['Higher IQ', 'Lower IQ'], fontsize=25, frameon=False)

    leg2 = fig.add_axes([0.88, 0.4, 0.1, 0.3])
    leg2.axis('off')
    handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c,
               markeredgecolor='black', markersize=10) for c in
               [region_colors.get(r, '#888') for r in CIR_REGIONS]]
    leg2.legend(handles, CIR_REGIONS, fontsize=25, frameon=False)

    leg3 = fig.add_axes([0.85, 0.1, 0.1, 0.1])
    leg3.axis('off')
    stars = [plt.Line2D([], [], color='black', marker='', linestyle='None') for _ in range(3)]
    leg3.legend(stars, ['* p < 0.05', '** p < 0.01', '*** p < 0.001'],
                fontsize=25, frameon=False)

    plt.show()
    return fig


fig = PlotPhenotype_dots(HIQ_Cir, LIQ_Cir, HIQ_boot, LIQ_boot,
                         IQ_PhenotypeDF, CIR_REGIONS, REGION_COLORS)


# %%
# Bar plot version (per region) — used in main text figure
def PlotPhenotype_bars(Dat1, Err1, Dat2, Err2, REG, IQ_PhenotypeDF,
                       label1="", label2="", fig_size=0):
    """Bar plot of HIQ vs LIQ bias per structure within a brain region."""
    Pvalues = IQ_PhenotypeDF.loc[Dat1.index.values, "Pvalue"].values
    sort_idx = np.argsort(Dat1["EFFECT"].values)
    X = np.arange(Dat1.shape[0])
    if fig_size == 0:
        fig_size = (2.5 + 0.4 * Dat1.shape[0], 6)

    fig, ax = plt.subplots(dpi=120, figsize=fig_size)

    dat1 = Dat1["EFFECT"].values[sort_idx]
    err1 = Err1[sort_idx]
    dat2 = Dat2["EFFECT"].values[sort_idx]
    err2 = Err2[sort_idx]

    bar_width = 0.3
    ax.bar(X - 0.15, dat1, yerr=err1, color="#AED6F1", width=bar_width,
           label=label1, edgecolor='black')
    ax.bar(X + 0.15, dat2, yerr=err2, color="#21618C", width=bar_width,
           label=label2, edgecolor='black')

    Pvalues = Pvalues[sort_idx]
    for i, p in enumerate(Pvalues):
        if p >= 0.05:
            continue
        elif p < 0.001:
            ds = r'***'
        elif p < 0.01:
            ds = r'**'
        else:
            ds = r'*'
        height = 0.05 + max(dat1[i] + err1[i], dat2[i] + err2[i])
        bar_centers = X[i] - 0.3 + np.array([0.5, 1.5]) * bar_width
        significance_bar(bar_centers[0], bar_centers[1], height, ds)

    ax.set_ylabel('Expression Bias', fontsize=15)
    ax.set_xticks(X)
    ax.set_xticklabels(drop_fromList(Dat1.index.values[sort_idx]),
                       rotation=45, fontsize=10, ha='right', rotation_mode="anchor")
    ax.set_title(" ".join(REG.split("_")), fontsize=20)
    ax.set_ylim(-0.2, 0.84)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig


for REG in CIR_REGIONS:
    HIQ_dat = HIQ_Cir[HIQ_Cir["REGION"] == REG]
    LIQ_dat = LIQ_Cir[LIQ_Cir["REGION"] == REG]
    if len(HIQ_dat) == 0:
        continue
    REG_STRs = HIQ_dat.index.values
    HIQ_err = HIQ_boot.loc[REG_STRs].std(axis=1).values
    LIQ_err = LIQ_boot.loc[REG_STRs].std(axis=1).values

    fig_size = (max(4.5, 0.8 * len(REG_STRs) + 2), 6)
    PlotPhenotype_bars(HIQ_dat, HIQ_err, LIQ_dat, LIQ_err, REG,
                       IQ_PhenotypeDF, "Higher IQ", "Lower IQ",
                       fig_size=fig_size)

# %% [markdown]
# ## 4. Subsampling Analysis
#
# Subsample LIQ mutations to match HIQ size and test if the bias pattern
# is robust to the larger LIQ sample size.

# %%
# Subsample LIQ to match HIQ size
N_ITERATIONS = 100
RANDOM_SEED = 42

n_hiq_muts = len(HIQ_Muts)
n_liq_muts = len(LIQ_Muts)
print(f"Subsampling LIQ ({n_liq_muts}) to match HIQ ({n_hiq_muts}), {N_ITERATIONS} iterations...")

# HIQ regional biases (baseline)
HIQ_regional_biases = {}
for REG in CIR_REGIONS:
    HIQ_regional_biases[REG] = HIQ_Cir[HIQ_Cir["REGION"] == REG]["EFFECT"].mean()

# Run subsampling
subsample_results = {'iteration': [], 'region': [], 'bias': []}
subsample_str_results = {'iteration': [], 'str': [], 'region': [], 'bias': []}

np.random.seed(RANDOM_SEED)
for iteration in range(N_ITERATIONS):
    LIQ_sub = LIQ_Muts.sample(n=n_hiq_muts, replace=False,
                               random_state=RANDOM_SEED + iteration)
    LIQ_sub_GW = Mut2GeneDF(LIQ_sub, LGD=True, Dmis=True,
                           gene_symbol_to_entrez=GeneSymbol2Entrez)
    LIQ_sub_DF = MouseSTR_AvgZ_Weighted(ExpZ2Mat, LIQ_sub_GW)

    LIQ_sub_Cir = LIQ_sub_DF.loc[ASD_Circuits, :]
    LIQ_sub_Cir.loc["Bed_nuclei_of_the_stria_terminalis", "REGION"] = "Amygdala"

    for REG in CIR_REGIONS:
        bias = LIQ_sub_Cir[LIQ_sub_Cir["REGION"] == REG]["EFFECT"].mean()
        subsample_results['iteration'].append(iteration)
        subsample_results['region'].append(REG)
        subsample_results['bias'].append(bias)

    for str_name in LIQ_sub_Cir.index:
        subsample_str_results['iteration'].append(iteration)
        subsample_str_results['str'].append(str_name)
        subsample_str_results['region'].append(LIQ_sub_Cir.loc[str_name, "REGION"])
        subsample_str_results['bias'].append(LIQ_sub_Cir.loc[str_name, "EFFECT"])

    if (iteration + 1) % 25 == 0:
        print(f"  Completed {iteration + 1}/{N_ITERATIONS}")

subsample_df = pd.DataFrame(subsample_results)
subsample_str_df = pd.DataFrame(subsample_str_results)

# %%
# Consistency check: how often is subsampled LIQ > HIQ?
print("Consistency: % of subsamples where LIQ > HIQ by region")
print("-" * 50)
for REG in CIR_REGIONS:
    liq_biases = subsample_df[subsample_df['region'] == REG]['bias'].values
    hiq_bias = HIQ_regional_biases[REG]
    pct = np.sum(liq_biases > hiq_bias) / N_ITERATIONS * 100
    print(f"  {REG:20s}: {pct:5.1f}%")

# %%
# Subsampling visualization: structure-level
str_summary = subsample_str_df.groupby('str')['bias'].agg(['mean', 'std']).reset_index()
str_summary.columns = ['str', 'mean_bias', 'std_bias']

HIQ_str_biases = {s: HIQ_Cir.loc[s, "EFFECT"] for s in HIQ_Cir.index}
str_summary['HIQ_bias'] = str_summary['str'].map(HIQ_str_biases)
str_summary['region'] = str_summary['str'].map(
    {s: HIQ_Cir.loc[s, "REGION"] for s in HIQ_Cir.index}
)

fig = plt.figure(figsize=(30, 12), dpi=150)
ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

x_offset = 0
all_ticks = []
all_positions = []

for REG in CIR_REGIONS:
    reg_strs = str_summary[str_summary['region'] == REG].copy()
    if len(reg_strs) == 0:
        continue
    reg_strs = reg_strs.sort_values('HIQ_bias')
    X = np.arange(len(reg_strs))
    color = REGION_COLORS.get(REG, '#888')

    # HIQ diamonds
    ax.scatter(X + x_offset, reg_strs['HIQ_bias'].values, color='#AED6F1', s=200,
               marker='D', edgecolor='black', linewidth=2, zorder=10,
               label='HIQ' if REG == CIR_REGIONS[0] else '')

    # LIQ subsampled error bars
    for i, (_, row) in enumerate(reg_strs.iterrows()):
        ax.errorbar(i + x_offset, row['mean_bias'], yerr=row['std_bias'],
                    fmt='o', markersize=8, color=color,
                    markeredgecolor='black', markeredgewidth=1.5,
                    ecolor=color, capsize=4, alpha=0.8, zorder=5)

    all_positions.extend(X + x_offset)
    all_ticks.extend(drop_fromList(reg_strs['str'].values))
    x_offset += len(reg_strs)

ax.set_xticks(all_positions)
ax.set_xticklabels(all_ticks, rotation=45, ha='right', rotation_mode="anchor", fontsize=12)
ax.set_ylabel('Structural Bias', fontsize=20, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(-0.2, 0.84)
ax.tick_params(axis='y', labelsize=16)
ax.set_title('Structure-Level Bias: HIQ vs Subsampled LIQ (100 iterations)',
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

liq_higher = str_summary[str_summary['mean_bias'] > str_summary['HIQ_bias']]
print(f"\nStructures where LIQ mean > HIQ: {len(liq_higher)}/{len(str_summary)} "
      f"({len(liq_higher)/len(str_summary)*100:.1f}%)")

# %% [markdown]
# ## 5. Sex Stratification — Structural Bias

# %%
# Load phenotype data and split by sex
ASC_sex_dat = pd.read_excel(
    "../dat/Genetics/1-s2.0-S0092867419313984-mmc4.xlsx",
    sheet_name="Phenotype"
)
ASC_sex_dat = ASC_sex_dat[ASC_sex_dat["Role"] == "Proband"]
ASC_sex_dat = ASC_sex_dat.dropna(subset=['IQ'])
ASC_Male = ASC_sex_dat[ASC_sex_dat["Sex"] == "Male"]["Phenotype_ID"].values
ASC_Female = ASC_sex_dat[ASC_sex_dat["Sex"] == "Female"]["Phenotype_ID"].values

Spark_sex_dat = pd.read_csv(
    "/home/jw3514/Work/ASD_Phenotype/Old/SPARK_Collection_Version6/core_descriptive_variables.csv"
)
Spark_sex_dat = Spark_sex_dat[Spark_sex_dat["asd"] == True]
Spark_sex_dat = Spark_sex_dat.dropna(subset=['fsiq'])
Spark_Male = Spark_sex_dat[Spark_sex_dat["sex"] == "Male"]["subject_sp_id"].values
Spark_Female = Spark_sex_dat[Spark_sex_dat["sex"] == "Female"]["subject_sp_id"].values

Male = np.concatenate([ASC_Male, Spark_Male])
Female = np.concatenate([ASC_Female, Spark_Female])

# Same mutations as IQ section (HighConfMuts already filtered)
Male_Muts = HighConfMuts[HighConfMuts["IID"].isin(Male)]
Female_Muts = HighConfMuts[HighConfMuts["IID"].isin(Female)]
print(f"Male mutations: {Male_Muts.shape[0]}, Female mutations: {Female_Muts.shape[0]}")

# Save filtered mutations
Male_Muts.to_csv("../dat/Unionize_bias/Gender.Male.Highconf.Muts.csv", index=False)
Female_Muts.to_csv("../dat/Unionize_bias/Gender.Female.Highconf.Muts.csv", index=False)

# %%
# Compute gene weights and structural bias for Male/Female (ALL, LGD-only, Dmis-only)
Male_GW = Mut2GeneDF(Male_Muts, LGD=True, Dmis=True,
                     gene_symbol_to_entrez=GeneSymbol2Entrez)
Male_GW_LGD = Mut2GeneDF(Male_Muts, LGD=True, Dmis=False,
                          gene_symbol_to_entrez=GeneSymbol2Entrez)
Male_GW_Dmis = Mut2GeneDF(Male_Muts, LGD=False, Dmis=True,
                           gene_symbol_to_entrez=GeneSymbol2Entrez)
Male_ALL = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Male_GW,
                                   csv_fil="../dat/Unionize_bias/ASD.Male.ALL.bias.csv")
Male_LGD = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Male_GW_LGD,
                                   csv_fil="../dat/Unionize_bias/ASD.Male.LGD.bias.csv")
Male_Dmis = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Male_GW_Dmis,
                                    csv_fil="../dat/Unionize_bias/ASD.Male.Dmis.bias.csv")

Female_GW = Mut2GeneDF(Female_Muts, LGD=True, Dmis=True,
                       gene_symbol_to_entrez=GeneSymbol2Entrez)
Female_GW_LGD = Mut2GeneDF(Female_Muts, LGD=True, Dmis=False,
                            gene_symbol_to_entrez=GeneSymbol2Entrez)
Female_GW_Dmis = Mut2GeneDF(Female_Muts, LGD=False, Dmis=True,
                             gene_symbol_to_entrez=GeneSymbol2Entrez)
Female_ALL = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Female_GW,
                                     csv_fil="../dat/Unionize_bias/ASD.Female.ALL.bias.csv")
Female_LGD = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Female_GW_LGD,
                                     csv_fil="../dat/Unionize_bias/ASD.Female.LGD.bias.csv")
Female_Dmis = MouseSTR_AvgZ_Weighted(ExpZ2Mat, Female_GW_Dmis,
                                      csv_fil="../dat/Unionize_bias/ASD.Female.Dmis.bias.csv")

# %%
# Extract circuit structures
Male_ALL_Cir = Male_ALL.loc[ASD_Circuits, :]
Female_ALL_Cir = Female_ALL.loc[ASD_Circuits, :]
Male_ALL_Cir.loc["Bed_nuclei_of_the_stria_terminalis", "REGION"] = "Amygdala"
Female_ALL_Cir.loc["Bed_nuclei_of_the_stria_terminalis", "REGION"] = "Amygdala"

# Load bootstrap bias (from parquet cache or legacy CSVs)
print("Loading Male bootstrap bias...")
Male_boot = bootstrap_phenotype_bias(
    Male_Muts, ExpZ2Mat, GeneSymbol2Entrez,
    n_boot=1000, cache_dir=BOOT_CACHE, group_name="ASD.Male",
    legacy_csv_dir=f"{LEGACY_BOOT}/ASD.Male",
)["ALL"]

print("Loading Female bootstrap bias...")
Female_boot = bootstrap_phenotype_bias(
    Female_Muts, ExpZ2Mat, GeneSymbol2Entrez,
    n_boot=1000, cache_dir=BOOT_CACHE, group_name="ASD.Female",
    legacy_csv_dir=f"{LEGACY_BOOT}/ASD.Female",
)["ALL"]

print(f"Loaded Male {Male_boot.shape} + Female {Female_boot.shape} bootstrap replicates")

# %%
# Regional bias comparison: Male vs Female
Male_dat = []
Female_dat = []
Male_err = []
Female_err = []
for REG in CIR_REGIONS:
    Male_dat.append(Male_ALL_Cir[Male_ALL_Cir["REGION"] == REG]["EFFECT"].mean())
    Female_dat.append(Female_ALL_Cir[Female_ALL_Cir["REGION"] == REG]["EFFECT"].mean())
    Male_err.append(Male_boot.loc[CIR_REGIONS_Dict[REG]].mean(axis=0).std())
    Female_err.append(Female_boot.loc[CIR_REGIONS_Dict[REG]].mean(axis=0).std())

Male_dat = np.array(Male_dat)
Female_dat = np.array(Female_dat)
Male_err = np.array(Male_err)
Female_err = np.array(Female_err)

sort_idx = np.argsort(Male_dat)

fig, ax = plt.subplots(dpi=480)
X = np.arange(len(CIR_REGIONS))
ax.bar(X - 0.1, Male_dat[sort_idx], yerr=Male_err[sort_idx],
       color="#e31a1c", width=0.2, label="Male", edgecolor='black')
ax.bar(X + 0.1, Female_dat[sort_idx], yerr=Female_err[sort_idx],
       color="#ff7f00", width=0.2, label="Female", edgecolor='black')

ax.set_ylabel('Mutation Bias', fontsize=20)
plt.xticks(X, np.array(CIR_REGIONS)[sort_idx], rotation=45, fontsize=10,
           ha='right', rotation_mode="anchor")
ax.legend(loc="upper left")
ax.grid(True, axis="y", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Output | Description |
# |--------|-------------|
# | `ASD.HIQ_spec.bias.csv` | HIQ structural bias |
# | `ASD.LIQ_spec.bias.csv` | LIQ structural bias |
# | `ASD.Male.ALL.bias.csv` | Male structural bias (ALL mutations) |
# | `ASD.Female.ALL.bias.csv` | Female structural bias (ALL mutations) |
# | `Gender.Male.Highconf.Muts.csv` | Male high-confidence mutation list |
# | `Gender.Female.Highconf.Muts.csv` | Female high-confidence mutation list |
