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
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
sys.path.insert(1, os.path.join(ProjDIR, "scripts"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted,
    Mut2GeneDF, Filt_LGD_Mis,
)
from script_phenotype_bootstrap import bootstrap_phenotype_bias
from script_phenotype_permutation import permutation_test_phenotype

os.chdir(os.path.join(ProjDIR, "notebook_phenotype"))
print(f"Project root: {ProjDIR}")

SEED = 42

# %% [markdown]
# # 02. Phenotype Stratification
#
# Stratify ASD mutation carriers by six phenotype dimensions and compare
# brain-structure-level expression bias between High and Low groups.
#
# **Phenotypes analysed:**
# 1. RBS-R total (repetitive behaviour; higher = worse)
# 2. DCDQ total (motor coordination; higher = better)
# 3. Vineland ABC (adaptive behaviour; higher = better)
# 4. SRS total T-score (social responsiveness; higher = worse)
# 5. Age of first words (developmental milestone; higher = worse)
# 6. IQ / FSIQ (cognitive; split at 70, replicating NB06)
#
# Each phenotype is split at the median (or a clinical threshold for IQ),
# and the resulting mutation subsets are compared via:
# - Weighted structural bias (MouseSTR_AvgZ_Weighted)
# - Bootstrap CI on each group (1000 resamples)
# - Permutation test for between-group differences (10000 permutations)
#
# **Outputs:**
# - `results/phenotype/stratification/<pheno>_permutation.csv` -- per-structure p-values
# - `results/phenotype/cache/<group>.ALL.parquet` -- bootstrap bias matrices
# - `results/phenotype/figs/<pheno>_*.png` -- figures
# - Summary table printed at the end

# %% [markdown]
# ## 1. Setup: Load Data

# %%
# Load config and expression matrix
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
ExpZ2Mat = pd.read_parquet(f"../{expr_matrix_path}")
print(f"Expression matrix: {ExpZ2Mat.shape}")

# Gene annotations
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

# %%
# Load master table and subject gene weights from NB01
master = pd.read_parquet("../results/phenotype/mutation_phenotype_master.parquet")
with open("../results/phenotype/subject_gene_weights.pkl", "rb") as f:
    subject_gene_weights = pk.load(f)

print(f"Master table: {master.shape[0]} subjects x {master.shape[1]} columns")
print(f"Subject gene weights: {len(subject_gene_weights)} entries")

# %%
# Reload raw mutations and filter to HC genes + LGD/Dmis
# (needed for bootstrap and permutation functions which operate on mutation rows)
discov = pd.read_csv(
    os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Discov_DNVs.txt"),
    sep="\t", low_memory=False,
)
rep = pd.read_csv(
    os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Rep_DNVs.txt"),
    sep="\t", low_memory=False,
)
if "Cohort" not in rep.columns:
    rep["Cohort"] = "SPARK"
if "DNASource" not in rep.columns:
    rep["DNASource"] = "."
mut_all = pd.concat([discov, rep], ignore_index=True)

# HC gene filter
table_s7 = pd.read_excel(
    os.path.join(ProjDIR, "dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx"),
    sheet_name="Table S7", skiprows=2,
)
table_s7["pDenovoWEST_Meta"] = pd.to_numeric(
    table_s7["pDenovoWEST_Meta"], errors="coerce"
)
hc_gene_set = set(
    table_s7.loc[table_s7["pDenovoWEST_Meta"] <= 1.3e-6, "HGNC"].values
)
HighConfMuts = Filt_LGD_Mis(
    mut_all[mut_all["HGNC"].isin(hc_gene_set)].copy(), Dmis=True
)
print(f"High-confidence LGD/Dmis mutations: {len(HighConfMuts)}")
print(f"HC gene set: {len(hc_gene_set)} genes")

# %%
# Output directories
CACHE_DIR = os.path.join(ProjDIR, "results/phenotype/cache")
STRAT_DIR = os.path.join(ProjDIR, "results/phenotype/stratification")
FIG_DIR = os.path.join(ProjDIR, "results/phenotype/figs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STRAT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Region Palette and Utility Functions

# %%
# Region color palette (consistent with src/plot.py and NB06)
REGION_COLORS = {
    'Isocortex': '#268ad5',
    'Olfactory_areas': '#5ab4ac',
    'Cortical_subplate': '#7ac3fa',
    'Hippocampus': '#2c9d39',
    'Amygdala': '#742eb5',
    'Striatum': '#ed8921',
    'Thalamus': '#e82315',
    'Hypothalamus': '#c27ba0',
    'Midbrain': '#f6b26b',
    'Pallidum': '#2ECC71',
    'Cerebellum': '#8B4513',
    'Medulla': '#708090',
    'Pons': '#A0522D',
}

# Region display order (rostral to caudal)
REGION_ORDER = [
    'Isocortex', 'Olfactory_areas', 'Cortical_subplate',
    'Hippocampus', 'Amygdala', 'Striatum', 'Pallidum',
    'Thalamus', 'Hypothalamus', 'Midbrain', 'Pons',
    'Medulla', 'Cerebellum',
]

# Region override: BNST -> Amygdala for display
REGION_OVERRIDES = {
    'Bed_nuclei_of_the_stria_terminalis': 'Amygdala',
}


def _apply_region_overrides(bias_df):
    """Apply region overrides (BNST -> Amygdala) to a bias DataFrame with REGION column."""
    for struct, new_reg in REGION_OVERRIDES.items():
        if struct in bias_df.index and "REGION" in bias_df.columns:
            bias_df.loc[struct, "REGION"] = new_reg
    return bias_df


def drop_fromList(_list):
    """Replace underscores with spaces in structure names."""
    return [" ".join(x.split("_")) for x in _list]


def significance_bar(start, end, height, displaystring, linewidth=1.2,
                     markersize=8, fontsize=15, color='k'):
    """Draw a significance bar between two positions on the current axes."""
    plt.plot([start, end], [height] * 2, '-', color=color, lw=linewidth,
             marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    plt.text(0.5 * (start + end), height + 0.03, displaystring,
             ha='center', va='center', fontsize=fontsize)


def p_to_stars(p):
    """Convert a p-value to a significance star string."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''

# %% [markdown]
# ## 3. Stratification Framework

# %%
def stratify_and_analyze(pheno_col, pheno_label, split="median",
                         threshold=None, higher_is_worse=True,
                         n_boot=1000, n_perm=10000, n_jobs=10):
    """Stratify subjects by a phenotype, compute bias per group, and test differences.

    Parameters
    ----------
    pheno_col : str
        Column name in the master table (e.g. 'rbsr_total', 'iq_fsiq').
    pheno_label : str
        Human-readable label for plots and cache files (e.g. 'RBSR', 'IQ').
    split : str
        'median' to split at the median, or 'threshold' to use a fixed value.
    threshold : float or None
        Fixed threshold when split='threshold'. Ignored when split='median'.
    higher_is_worse : bool
        If True, subjects above the split are "More Affected" (e.g. RBS-R).
        If False, subjects above the split are "Less Affected" (e.g. IQ, DCDQ).
    n_boot : int
        Number of bootstrap iterations per group.
    n_perm : int
        Number of permutations for the between-group test.
    n_jobs : int
        Parallel workers for bootstrap and permutation.

    Returns
    -------
    dict with keys:
        'pheno_col', 'pheno_label', 'threshold', 'higher_is_worse',
        'n_high', 'n_low', 'high_label', 'low_label',
        'high_bias', 'low_bias' (DataFrames),
        'high_boot', 'low_boot' (bootstrap DataFrames),
        'perm_result' (DataFrame with Pvalue column),
        'high_muts', 'low_muts' (filtered mutation DataFrames)
    """
    # --- 1. Identify subjects with valid phenotype ---
    valid = master.dropna(subset=[pheno_col])
    valid_iids = set(valid["IID"].values)

    # --- 2. Determine split point ---
    if split == "median":
        split_val = valid[pheno_col].median()
    elif split == "threshold":
        assert threshold is not None, "Must provide threshold when split='threshold'"
        split_val = threshold
    else:
        raise ValueError(f"Unknown split type: {split}")

    # --- 3. Assign group labels ---
    if higher_is_worse:
        # Above threshold = More Affected
        high_iids = set(valid.loc[valid[pheno_col] > split_val, "IID"])
        low_iids = set(valid.loc[valid[pheno_col] <= split_val, "IID"])
        high_label = "More Affected"
        low_label = "Less Affected"
    else:
        # Below threshold = More Affected (e.g., low IQ, low DCDQ)
        high_iids = set(valid.loc[valid[pheno_col] <= split_val, "IID"])
        low_iids = set(valid.loc[valid[pheno_col] > split_val, "IID"])
        high_label = "More Affected"
        low_label = "Less Affected"

    # --- 4. Filter mutations to each group ---
    high_muts = HighConfMuts[HighConfMuts["IID"].isin(high_iids)].copy()
    low_muts = HighConfMuts[HighConfMuts["IID"].isin(low_iids)].copy()

    print(f"\n{'='*60}")
    print(f"Phenotype: {pheno_label} ({pheno_col})")
    print(f"  Split: {'median' if split == 'median' else 'threshold'} = {split_val}")
    print(f"  {high_label}: {len(high_iids)} subjects, {len(high_muts)} mutations")
    print(f"  {low_label}: {len(low_iids)} subjects, {len(low_muts)} mutations")
    print(f"  higher_is_worse={higher_is_worse}")

    # --- 5. Compute gene weights and bias per group ---
    high_gw = Mut2GeneDF(high_muts, LGD=True, Dmis=True,
                         gene_symbol_to_entrez=GeneSymbol2Entrez)
    low_gw = Mut2GeneDF(low_muts, LGD=True, Dmis=True,
                        gene_symbol_to_entrez=GeneSymbol2Entrez)
    high_bias = MouseSTR_AvgZ_Weighted(ExpZ2Mat, high_gw)
    low_bias = MouseSTR_AvgZ_Weighted(ExpZ2Mat, low_gw)

    print(f"  {high_label} genes: {len(high_gw)}, {low_label} genes: {len(low_gw)}")
    print(f"  {high_label} mean bias: {high_bias['EFFECT'].mean():.4f}")
    print(f"  {low_label} mean bias: {low_bias['EFFECT'].mean():.4f}")

    # --- 6. Bootstrap CI per group ---
    print(f"  Running bootstrap ({n_boot} iterations)...")
    high_boot_dict = bootstrap_phenotype_bias(
        high_muts, ExpZ2Mat, GeneSymbol2Entrez,
        n_boot=n_boot, n_jobs=n_jobs, seed=SEED,
        cache_dir=CACHE_DIR,
        group_name=f"{pheno_label}.MoreAffected",
    )
    high_boot = high_boot_dict["ALL"]

    low_boot_dict = bootstrap_phenotype_bias(
        low_muts, ExpZ2Mat, GeneSymbol2Entrez,
        n_boot=n_boot, n_jobs=n_jobs, seed=SEED + 100000,
        cache_dir=CACHE_DIR,
        group_name=f"{pheno_label}.LessAffected",
    )
    low_boot = low_boot_dict["ALL"]

    # --- 7. Permutation test (shuffle phenotype labels across mutations) ---
    # Build a mutation-level table annotated with the phenotype value.
    # The permutation function shuffles this column and re-splits.
    muts_with_pheno = HighConfMuts[HighConfMuts["IID"].isin(valid_iids)].copy()
    iid_to_pheno = dict(zip(valid["IID"], valid[pheno_col]))
    muts_with_pheno["_pheno_val"] = muts_with_pheno["IID"].map(iid_to_pheno)
    muts_with_pheno = muts_with_pheno.dropna(subset=["_pheno_val"])

    perm_cache = os.path.join(
        STRAT_DIR, f"{pheno_label}_permutation.csv"
    )
    print(f"  Running permutation test ({n_perm} permutations)...")
    perm_result = permutation_test_phenotype(
        muts_with_pheno,
        phenotype_col="_pheno_val",
        threshold=split_val,
        exp_mat=ExpZ2Mat,
        gene_symbol_to_entrez=GeneSymbol2Entrez,
        n_perm=n_perm, n_jobs=n_jobs, seed=SEED,
        cache_path=perm_cache,
    )

    n_sig_05 = (perm_result["Pvalue"] < 0.05).sum()
    n_sig_01 = (perm_result["Pvalue"] < 0.01).sum()
    print(f"  Significant structures: {n_sig_05} at p<0.05, {n_sig_01} at p<0.01")

    return {
        "pheno_col": pheno_col,
        "pheno_label": pheno_label,
        "threshold": split_val,
        "higher_is_worse": higher_is_worse,
        "n_high": len(high_iids),
        "n_low": len(low_iids),
        "high_label": high_label,
        "low_label": low_label,
        "high_bias": high_bias,
        "low_bias": low_bias,
        "high_boot": high_boot,
        "low_boot": low_boot,
        "perm_result": perm_result,
        "high_muts": high_muts,
        "low_muts": low_muts,
    }

# %% [markdown]
# ## 4. Plotting Function

# %%
def plot_phenotype_comparison(result, save=True):
    """Regional bar plot comparing More Affected vs Less Affected groups.

    Bars are grouped by brain region, ordered rostral-to-caudal. Bootstrap
    standard errors provide error bars. Significance stars from permutation
    p-values are drawn between bar pairs.

    Parameters
    ----------
    result : dict
        Output from stratify_and_analyze().
    save : bool
        If True, save figure to FIG_DIR.

    Returns
    -------
    matplotlib.figure.Figure
    """
    high_bias = result["high_bias"]
    low_bias = result["low_bias"]
    high_boot = result["high_boot"]
    low_boot = result["low_boot"]
    perm_df = result["perm_result"]
    pheno_label = result["pheno_label"]
    high_label = result["high_label"]
    low_label = result["low_label"]

    # Assign region and apply overrides
    structures = high_bias.index
    struct_regions = pd.Series(
        {s: REGION_OVERRIDES.get(s, str2reg.get(s, "Other")) for s in structures}
    )

    # Collect structures per region in display order
    region_structs = {}
    for reg in REGION_ORDER:
        strs = sorted(struct_regions[struct_regions == reg].index.tolist())
        if strs:
            region_structs[reg] = strs

    # Build plotting arrays
    fig = plt.figure(dpi=150, figsize=(28, 10))
    ax = fig.add_axes([0.06, 0.20, 0.72, 0.72])

    x_offset = 0
    all_ticks = []
    all_positions = []
    bar_width = 0.35
    region_boundaries = []  # for subtle vertical separators

    for reg in REGION_ORDER:
        if reg not in region_structs:
            continue
        strs = region_structs[reg]
        color = REGION_COLORS.get(reg, '#888888')

        # Sort structures by More Affected bias within each region
        sort_order = np.argsort(high_bias.loc[strs, "EFFECT"].values)
        strs_sorted = [strs[i] for i in sort_order]

        X = np.arange(len(strs_sorted))
        high_vals = high_bias.loc[strs_sorted, "EFFECT"].values
        low_vals = low_bias.loc[strs_sorted, "EFFECT"].values
        high_err = high_boot.loc[strs_sorted].std(axis=1).values
        low_err = low_boot.loc[strs_sorted].std(axis=1).values

        # More Affected: outlined (unfilled)
        ax.bar(X + x_offset - bar_width / 2, high_vals, yerr=high_err,
               color='none', width=bar_width,
               edgecolor=color, linewidth=2.5, capsize=2)
        # Less Affected: solid filled
        ax.bar(X + x_offset + bar_width / 2, low_vals, yerr=low_err,
               color=color, width=bar_width,
               edgecolor=color, linewidth=0.8, capsize=2, alpha=0.8)

        # Significance bars from permutation test
        for i, s in enumerate(strs_sorted):
            if s not in perm_df.index:
                continue
            p = perm_df.loc[s, "Pvalue"]
            stars = p_to_stars(p)
            if not stars:
                continue
            height = 0.04 + max(
                high_vals[i] + high_err[i],
                low_vals[i] + low_err[i],
            )
            left = X[i] + x_offset - bar_width / 2
            right = X[i] + x_offset + bar_width / 2
            significance_bar(left, right, height, stars,
                             markersize=6, fontsize=10)

        all_positions.extend(X + x_offset)
        all_ticks.extend(drop_fromList(strs_sorted))
        x_offset += len(strs_sorted)
        region_boundaries.append(x_offset - 0.5)

    # Faint vertical region separators
    for xb in region_boundaries[:-1]:
        ax.axvline(xb, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_ticks, rotation=60, ha='right',
                       rotation_mode="anchor", fontsize=7)
    ax.set_ylabel('Structural Bias', fontsize=14)
    ax.set_title(f'{pheno_label}: {high_label} vs {low_label} '
                 f'(N={result["n_high"]} vs {result["n_low"]}, '
                 f'split={result["threshold"]:.1f})',
                 fontsize=13)
    ax.grid(True, axis="y", alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=10)

    # --- Legends ---
    # Group legend
    leg_ax1 = fig.add_axes([0.80, 0.72, 0.18, 0.18])
    leg_ax1.axis('off')
    h1 = plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=2.5)
    h2 = plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', alpha=0.8)
    leg_ax1.legend([h1, h2], [high_label, low_label],
                   fontsize=11, frameon=False, loc='center')

    # Region legend
    leg_ax2 = fig.add_axes([0.80, 0.28, 0.18, 0.42])
    leg_ax2.axis('off')
    reg_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REGION_COLORS.get(r, '#888'),
                       edgecolor='black', linewidth=0.5)
        for r in REGION_ORDER if r in region_structs
    ]
    reg_labels = [r.replace('_', ' ') for r in REGION_ORDER if r in region_structs]
    leg_ax2.legend(reg_handles, reg_labels, fontsize=8,
                   frameon=False, loc='center', ncol=1)

    # P-value legend
    leg_ax3 = fig.add_axes([0.80, 0.10, 0.18, 0.16])
    leg_ax3.axis('off')
    star_handles = [
        plt.Line2D([], [], color='black', marker='', linestyle='None')
        for _ in range(3)
    ]
    leg_ax3.legend(star_handles,
                   ['* p < 0.05', '** p < 0.01', '*** p < 0.001'],
                   fontsize=9, frameon=False, loc='center')

    # Transparent backgrounds
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    if save:
        fig_path = os.path.join(FIG_DIR, f"{pheno_label}_stratification.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"  Saved: {fig_path}")

    plt.show()
    return fig

# %% [markdown]
# ## 5. Run All Six Phenotypes

# %%
# Define the phenotype specifications
PHENOTYPE_SPECS = [
    {
        "pheno_col": "rbsr_total",
        "pheno_label": "RBSR",
        "split": "median",
        "higher_is_worse": True,
    },
    {
        "pheno_col": "dcdq_total",
        "pheno_label": "DCDQ",
        "split": "median",
        "higher_is_worse": False,  # higher = better coordination
    },
    {
        "pheno_col": "vine_abc",
        "pheno_label": "Vineland",
        "split": "median",
        "higher_is_worse": False,  # lower = worse adaptive behaviour
    },
    {
        "pheno_col": "srs_total_t",
        "pheno_label": "SRS",
        "split": "median",
        "higher_is_worse": True,
    },
    {
        "pheno_col": "milestone_words_mos",
        "pheno_label": "FirstWords",
        "split": "median",
        "higher_is_worse": True,  # later first words = worse
    },
    {
        "pheno_col": "iq_fsiq",
        "pheno_label": "IQ",
        "split": "threshold",
        "threshold": 70,
        "higher_is_worse": False,  # lower IQ = more affected
    },
]

# %%
# Run stratification for each phenotype
all_results = {}
for spec in PHENOTYPE_SPECS:
    label = spec["pheno_label"]
    res = stratify_and_analyze(
        pheno_col=spec["pheno_col"],
        pheno_label=label,
        split=spec["split"],
        threshold=spec.get("threshold"),
        higher_is_worse=spec["higher_is_worse"],
        n_boot=1000,
        n_perm=10000,
        n_jobs=10,
    )
    all_results[label] = res

# %%
# Plot each phenotype
for label, res in all_results.items():
    plot_phenotype_comparison(res, save=True)

# %% [markdown]
# ## 6. Summary Table

# %%
# Build summary table across all phenotypes
summary_rows = []
for label, res in all_results.items():
    perm_df = res["perm_result"]
    n_sig_05 = int((perm_df["Pvalue"] < 0.05).sum())
    n_sig_01 = int((perm_df["Pvalue"] < 0.01).sum())

    # Top 5 structures by absolute bias difference (most different between groups)
    top5 = perm_df.nsmallest(5, "Pvalue")
    top5_strs = ", ".join(
        f"{s.replace('_', ' ')} (p={top5.loc[s, 'Pvalue']:.4f})"
        for s in top5.index
    )

    summary_rows.append({
        "Phenotype": label,
        "Column": res["pheno_col"],
        "Split": f"{res['threshold']:.1f}",
        "N_MoreAffected": res["n_high"],
        "N_LessAffected": res["n_low"],
        "Sig_p05": n_sig_05,
        "Sig_p01": n_sig_01,
        "Top5_Structures": top5_strs,
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("PHENOTYPE STRATIFICATION SUMMARY")
print("=" * 80)
for _, row in summary_df.iterrows():
    print(f"\n{row['Phenotype']} ({row['Column']}, split={row['Split']})")
    print(f"  N: {row['N_MoreAffected']} More Affected vs {row['N_LessAffected']} Less Affected")
    print(f"  Significant structures: {row['Sig_p05']} at p<0.05, {row['Sig_p01']} at p<0.01")
    print(f"  Top 5: {row['Top5_Structures']}")

# %%
# Save summary table
summary_path = os.path.join(STRAT_DIR, "phenotype_stratification_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

# %%
# Compact display
print("\nCompact summary:")
print(summary_df[["Phenotype", "N_MoreAffected", "N_LessAffected",
                   "Sig_p05", "Sig_p01"]].to_string(index=False))
