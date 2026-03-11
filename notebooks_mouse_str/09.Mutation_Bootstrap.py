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
# # Mutation Bootstrap Analysis
#
# Produces bootstrap bias profiles for three gene sets:
# 1. **ASD/SPARK** — weighted & uniform mutation resampling (61 genes)
# 2. **DDD** — weighted mutation resampling (237 genes, ASD-excluded)
# 3. **LOEUF25** — gene set resampling (constrained genes)
#
# Then computes:
# - **Residual CI** for DDD and LOEUF25 (regression residuals vs ASD bias)

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
Anno = STR2Region()

N_BOOT = 1000
N_WORKERS = 10

# %% [markdown]
# # Section 1: ASD/SPARK Bootstrap

# %%
# Load SPARK mutation data
Spark_Meta_2stage = pd.read_excel("../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
                                  skiprows=2, sheet_name="Table S7")
Spark_Meta_2stage = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] != "."]
Spark_Meta_ExomeWide = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"] <= 1.3e-6]
print(f"SPARK ExomeWide significant genes: {Spark_Meta_ExomeWide.shape[0]}")
print(f"  LoF total: {Spark_Meta_ExomeWide['AutismMerged_LoF'].sum()}")
print(f"  Dmis total: {Spark_Meta_ExomeWide['AutismMerged_Dmis_REVEL0.5'].sum()}")

# %%
# Compute observed ASD gene weights and bias (no BGMR correction — matches main analysis)
Spark_ExomeWide_Genes = Spark_Meta_ExomeWide[["GeneID", "EntrezID", "HGNC", "ExACpLI",
                                               "LOEUF", "AutismMerged_LoF",
                                               "AutismMerged_Dmis_REVEL0.5", "pDenovoWEST_Meta"]]

_, ASD_GW_observed = SPARK_Gene_Weights(Spark_Meta_ExomeWide, None, Bmis=False)
ASD_STR_Bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, ASD_GW_observed)
print(f"ASD gene weights: {len(ASD_GW_observed)} genes")

# %%
# Bootstrap ASD mutations (weighted + uniform)
boot_DFs_ASD_weighted = bootstrap_gene_mutations(Spark_ExomeWide_Genes, N_BOOT, weighted=True)
boot_DFs_ASD_uniform = bootstrap_gene_mutations(Spark_ExomeWide_Genes, N_BOOT, weighted=False)
print(f"Created {N_BOOT} weighted + {N_BOOT} uniform ASD bootstrap replicates")

# %%
# Parallel ASD bootstrap bias computation
def process_ASD_bootstrap_iter(args):
    """Worker: compute ASD bootstrap bias for one iteration (no BGMR)."""
    i, DF, save_dir, str_bias_mat = args
    _, boot_gw = SPARK_Gene_Weights(DF, None, Bmis=False)
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"))
    return i, boot_bias

# Weighted resampling
save_dir_asd_w = "../results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling"
os.makedirs(save_dir_asd_w, exist_ok=True)

args_w = [(i, DF, save_dir_asd_w, STR_BiasMat)
          for i, DF in enumerate(boot_DFs_ASD_weighted)]

boot_bias_ASD_weighted = [None] * N_BOOT
print(f"Computing ASD weighted bootstrap bias ({N_WORKERS} workers)...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_ASD_bootstrap_iter, a): a[0] for a in args_w}
    done = 0
    for future in as_completed(futures):
        i, bias = future.result()
        boot_bias_ASD_weighted[i] = bias
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{N_BOOT}")
print(f"Done: {N_BOOT} ASD weighted bootstrap iterations")

# %%
# Uniform resampling
def process_ASD_bootstrap_iter_uniform(args):
    """Worker: compute ASD bootstrap bias (uniform) for one iteration (no BGMR)."""
    i, DF, save_dir, str_bias_mat = args
    _, boot_gw = SPARK_Gene_Weights(DF, None)
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"Spark_ExomeWide.GeneWeight.boot{i}.csv"))
    return i, boot_bias

save_dir_asd_u = "../results/Bootstrap_bias/Spark_ExomeWide/Uniform_Resampling"
os.makedirs(save_dir_asd_u, exist_ok=True)

args_u = [(i, DF, save_dir_asd_u, STR_BiasMat)
          for i, DF in enumerate(boot_DFs_ASD_uniform)]

boot_bias_ASD_uniform = [None] * N_BOOT
print(f"Computing ASD uniform bootstrap bias ({N_WORKERS} workers)...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_ASD_bootstrap_iter_uniform, a): a[0] for a in args_u}
    done = 0
    for future in as_completed(futures):
        i, bias = future.result()
        boot_bias_ASD_uniform[i] = bias
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{N_BOOT}")
print(f"Done: {N_BOOT} ASD uniform bootstrap iterations")

# %% [markdown]
# # Section 2: DDD Bootstrap

# %%
# Load DDD data and generate gene weight files
# Kaplanis et al. 2020: "We identified 285 genes that were significantly associated
# with developmental disorders" using Bonferroni correction (0.05 / (18762 * 2)).
df_ddd = pd.read_excel("../dat/Genetics/41586_2020_2832_MOESM4_ESM.xlsx")
df_ddd = df_ddd.sort_values("denovoWEST_p_full")
hc_df = df_ddd.nsmallest(285, "denovoWEST_p_full")
hc_df = hc_df.copy()
hc_df["EntrezID"] = [int(GeneSymbol2Entrez.get(x, -1)) for x in hc_df["symbol"].values]
print(f"DDD significant genes: {hc_df.shape[0]}")

# %%
# Produce DDD.top285.gw (canonical source for this file)
DDD_top285_GW = Aggregate_Gene_Weights_NDD(hc_df, out="../dat/Genetics/GeneWeights/DDD.top285.gw")
print(f"DDD.top285.gw: {len(DDD_top285_GW)} genes, top weight: {max(DDD_top285_GW.values()):.3f}")

# %%
# Exclude ASD genes (using DN weights) — done before column subsetting
ASD_DN_GW = Fil2Dict(ProjDIR + "dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw")
ASD_GENES = list(ASD_DN_GW.keys())
hc_df_excl = hc_df[~hc_df["EntrezID"].isin(ASD_GENES)]
print(f"DDD genes: {len(hc_df)}, after excluding {len(ASD_GENES)} ASD genes: {len(hc_df_excl)}")

# Produce DDD.top237.ExcludeASD.gw (canonical source for this file)
DDD_ExcludeASD_GW = Aggregate_Gene_Weights_NDD(hc_df_excl, out="../dat/Genetics/GeneWeights/DDD.top237.ExcludeASD.gw")
print(f"DDD.top237.ExcludeASD.gw: {len(DDD_ExcludeASD_GW)} genes")

# %%
# Prepare columns for NDD bootstrap (aggregate LoF variants)
hc_df_excl = hc_df_excl.copy()
hc_df_excl["NDD_LoF"] = (
    hc_df_excl["frameshift_variant"].fillna(0)
    + hc_df_excl["splice_acceptor_variant"].fillna(0)
    + hc_df_excl["splice_donor_variant"].fillna(0)
    + hc_df_excl["stop_gained"].fillna(0)
    + hc_df_excl["stop_lost"].fillna(0)
).astype(int).clip(lower=0)
hc_df_excl["NDD_Dmis"] = hc_df_excl["missense_variant"].fillna(0).astype(int).clip(lower=0)
hc_df_excl = hc_df_excl[["EntrezID", "symbol", "NDD_LoF", "NDD_Dmis"]]
hc_df_excl = hc_df_excl.set_index("EntrezID", drop=False)
print(f"DDD bootstrap input: {len(hc_df_excl)} genes, "
      f"LoF total: {hc_df_excl['NDD_LoF'].sum()}, Dmis total: {hc_df_excl['NDD_Dmis'].sum()}")

# %%
# Bootstrap DDD mutations (weighted)
boot_DFs_DDD = bootstrap_gene_mutations(hc_df_excl, N_BOOT, weighted=True,
                                         lof_col="NDD_LoF", dmis_col="NDD_Dmis")
print(f"Created {N_BOOT} DDD weighted bootstrap replicates")

# %%
# Parallel DDD bootstrap bias computation
def ndd_gene_weights(df, lof_col="NDD_LoF", dmis_col="NDD_Dmis"):
    """Compute NDD gene weights: weight = nLGD * 0.347 + nMis * 0.194."""
    return {int(row["EntrezID"]): row[lof_col] * 0.347 + row[dmis_col] * 0.194
            for _, row in df.iterrows()}

def process_DDD_bootstrap_iter(args):
    """Worker: compute DDD bootstrap bias for one iteration."""
    i, DF, save_dir, str_bias_mat = args
    boot_gw = ndd_gene_weights(DF)
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"DDD_ExomeWide.GeneWeight.boot{i}.csv"))
    return i, boot_bias

save_dir_ddd = "../results/Bootstrap_bias/DDD_ExomeWide/Weighted_Resampling"
os.makedirs(save_dir_ddd, exist_ok=True)

args_ddd = [(i, DF, save_dir_ddd, STR_BiasMat) for i, DF in enumerate(boot_DFs_DDD)]

boot_bias_DDD = [None] * N_BOOT
print(f"Computing DDD bootstrap bias ({N_WORKERS} workers)...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_DDD_bootstrap_iter, a): a[0] for a in args_ddd}
    done = 0
    for future in as_completed(futures):
        i, bias = future.result()
        boot_bias_DDD[i] = bias
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{N_BOOT}")
print(f"Done: {N_BOOT} DDD bootstrap iterations")

# %% [markdown]
# # Section 3: LOEUF25 Bootstrap

# %%
# Load gnomAD v4 constraint data and define LOEUF25 gene set
gnomad4 = pd.read_csv("../dat/Genetics/gnomad.v4.0.constraint_metrics.tsv", sep="\t")
gnomad4 = gnomad4[gnomad4["transcript"].str.contains("ENST")]
gnomad4 = gnomad4[gnomad4["mane_select"] == True]

gnomad4["Entrez"] = gnomad4["gene"].map(GeneSymbol2Entrez).fillna(0).astype(int)
gnomad4 = gnomad4[gnomad4["Entrez"] != 0]

bottom_25_threshold = gnomad4["lof.oe_ci.upper"].quantile(0.25)
gnomad4_bottom25 = gnomad4[gnomad4["lof.oe_ci.upper"] <= bottom_25_threshold]
LOEUF25_genes = gnomad4_bottom25["Entrez"].unique().tolist()
print(f"LOEUF25 gene set: {len(LOEUF25_genes)} genes (threshold: {bottom_25_threshold:.4f})")

# %%
# Bootstrap gene sets from LOEUF25
rng = np.random.default_rng(42)
gene_arr = np.array(LOEUF25_genes)
LOEUF25_boot_gene_sets = [rng.choice(gene_arr, size=len(gene_arr), replace=True).tolist()
                          for _ in range(N_BOOT)]
print(f"Created {N_BOOT} LOEUF25 bootstrap replicates")

# %%
# Parallel LOEUF25 bootstrap bias computation
def process_LOEUF25_bootstrap_iter(args):
    """Worker: compute LOEUF25 bootstrap bias for one iteration."""
    i, boot_genes, save_dir, str_bias_mat = args
    boot_gw = {gene: 1.0 for gene in boot_genes}
    boot_bias = MouseSTR_AvgZ_Weighted(str_bias_mat, boot_gw)
    boot_bias.to_csv(os.path.join(save_dir, f"LOEUF25.GeneWeight.boot{i}.csv"))
    return i, boot_bias

save_dir_loeuf = "../results/Bootstrap_bias/LOEUF25/Weighted_Resampling"
os.makedirs(save_dir_loeuf, exist_ok=True)

args_loeuf = [(i, genes, save_dir_loeuf, STR_BiasMat) for i, genes in enumerate(LOEUF25_boot_gene_sets)]

boot_bias_LOEUF25 = [None] * N_BOOT
print(f"Computing LOEUF25 bootstrap bias ({N_WORKERS} workers)...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_LOEUF25_bootstrap_iter, a): a[0] for a in args_loeuf}
    done = 0
    for future in as_completed(futures):
        i, bias = future.result()
        boot_bias_LOEUF25[i] = bias
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{N_BOOT}")
print(f"Done: {N_BOOT} LOEUF25 bootstrap iterations")

# %% [markdown]
# # Section 4: Residual CI
#
# For each bootstrap, merge the bootstrap bias with observed ASD bias,
# fit linear model, and collect residuals. Then compute 95% CI across bootstraps.

# %%
# Load observed ASD bias
ASD_STR_Bias_obs = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")


def compute_residual_ci(boot_bias_list, asd_bias, suffixes, ci_pct=95):
    """Compute bootstrap residual CI for a comparison vs ASD.

    For each bootstrap bias, merge with ASD, fit linear model, collect residuals.
    Then compute per-structure CI across all bootstraps.

    Parameters
    ----------
    boot_bias_list : list of DataFrame
        Bootstrap bias DataFrames (one per replicate)
    asd_bias : DataFrame
        Observed ASD bias (index=structure names, has 'EFFECT' column)
    suffixes : tuple of str
        Suffixes for merge (ASD, comparison)
    ci_pct : float
        Confidence interval percentile (default 95)

    Returns
    -------
    DataFrame with columns: mean, median, ci_lower, ci_upper, std
    """
    from sklearn.linear_model import LinearRegression

    all_residuals = {}
    for i, boot_bias in enumerate(boot_bias_list):
        merged = asd_bias[["EFFECT"]].join(boot_bias[["EFFECT"]], lsuffix=suffixes[0], rsuffix=suffixes[1])
        col_x = f"EFFECT{suffixes[1]}"
        col_y = f"EFFECT{suffixes[0]}"
        valid = merged[[col_x, col_y]].dropna()
        if len(valid) < 3:
            continue
        X = valid[col_x].values.reshape(-1, 1)
        y = valid[col_y].values
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        for s, r in zip(valid.index, residuals):
            if s not in all_residuals:
                all_residuals[s] = []
            all_residuals[s].append(r)

    tail = (100 - ci_pct) / 2
    records = []
    for s, res_list in all_residuals.items():
        arr = np.array(res_list)
        records.append({
            "Structure": s,
            "mean": arr.mean(),
            "median": np.median(arr),
            "ci_lower": np.percentile(arr, tail),
            "ci_upper": np.percentile(arr, 100 - tail),
            "std": arr.std(),
        })
    ci_df = pd.DataFrame(records).set_index("Structure")
    ci_df = ci_df.sort_values("mean", ascending=False)
    return ci_df


# %%
# DDD Residual CI
print("Computing DDD residual CI...")
DDD_residual_ci = compute_residual_ci(boot_bias_DDD, ASD_STR_Bias_obs,
                                       suffixes=("_ASD", "_DDD"))
save_dir_ddd_ci = "../results/Bootstrap_bias/DDD_ExomeWide/Residual_CI"
os.makedirs(save_dir_ddd_ci, exist_ok=True)
DDD_residual_ci.to_csv(os.path.join(save_dir_ddd_ci, "DDD_ExomeWide.Residual_CI_95.csv"))
print(f"Saved DDD Residual CI: {len(DDD_residual_ci)} structures")
DDD_residual_ci.head()

# %%
# LOEUF25 Residual CI
print("Computing LOEUF25 residual CI...")
LOEUF25_residual_ci = compute_residual_ci(boot_bias_LOEUF25, ASD_STR_Bias_obs,
                                           suffixes=("_ASD", "_LOEUF25"))
save_dir_loeuf_ci = "../results/Bootstrap_bias/LOEUF25/Residual_CI"
os.makedirs(save_dir_loeuf_ci, exist_ok=True)
LOEUF25_residual_ci.to_csv(os.path.join(save_dir_loeuf_ci, "LOEUF25.Residual_CI_95.csv"))
print(f"Saved LOEUF25 Residual CI: {len(LOEUF25_residual_ci)} structures")
LOEUF25_residual_ci.head()

# %% [markdown]
# # Summary
#
# Output files:
# - `results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling/` — 1000 ASD weighted bootstrap bias CSVs
# - `results/Bootstrap_bias/Spark_ExomeWide/Uniform_Resampling/` — 1000 ASD uniform bootstrap bias CSVs
# - `results/Bootstrap_bias/DDD_ExomeWide/Weighted_Resampling/` — 1000 DDD weighted bootstrap bias CSVs
# - `results/Bootstrap_bias/DDD_ExomeWide/Residual_CI/DDD_ExomeWide.Residual_CI_95.csv`
# - `results/Bootstrap_bias/LOEUF25/Weighted_Resampling/` — 1000 LOEUF25 bootstrap bias CSVs
# - `results/Bootstrap_bias/LOEUF25/Residual_CI/LOEUF25.Residual_CI_95.csv`
# - `dat/Genetics/GeneWeights/DDD.top285.gw` — DDD gene weights (285 genes)
# - `dat/Genetics/GeneWeights/DDD.top237.ExcludeASD.gw` — DDD gene weights (237 genes, ASD excluded)
