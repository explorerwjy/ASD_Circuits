#!/usr/bin/env python3
"""
Validate Z2 matrix and ASD bias comparison between Jon's raw expression matrix
and our rebuilt raw expression matrix.

Jon's matrix has 17,208 genes; ours has 17,133 genes (75 net difference).
- 93 genes in Jon's but not ours (due to older HOM mapping + transitive closure)
- 18 genes in ours but not Jon's
- 17,115 genes in common

Since downstream Z2 computation intersects genes (via expression matching), we
compare the FINAL Z2 matrices and ASD bias results from each pipeline:

  raw → log2(1+x) → QN → Z1 (per-gene z-score) → Z2 (expression-matched z-score)

Pipeline steps:
  1. log2(x + 1) transform
  2. quantileNormalize_withNA()
  3. Z1: per-gene z-score across structures (ZscoreConverting)
  4. Expression features (root expression → quantile ranks)
  5. Z2: expression-matched z-score via script_compute_Z2.py (using legacy match files)

Then compare:
  - Z2 value correlation (on common genes x structures)
  - ASD bias (weighted average using 61-gene weights)
  - CCS profile (circuit connectivity score at each circuit size)

Output: notebook_validation/z2_comparison_results.csv + plots

Usage:
    conda activate gencic
    python scripts/validate_z2_comparison.py
"""

import os
import sys
import time
import yaml
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

# Project setup
ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
os.chdir(ProjDIR)

from ASD_Circuits import (
    quantileNormalize_withNA,
    ZscoreConverting,
    MouseSTR_AvgZ_Weighted,
    Fil2Dict,
    ScoreCircuit_SI_Joint,
)

# Load config
with open(os.path.join(ProjDIR, "config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Directories
BIAS_DIR = os.path.join(ProjDIR, "dat/BiasMatrices")
ALLEN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp")
MATCH_DIR = os.path.join(ProjDIR, config["data_files"]["legacy_match_dir"])
OUTPUT_DIR = os.path.join(ProjDIR, "notebook_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache directory for intermediate results
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache_z2_comparison")
os.makedirs(CACHE_DIR, exist_ok=True)


###############################################################################
# Helper: full Z2 pipeline (raw → log2 → QN → Z1 → exp features → Z2)
###############################################################################

def run_z2_pipeline(raw_mat, label, n_jobs=10):
    """
    Run the full Z2 pipeline on a raw expression matrix.

    Parameters
    ----------
    raw_mat : pd.DataFrame
        Raw expression matrix (genes x 213 structures), index = Entrez IDs
    label : str
        Label for cache files (e.g., "jon" or "ours")
    n_jobs : int
        Number of parallel processes for Z2 computation

    Returns
    -------
    z2_mat : pd.DataFrame
        Z2 matrix (genes x structures)
    z1_mat : pd.DataFrame
        Z1 matrix (genes x structures)
    qn_mat : pd.DataFrame
        QN matrix (genes x structures)
    """
    z2_cache = os.path.join(CACHE_DIR, f"Z2_{label}.parquet")
    z1_cache = os.path.join(CACHE_DIR, f"Z1_{label}.parquet")
    qn_cache = os.path.join(CACHE_DIR, f"QN_{label}.parquet")
    expfeat_cache = os.path.join(CACHE_DIR, f"ExpFeatures_{label}.csv")

    # Check if all cached
    if all(os.path.exists(p) for p in [z2_cache, z1_cache, qn_cache]):
        print(f"[{label}] Loading cached results...")
        z2_mat = pd.read_parquet(z2_cache)
        z1_mat = pd.read_parquet(z1_cache)
        qn_mat = pd.read_parquet(qn_cache)
        print(f"  QN: {qn_mat.shape}, Z1: {z1_mat.shape}, Z2: {z2_mat.shape}")
        return z2_mat, z1_mat, qn_mat

    print(f"\n{'='*70}")
    print(f"[{label}] Running full Z2 pipeline on {raw_mat.shape[0]} genes x {raw_mat.shape[1]} structures")
    print(f"{'='*70}")

    # Step 1: log2(x + 1)
    t0 = time.time()
    log2_mat = np.log2(raw_mat + 1)
    print(f"[{label}] Step 1 - log2(x+1): {log2_mat.shape}")
    print(f"  Range: [{log2_mat.min().min():.4f}, {log2_mat.max().max():.4f}]")
    print(f"  NaN fraction: {log2_mat.isna().sum().sum() / log2_mat.size:.4f}")

    # Step 2: Quantile normalization
    print(f"[{label}] Step 2 - Quantile normalization...")
    qn_mat = quantileNormalize_withNA(log2_mat)
    qn_mat.to_parquet(qn_cache)
    print(f"  QN shape: {qn_mat.shape}")
    print(f"  Range: [{np.nanmin(qn_mat.values):.4f}, {np.nanmax(qn_mat.values):.4f}]")
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    # Step 3: Z1 (per-gene z-score)
    print(f"[{label}] Step 3 - Z1 conversion...")
    z1_data = []
    z1_genes = []
    for gene in qn_mat.index:
        z1 = ZscoreConverting(qn_mat.loc[gene].values)
        if not np.all(np.isnan(z1)):
            z1_data.append(z1)
            z1_genes.append(gene)

    z1_mat = pd.DataFrame(
        data=np.array(z1_data),
        index=z1_genes,
        columns=qn_mat.columns
    )
    z1_mat.index.name = None
    z1_mat.to_parquet(z1_cache)
    print(f"  Z1 shape: {z1_mat.shape} ({raw_mat.shape[0] - z1_mat.shape[0]} genes dropped, all-NaN)")
    print(f"  Range: [{z1_mat.min().min():.3f}, {z1_mat.max().max():.3f}]")

    # Step 4: Expression features for Z2 matching
    print(f"[{label}] Step 4 - Expression features...")
    root_exp = qn_mat.loc[z1_mat.index].mean(axis=1, skipna=True)
    exp_features = pd.DataFrame({"Genes": z1_mat.index, "EXP": root_exp.values})
    exp_features = exp_features.dropna(subset=["EXP"]).reset_index(drop=True)
    exp_features = exp_features.sort_values("EXP", ascending=True).reset_index(drop=True)
    exp_features["Rank"] = exp_features.index + 1
    exp_features["quantile"] = exp_features["Rank"] / len(exp_features)
    exp_features = exp_features.set_index("Genes")
    exp_features.to_csv(expfeat_cache)
    print(f"  {exp_features.shape[0]} genes with expression features")

    # Step 5: Z2 via external script
    print(f"[{label}] Step 5 - Z2 computation (match_dir={MATCH_DIR})...")
    cmd = [
        sys.executable,
        os.path.join(ProjDIR, "scripts/script_compute_Z2.py"),
        "--z1", z1_cache,
        "--exp-features", expfeat_cache,
        "--output", z2_cache,
        "--n-jobs", str(n_jobs),
        "--match-dir", MATCH_DIR,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Z2 computation failed for {label}")

    z2_mat = pd.read_parquet(z2_cache)
    total_elapsed = time.time() - t0
    print(f"  Z2 shape: {z2_mat.shape}")
    print(f"  NaN count: {z2_mat.isna().sum().sum()}")
    print(f"  Range: [{z2_mat.min().min():.4f}, {z2_mat.max().max():.4f}]")
    print(f"  Total pipeline: {total_elapsed:.1f}s")

    return z2_mat, z1_mat, qn_mat


###############################################################################
# CCS profile computation
###############################################################################

def compute_ccs_profile(bias_df, info_mat, sizes=None):
    """
    Compute CCS profile: for each size N, pick top-N structures by bias rank
    and score their connectivity.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Bias results with EFFECT and Rank columns, index = structure names
    info_mat : pd.DataFrame
        InfoMat for connectivity scoring (structures x structures)
    sizes : list of int, optional
        Circuit sizes to evaluate (default: 2 to 80)

    Returns
    -------
    pd.DataFrame with columns: Size, CCS
    """
    if sizes is None:
        sizes = list(range(2, 81))

    # Get structures sorted by bias (highest first)
    bias_sorted = bias_df.sort_values("EFFECT", ascending=False)
    all_strs = bias_sorted.index.values

    # Filter to structures present in InfoMat
    valid_strs = [s for s in all_strs if s in info_mat.index]

    results = []
    for n in sizes:
        if n > len(valid_strs):
            results.append({"Size": n, "CCS": np.nan})
            continue
        top_n = valid_strs[:n]
        ccs = ScoreCircuit_SI_Joint(top_n, info_mat)
        results.append({"Size": n, "CCS": ccs})

    return pd.DataFrame(results)


###############################################################################
# Main comparison
###############################################################################

def main():
    print("=" * 70)
    print("Z2 COMPARISON: Jon's raw matrix vs Our rebuilt raw matrix")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load raw matrices
    # -------------------------------------------------------------------------
    print("\n--- Loading raw matrices ---")

    jon_raw_path = os.path.join(ProjDIR, config["data_files"]["jon_exp_raw"])
    jon_raw = pd.read_csv(jon_raw_path, index_col="ROW")
    print(f"Jon's raw:  {jon_raw.shape} ({jon_raw.index.name})")

    ours_raw = pd.read_parquet(os.path.join(ProjDIR, "dat/allen-mouse-exp/ExpressionMatrix_raw.parquet"))
    print(f"Our raw:    {ours_raw.shape} ({ours_raw.index.name})")

    # Gene overlap
    jon_genes = set(jon_raw.index)
    ours_genes = set(ours_raw.index)
    common_genes = jon_genes & ours_genes
    jon_only = jon_genes - ours_genes
    ours_only = ours_genes - jon_genes
    print(f"\nGene overlap:")
    print(f"  Common:     {len(common_genes)}")
    print(f"  Jon only:   {len(jon_only)} (older HOM mapping + transitive closure)")
    print(f"  Ours only:  {len(ours_only)}")

    # Compare raw values on common genes
    jon_common = jon_raw.loc[sorted(common_genes)].sort_index()
    ours_common = ours_raw.loc[sorted(common_genes)].sort_index()
    # Align columns
    shared_cols = sorted(set(jon_common.columns) & set(ours_common.columns))
    jon_common = jon_common[shared_cols]
    ours_common = ours_common[shared_cols]

    # Flatten for correlation (exclude NaNs from both)
    jon_flat = jon_common.values.flatten()
    ours_flat = ours_common.values.flatten()
    valid = ~(np.isnan(jon_flat) | np.isnan(ours_flat))
    r_raw, p_raw = stats.pearsonr(jon_flat[valid], ours_flat[valid])
    max_diff = np.nanmax(np.abs(jon_flat[valid] - ours_flat[valid]))
    print(f"\nRaw values on common genes ({np.sum(valid)} values):")
    print(f"  Pearson r = {r_raw:.6f}")
    print(f"  Max absolute diff = {max_diff:.6e}")

    # -------------------------------------------------------------------------
    # Run Z2 pipeline for both
    # -------------------------------------------------------------------------
    z2_jon, z1_jon, qn_jon = run_z2_pipeline(jon_raw, "jon", n_jobs=10)
    z2_ours, z1_ours, qn_ours = run_z2_pipeline(ours_raw, "ours", n_jobs=10)

    # -------------------------------------------------------------------------
    # Also load production Z2 for reference
    # -------------------------------------------------------------------------
    z2_prod_path = os.path.join(ProjDIR, config["analysis_types"]["STR_ISH"]["expr_matrix"])
    z2_prod = pd.read_parquet(z2_prod_path)
    print(f"\nProduction Z2 (from Jon's R-QN): {z2_prod.shape}")

    # -------------------------------------------------------------------------
    # Comparison 1: Z2 value correlation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON 1: Z2 Value Correlation (common genes x structures)")
    print("=" * 70)

    common_z2_genes = sorted(set(z2_jon.index) & set(z2_ours.index))
    common_z2_cols = sorted(set(z2_jon.columns) & set(z2_ours.columns))
    print(f"Common Z2 genes: {len(common_z2_genes)}")
    print(f"Common Z2 structures: {len(common_z2_cols)}")

    z2_jon_c = z2_jon.loc[common_z2_genes, common_z2_cols]
    z2_ours_c = z2_ours.loc[common_z2_genes, common_z2_cols]

    # Overall correlation (flattened)
    j_flat = z2_jon_c.values.flatten()
    o_flat = z2_ours_c.values.flatten()
    valid = ~(np.isnan(j_flat) | np.isnan(o_flat))
    r_z2, p_z2 = stats.pearsonr(j_flat[valid], o_flat[valid])
    rho_z2, _ = stats.spearmanr(j_flat[valid], o_flat[valid])
    rmse_z2 = np.sqrt(np.mean((j_flat[valid] - o_flat[valid]) ** 2))
    max_diff_z2 = np.max(np.abs(j_flat[valid] - o_flat[valid]))
    print(f"\nOverall Z2 (flattened, {np.sum(valid)} values):")
    print(f"  Pearson r  = {r_z2:.6f}")
    print(f"  Spearman rho = {rho_z2:.6f}")
    print(f"  RMSE       = {rmse_z2:.6f}")
    print(f"  Max |diff| = {max_diff_z2:.4f}")

    # Also compare Jon's pipeline Z2 to production Z2 (should be identical
    # if same QN was used; but we're using Python QN here, production uses R QN)
    common_prod_genes = sorted(set(z2_jon.index) & set(z2_prod.index))
    z2_jon_p = z2_jon.loc[common_prod_genes, common_z2_cols]
    z2_prod_p = z2_prod.loc[common_prod_genes, common_z2_cols]
    j_flat2 = z2_jon_p.values.flatten()
    p_flat2 = z2_prod_p.values.flatten()
    valid2 = ~(np.isnan(j_flat2) | np.isnan(p_flat2))
    r_prod, _ = stats.pearsonr(j_flat2[valid2], p_flat2[valid2])
    print(f"\nJon-pipeline Z2 vs Production Z2 (R-QN based, {np.sum(valid2)} values):")
    print(f"  Pearson r = {r_prod:.6f}")
    print(f"  (This measures Python-QN vs R-QN difference)")

    # Per-structure Z2 correlation
    print(f"\nPer-structure Z2 correlation (Pearson r):")
    str_corrs = []
    for s in common_z2_cols:
        j_col = z2_jon_c[s].values
        o_col = z2_ours_c[s].values
        v = ~(np.isnan(j_col) | np.isnan(o_col))
        if np.sum(v) > 10:
            r, _ = stats.pearsonr(j_col[v], o_col[v])
            str_corrs.append({"Structure": s, "r": r, "n_valid": int(np.sum(v))})
    str_corrs_df = pd.DataFrame(str_corrs)
    print(f"  Min r = {str_corrs_df['r'].min():.6f} ({str_corrs_df.loc[str_corrs_df['r'].idxmin(), 'Structure']})")
    print(f"  Mean r = {str_corrs_df['r'].mean():.6f}")
    print(f"  Max r = {str_corrs_df['r'].max():.6f}")

    # Per-gene Z2 correlation
    print(f"\nPer-gene Z2 correlation (Pearson r):")
    gene_corrs = []
    for g in common_z2_genes:
        j_row = z2_jon_c.loc[g].values
        o_row = z2_ours_c.loc[g].values
        v = ~(np.isnan(j_row) | np.isnan(o_row))
        if np.sum(v) > 10:
            r, _ = stats.pearsonr(j_row[v], o_row[v])
            gene_corrs.append({"Gene": g, "r": r})
    gene_corrs_df = pd.DataFrame(gene_corrs)
    print(f"  Min r = {gene_corrs_df['r'].min():.6f}")
    print(f"  Mean r = {gene_corrs_df['r'].mean():.6f}")
    print(f"  Median r = {gene_corrs_df['r'].median():.6f}")
    print(f"  Genes with r < 0.99: {(gene_corrs_df['r'] < 0.99).sum()}")
    print(f"  Genes with r < 0.95: {(gene_corrs_df['r'] < 0.95).sum()}")
    print(f"  Genes with r < 0.90: {(gene_corrs_df['r'] < 0.90).sum()}")

    # -------------------------------------------------------------------------
    # Comparison 2: ASD Bias
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON 2: ASD Bias (61-gene weighted average)")
    print("=" * 70)

    # Load gene weights
    gw_path = os.path.join(ProjDIR, config["data_files"]["asd_gene_weights_v2"])
    gene_weights = Fil2Dict(gw_path)
    print(f"Gene weights: {len(gene_weights)} genes from {gw_path}")

    # Compute bias for each Z2
    bias_jon = MouseSTR_AvgZ_Weighted(z2_jon, gene_weights)
    bias_ours = MouseSTR_AvgZ_Weighted(z2_ours, gene_weights)
    bias_prod = MouseSTR_AvgZ_Weighted(z2_prod, gene_weights)

    # Count valid ASD genes in each Z2
    gw_genes = set(gene_weights.keys())
    n_jon = len(gw_genes & set(z2_jon.index))
    n_ours = len(gw_genes & set(z2_ours.index))
    n_prod = len(gw_genes & set(z2_prod.index))
    print(f"ASD genes in Jon Z2: {n_jon}")
    print(f"ASD genes in Ours Z2: {n_ours}")
    print(f"ASD genes in Prod Z2: {n_prod}")

    # Align biases on common structures
    common_strs = sorted(set(bias_jon.index) & set(bias_ours.index))
    bias_j = bias_jon.loc[common_strs]
    bias_o = bias_ours.loc[common_strs]
    bias_p = bias_prod.loc[common_strs]

    # Pearson and Spearman correlations
    r_bias, _ = stats.pearsonr(bias_j["EFFECT"], bias_o["EFFECT"])
    rho_bias, _ = stats.spearmanr(bias_j["EFFECT"], bias_o["EFFECT"])
    rmse_bias = np.sqrt(np.mean((bias_j["EFFECT"] - bias_o["EFFECT"]) ** 2))

    r_bias_jp, _ = stats.pearsonr(bias_j["EFFECT"], bias_p["EFFECT"])
    r_bias_op, _ = stats.pearsonr(bias_o["EFFECT"], bias_p["EFFECT"])

    print(f"\nBias comparison (Jon vs Ours):")
    print(f"  Pearson r    = {r_bias:.6f}")
    print(f"  Spearman rho = {rho_bias:.6f}")
    print(f"  RMSE         = {rmse_bias:.6f}")
    print(f"\nBias vs Production (R-QN based):")
    print(f"  Jon-pipeline  vs Production: r = {r_bias_jp:.6f}")
    print(f"  Ours-pipeline vs Production: r = {r_bias_op:.6f}")

    # Top-N overlap
    print(f"\nTop-N structure overlap (Jon vs Ours):")
    for n in [10, 20, 46, 50]:
        top_j = set(bias_j.nsmallest(n, "Rank").index)
        top_o = set(bias_o.nsmallest(n, "Rank").index)
        overlap = len(top_j & top_o)
        print(f"  Top {n:2d}: {overlap}/{n} overlap ({100*overlap/n:.1f}%)")

    # Rank difference for top structures
    print(f"\nTop-20 structures ranked by Jon's bias:")
    top20_j = bias_j.nsmallest(20, "Rank")
    for s in top20_j.index:
        rank_j = int(bias_j.loc[s, "Rank"])
        rank_o = int(bias_o.loc[s, "Rank"])
        rank_p = int(bias_p.loc[s, "Rank"])
        eff_j = bias_j.loc[s, "EFFECT"]
        eff_o = bias_o.loc[s, "EFFECT"]
        print(f"  {s:55s}  Jon={rank_j:3d}  Ours={rank_o:3d}  Prod={rank_p:3d}  "
              f"dRank={rank_o-rank_j:+3d}  EFFECT: {eff_j:.4f} vs {eff_o:.4f}")

    # -------------------------------------------------------------------------
    # Comparison 3: CCS Profile
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON 3: CCS Profile (Circuit Connectivity Score)")
    print("=" * 70)

    # Load InfoMat
    infomat_path = os.path.join(ProjDIR, config["data_files"]["infomat_ipsi"])
    InfoMat = pd.read_csv(infomat_path, index_col=0)
    print(f"InfoMat: {InfoMat.shape}")

    sizes = list(range(2, 81))
    ccs_jon = compute_ccs_profile(bias_jon, InfoMat, sizes)
    ccs_ours = compute_ccs_profile(bias_ours, InfoMat, sizes)
    ccs_prod = compute_ccs_profile(bias_prod, InfoMat, sizes)

    # Find peak
    peak_jon_idx = ccs_jon["CCS"].idxmax()
    peak_ours_idx = ccs_ours["CCS"].idxmax()
    peak_prod_idx = ccs_prod["CCS"].idxmax()

    peak_jon_size = int(ccs_jon.loc[peak_jon_idx, "Size"])
    peak_ours_size = int(ccs_ours.loc[peak_ours_idx, "Size"])
    peak_prod_size = int(ccs_prod.loc[peak_prod_idx, "Size"])

    peak_jon_ccs = ccs_jon.loc[peak_jon_idx, "CCS"]
    peak_ours_ccs = ccs_ours.loc[peak_ours_idx, "CCS"]
    peak_prod_ccs = ccs_prod.loc[peak_prod_idx, "CCS"]

    print(f"\nCCS peak locations:")
    print(f"  Jon pipeline:  size={peak_jon_size}, CCS={peak_jon_ccs:.6f}")
    print(f"  Ours pipeline: size={peak_ours_size}, CCS={peak_ours_ccs:.6f}")
    print(f"  Production:    size={peak_prod_size}, CCS={peak_prod_ccs:.6f}")

    # Also check local peak around size 46 (the paper's circuit size)
    ccs_at_46_jon = ccs_jon.loc[ccs_jon["Size"] == 46, "CCS"].values[0]
    ccs_at_46_ours = ccs_ours.loc[ccs_ours["Size"] == 46, "CCS"].values[0]
    ccs_at_46_prod = ccs_prod.loc[ccs_prod["Size"] == 46, "CCS"].values[0]
    print(f"\nCCS at size 46 (paper's circuit size):")
    print(f"  Jon pipeline:  {ccs_at_46_jon:.6f}")
    print(f"  Ours pipeline: {ccs_at_46_ours:.6f}")
    print(f"  Production:    {ccs_at_46_prod:.6f}")

    # CCS profile correlation (all sizes)
    valid_ccs = ~(ccs_jon["CCS"].isna() | ccs_ours["CCS"].isna())
    r_ccs, _ = stats.pearsonr(ccs_jon.loc[valid_ccs, "CCS"], ccs_ours.loc[valid_ccs, "CCS"])
    print(f"\nCCS profile correlation (all sizes 2-80):")
    print(f"  Jon vs Ours: r = {r_ccs:.6f}")

    r_ccs_jp, _ = stats.pearsonr(
        ccs_jon.loc[valid_ccs, "CCS"],
        ccs_prod.loc[valid_ccs, "CCS"]
    )
    r_ccs_op, _ = stats.pearsonr(
        ccs_ours.loc[valid_ccs, "CCS"],
        ccs_prod.loc[valid_ccs, "CCS"]
    )
    print(f"  Jon vs Prod:  r = {r_ccs_jp:.6f}")
    print(f"  Ours vs Prod: r = {r_ccs_op:.6f}")

    # CCS profile correlation excluding small sizes (>= 10)
    # Small circuit sizes are noisy because a single rank swap changes the set
    mask_ge10 = (ccs_jon["Size"] >= 10) & valid_ccs
    r_ccs_10, _ = stats.pearsonr(ccs_jon.loc[mask_ge10, "CCS"], ccs_ours.loc[mask_ge10, "CCS"])
    r_ccs_10_jp, _ = stats.pearsonr(ccs_jon.loc[mask_ge10, "CCS"], ccs_prod.loc[mask_ge10, "CCS"])
    r_ccs_10_op, _ = stats.pearsonr(ccs_ours.loc[mask_ge10, "CCS"], ccs_prod.loc[mask_ge10, "CCS"])
    print(f"\nCCS profile correlation (sizes >= 10, more stable):")
    print(f"  Jon vs Ours: r = {r_ccs_10:.6f}")
    print(f"  Jon vs Prod:  r = {r_ccs_10_jp:.6f}")
    print(f"  Ours vs Prod: r = {r_ccs_10_op:.6f}")

    # Local peak in the size 20-60 range (the biologically relevant range)
    mask_2060 = (ccs_jon["Size"] >= 20) & (ccs_jon["Size"] <= 60)
    local_peak_jon = int(ccs_jon.loc[mask_2060].loc[ccs_jon.loc[mask_2060, "CCS"].idxmax(), "Size"])
    local_peak_ours = int(ccs_ours.loc[mask_2060].loc[ccs_ours.loc[mask_2060, "CCS"].idxmax(), "Size"])
    local_peak_prod = int(ccs_prod.loc[mask_2060].loc[ccs_prod.loc[mask_2060, "CCS"].idxmax(), "Size"])
    print(f"\nLocal peak in size 20-60 range:")
    print(f"  Jon pipeline:  size={local_peak_jon}")
    print(f"  Ours pipeline: size={local_peak_ours}")
    print(f"  Production:    size={local_peak_prod}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Summary CSV
    summary = {
        "Metric": [
            "N_genes_jon_raw", "N_genes_ours_raw", "N_genes_common_raw",
            "N_genes_jon_only", "N_genes_ours_only",
            "Raw_value_r", "Raw_max_diff",
            "N_genes_z2_jon", "N_genes_z2_ours", "N_genes_z2_common",
            "Z2_pearson_r", "Z2_spearman_rho", "Z2_rmse", "Z2_max_diff",
            "Z2_per_structure_r_min", "Z2_per_structure_r_mean",
            "Z2_per_gene_r_min", "Z2_per_gene_r_mean", "Z2_per_gene_r_median",
            "N_genes_r_below_0.99", "N_genes_r_below_0.95",
            "Jon_Z2_vs_prod_r",
            "ASD_genes_jon", "ASD_genes_ours", "ASD_genes_prod",
            "Bias_pearson_r", "Bias_spearman_rho", "Bias_rmse",
            "Bias_jon_vs_prod_r", "Bias_ours_vs_prod_r",
            "Top10_overlap", "Top20_overlap", "Top46_overlap",
            "CCS_peak_jon", "CCS_peak_ours", "CCS_peak_prod",
            "CCS_peak_val_jon", "CCS_peak_val_ours", "CCS_peak_val_prod",
            "CCS_at_46_jon", "CCS_at_46_ours", "CCS_at_46_prod",
            "CCS_profile_r_jon_ours", "CCS_profile_r_jon_prod", "CCS_profile_r_ours_prod",
            "CCS_profile_r_ge10_jon_ours", "CCS_profile_r_ge10_jon_prod", "CCS_profile_r_ge10_ours_prod",
            "CCS_local_peak_2060_jon", "CCS_local_peak_2060_ours", "CCS_local_peak_2060_prod",
        ],
        "Value": [
            jon_raw.shape[0], ours_raw.shape[0], len(common_genes),
            len(jon_only), len(ours_only),
            r_raw, max_diff,
            z2_jon.shape[0], z2_ours.shape[0], len(common_z2_genes),
            r_z2, rho_z2, rmse_z2, max_diff_z2,
            str_corrs_df["r"].min(), str_corrs_df["r"].mean(),
            gene_corrs_df["r"].min(), gene_corrs_df["r"].mean(), gene_corrs_df["r"].median(),
            int((gene_corrs_df["r"] < 0.99).sum()), int((gene_corrs_df["r"] < 0.95).sum()),
            r_prod,
            n_jon, n_ours, n_prod,
            r_bias, rho_bias, rmse_bias,
            r_bias_jp, r_bias_op,
            len(set(bias_j.nsmallest(10, "Rank").index) & set(bias_o.nsmallest(10, "Rank").index)),
            len(set(bias_j.nsmallest(20, "Rank").index) & set(bias_o.nsmallest(20, "Rank").index)),
            len(set(bias_j.nsmallest(46, "Rank").index) & set(bias_o.nsmallest(46, "Rank").index)),
            peak_jon_size, peak_ours_size, peak_prod_size,
            peak_jon_ccs, peak_ours_ccs, peak_prod_ccs,
            ccs_at_46_jon, ccs_at_46_ours, ccs_at_46_prod,
            r_ccs, r_ccs_jp, r_ccs_op,
            r_ccs_10, r_ccs_10_jp, r_ccs_10_op,
            local_peak_jon, local_peak_ours, local_peak_prod,
        ],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, "z2_comparison_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")

    # Save CCS profiles
    ccs_combined = pd.DataFrame({
        "Size": ccs_jon["Size"],
        "CCS_jon": ccs_jon["CCS"],
        "CCS_ours": ccs_ours["CCS"],
        "CCS_prod": ccs_prod["CCS"],
    })
    ccs_path = os.path.join(OUTPUT_DIR, "z2_comparison_ccs_profiles.csv")
    ccs_combined.to_csv(ccs_path, index=False)
    print(f"CCS profiles: {ccs_path}")

    # Save bias comparison
    bias_combined = pd.DataFrame({
        "EFFECT_jon": bias_j["EFFECT"],
        "Rank_jon": bias_j["Rank"],
        "EFFECT_ours": bias_o["EFFECT"],
        "Rank_ours": bias_o["Rank"],
        "EFFECT_prod": bias_p["EFFECT"],
        "Rank_prod": bias_p["Rank"],
    })
    bias_path = os.path.join(OUTPUT_DIR, "z2_comparison_bias.csv")
    bias_combined.to_csv(bias_path)
    print(f"Bias comparison: {bias_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    print("\n--- Generating plots ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_alpha(0)

        # Panel 1: Z2 scatter (flattened, subsample for speed)
        ax = axes[0, 0]
        ax.patch.set_alpha(0)
        n_sample = min(100000, np.sum(valid))
        rng = np.random.default_rng(42)
        idx = rng.choice(np.where(valid)[0], size=n_sample, replace=False)
        ax.scatter(j_flat[idx], o_flat[idx], s=0.5, alpha=0.1, color="steelblue", rasterized=True)
        ax.plot([-5, 5], [-5, 5], "r--", linewidth=1)
        ax.set_xlabel("Z2 (Jon pipeline)")
        ax.set_ylabel("Z2 (Ours pipeline)")
        ax.set_title(f"Z2 values (r={r_z2:.4f})")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        # Panel 2: Bias scatter (Jon vs Ours)
        ax = axes[0, 1]
        ax.patch.set_alpha(0)
        ax.scatter(bias_j["EFFECT"], bias_o["EFFECT"], s=20, color="steelblue", alpha=0.7)
        ax.plot([bias_j["EFFECT"].min(), bias_j["EFFECT"].max()],
                [bias_j["EFFECT"].min(), bias_j["EFFECT"].max()], "r--", linewidth=1)
        ax.set_xlabel("Bias EFFECT (Jon pipeline)")
        ax.set_ylabel("Bias EFFECT (Ours pipeline)")
        ax.set_title(f"ASD Bias (r={r_bias:.4f}, rho={rho_bias:.4f})")

        # Panel 3: Bias scatter (Jon vs Production)
        ax = axes[0, 2]
        ax.patch.set_alpha(0)
        ax.scatter(bias_j["EFFECT"], bias_p["EFFECT"], s=20, color="darkorange", alpha=0.7)
        ax.plot([bias_j["EFFECT"].min(), bias_j["EFFECT"].max()],
                [bias_j["EFFECT"].min(), bias_j["EFFECT"].max()], "r--", linewidth=1)
        ax.set_xlabel("Bias EFFECT (Jon, Python QN)")
        ax.set_ylabel("Bias EFFECT (Production, R QN)")
        ax.set_title(f"Jon Python-QN vs Production R-QN (r={r_bias_jp:.4f})")

        # Panel 4: CCS profile (all three)
        ax = axes[1, 0]
        ax.patch.set_alpha(0)
        ax.plot(ccs_jon["Size"], ccs_jon["CCS"], "b-", linewidth=2, label=f"Jon (peak={peak_jon_size})")
        ax.plot(ccs_ours["Size"], ccs_ours["CCS"], "g--", linewidth=2, label=f"Ours (peak={peak_ours_size})")
        ax.plot(ccs_prod["Size"], ccs_prod["CCS"], "r:", linewidth=2, label=f"Prod (peak={peak_prod_size})")
        ax.axvline(46, color="gray", linestyle="--", alpha=0.5, label="Size 46")
        ax.set_xlabel("Circuit Size")
        ax.set_ylabel("CCS")
        ax.set_title("CCS Profile")
        ax.legend(fontsize=9)

        # Panel 5: Per-gene Z2 correlation histogram
        ax = axes[1, 1]
        ax.patch.set_alpha(0)
        ax.hist(gene_corrs_df["r"], bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(gene_corrs_df["r"].median(), color="red", linestyle="--",
                   label=f"median={gene_corrs_df['r'].median():.4f}")
        ax.set_xlabel("Per-gene Pearson r (Jon vs Ours Z2)")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-gene Z2 correlation ({len(gene_corrs_df)} genes)")
        ax.legend()

        # Panel 6: Bias rank difference histogram
        ax = axes[1, 2]
        ax.patch.set_alpha(0)
        rank_diff = bias_o["Rank"].loc[common_strs] - bias_j["Rank"].loc[common_strs]
        ax.hist(rank_diff, bins=30, color="darkorange", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Rank difference (Ours - Jon)")
        ax.set_ylabel("Count")
        ax.set_title(f"Bias rank difference (median={rank_diff.median():.1f})")

        plt.suptitle("Z2 Comparison: Jon's raw matrix vs Our rebuilt raw matrix", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fig_path = os.path.join(OUTPUT_DIR, "z2_comparison_plots.png")
        plt.savefig(fig_path, dpi=150, transparent=True, bbox_inches="tight")
        print(f"Plots: {fig_path}")
        plt.close()

    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Raw matrices: Jon={jon_raw.shape[0]} genes, Ours={ours_raw.shape[0]} genes, Common={len(common_genes)}")
    print(f"  93 Jon-only genes: older HOM mapping with transitive closure")
    print(f"  18 Ours-only genes: newer mapping finds additional matches")
    print(f"")
    print(f"Z2 overall correlation:  r = {r_z2:.6f}")
    print(f"ASD bias correlation:    r = {r_bias:.6f}, rho = {rho_bias:.6f}")
    print(f"Top-46 overlap:          {len(set(bias_j.nsmallest(46, 'Rank').index) & set(bias_o.nsmallest(46, 'Rank').index))}/46")
    print(f"CCS peaks:               Jon={peak_jon_size}, Ours={peak_ours_size}, Prod={peak_prod_size}")
    print(f"CCS at size 46:          Jon={ccs_at_46_jon:.6f}, Ours={ccs_at_46_ours:.6f}, Prod={ccs_at_46_prod:.6f}")
    print(f"CCS profile correlation: r = {r_ccs:.6f} (all sizes), r = {r_ccs_10:.6f} (sizes >= 10)")
    print(f"CCS local peak (20-60): Jon={local_peak_jon}, Ours={local_peak_ours}, Prod={local_peak_prod}")
    print(f"")
    print(f"Note: Both Jon and Ours pipelines use Python QN here.")
    print(f"Production Z2 uses Jon's R-generated QN (the actual pipeline).")
    print(f"The Jon-vs-Ours comparison isolates the effect of gene differences only.")
    print(f"The vs-Production comparison shows the additional effect of QN implementation.")


if __name__ == "__main__":
    main()
