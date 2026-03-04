#!/usr/bin/env python3
"""
Test whether fresh seed-42 expression matching (instead of legacy match files)
gives CCS peak at size 46 for our rebuilt expression matrix.

Compares three Z2 variants:
  A) Jon's raw + legacy match files  (= production)
  B) Ours raw + legacy match files   (previous validation)
  C) Ours raw + fresh seed-42 match  (NEW — no legacy dependency)

Reuses cached Z1/QN from the previous validation run where possible.
"""
import sys, os, time, subprocess, json
import numpy as np
import pandas as pd

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (quantileNormalize_withNA, ZscoreConverting,
                          MouseSTR_AvgZ_Weighted, ScoreCircuit_SI_Joint,
                          Fil2Dict)

import yaml
with open(os.path.join(ProjDIR, "config/config.yaml")) as f:
    config = yaml.safe_load(f)

CACHE_DIR = os.path.join(ProjDIR, "notebook_validation/cache_z2_comparison")
OUTPUT_DIR = os.path.join(ProjDIR, "notebook_validation")
MATCH_DIR = os.path.join(ProjDIR, config["data_files"]["legacy_match_dir"])

os.makedirs(CACHE_DIR, exist_ok=True)

###############################################################################
# CCS profile
###############################################################################

def compute_ccs_profile(bias_df, info_mat, sizes=None):
    if sizes is None:
        sizes = list(range(2, 81))
    bias_sorted = bias_df.sort_values("EFFECT", ascending=False)
    valid_strs = [s for s in bias_sorted.index if s in info_mat.index]
    results = []
    for n in sizes:
        if n > len(valid_strs):
            results.append({"Size": n, "CCS": np.nan})
            continue
        ccs = ScoreCircuit_SI_Joint(valid_strs[:n], info_mat)
        results.append({"Size": n, "CCS": ccs})
    return pd.DataFrame(results)


def find_local_peak(ccs_df, lo=20, hi=60):
    sub = ccs_df[(ccs_df["Size"] >= lo) & (ccs_df["Size"] <= hi)]
    idx = sub["CCS"].idxmax()
    return int(sub.loc[idx, "Size"]), float(sub.loc[idx, "CCS"])


###############################################################################
# Helpers to get Z1 + exp_features (reuse cache)
###############################################################################

def build_z1_and_features(raw_mat, label):
    """Build Z1 + exp features from raw matrix (with caching)."""
    z1_cache = os.path.join(CACHE_DIR, f"Z1_{label}.parquet")
    qn_cache = os.path.join(CACHE_DIR, f"QN_{label}.parquet")
    expfeat_cache = os.path.join(CACHE_DIR, f"ExpFeatures_{label}.csv")

    if os.path.exists(z1_cache) and os.path.exists(expfeat_cache):
        z1 = pd.read_parquet(z1_cache)
        ef = pd.read_csv(expfeat_cache, index_col="Genes")
        print(f"[{label}] Loaded cached Z1 ({z1.shape}) and ExpFeatures ({ef.shape[0]} genes)")
        return z1, ef

    print(f"[{label}] Building Z1 from raw ({raw_mat.shape})...")
    t0 = time.time()

    # log2(x+1)
    log2_mat = np.log2(raw_mat + 1)

    # QN
    print(f"[{label}]   QN...")
    qn_mat = quantileNormalize_withNA(log2_mat)
    qn_mat.to_parquet(qn_cache)

    # Z1: per-gene z-score, clip ±3
    print(f"[{label}]   Z1...")
    z1_data, z1_genes = [], []
    for gene in qn_mat.index:
        z1 = ZscoreConverting(qn_mat.loc[gene].values)
        if not np.all(np.isnan(z1)):
            z1_data.append(z1)
            z1_genes.append(gene)
    z1_mat = pd.DataFrame(np.array(z1_data), index=z1_genes, columns=qn_mat.columns)
    z1_mat.index.name = None
    z1_mat.to_parquet(z1_cache)

    # Expression features
    root_exp = qn_mat.loc[z1_mat.index].mean(axis=1, skipna=True)
    ef = pd.DataFrame({"Genes": z1_mat.index, "EXP": root_exp.values})
    ef = ef.dropna(subset=["EXP"]).reset_index(drop=True)
    ef = ef.sort_values("EXP", ascending=True).reset_index(drop=True)
    ef["Rank"] = ef.index + 1
    ef["quantile"] = ef["Rank"] / len(ef)
    ef = ef.set_index("Genes")
    ef.to_csv(expfeat_cache)

    elapsed = time.time() - t0
    print(f"[{label}]   Done in {elapsed:.1f}s. Z1: {z1_mat.shape}, ExpFeat: {ef.shape[0]} genes")
    return z1_mat, ef


def compute_z2(z1_cache, expfeat_cache, z2_out, match_dir=None, seed=42, n_jobs=10):
    """Run script_compute_Z2.py with given parameters."""
    cmd = [
        sys.executable,
        os.path.join(ProjDIR, "scripts/script_compute_Z2.py"),
        "--z1", z1_cache,
        "--exp-features", expfeat_cache,
        "--output", z2_out,
        "--n-jobs", str(n_jobs),
        "--seed", str(seed),
    ]
    if match_dir:
        cmd.extend(["--match-dir", match_dir])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Z2 computation failed")
    print(f"  Elapsed: {elapsed:.1f}s")
    return pd.read_parquet(z2_out)


###############################################################################
# Main
###############################################################################

def main():
    print("=" * 70)
    print("Z2 SEED-42 MATCHING TEST")
    print("Does fresh seed-42 expression matching give CCS peak at 46?")
    print("=" * 70)

    # Load connectivity
    info_path = os.path.join(ProjDIR, config["data_files"]["infomat_ipsi"])
    InfoMat = pd.read_csv(info_path, index_col=0)
    print(f"InfoMat: {InfoMat.shape}")

    # Load ASD gene weights
    gw_path = os.path.join(ProjDIR, config["data_files"]["asd_gene_weights_v2"])
    ASD_Weights = Fil2Dict(gw_path)
    print(f"ASD gene weights: {len(ASD_Weights)} genes")

    # =========================================================================
    # A) Production Z2 (Jon's raw + legacy match) — just load
    # =========================================================================
    print("\n--- A) Production Z2 (Jon + legacy match) ---")
    z2_prod = pd.read_parquet(os.path.join(ProjDIR, config["analysis_types"]["STR_ISH"]["expr_matrix"]))
    print(f"Production Z2: {z2_prod.shape}")
    bias_prod = MouseSTR_AvgZ_Weighted(z2_prod, ASD_Weights)
    ccs_prod = compute_ccs_profile(bias_prod, InfoMat)
    peak_prod = find_local_peak(ccs_prod)
    print(f"CCS peak (20-60): size {peak_prod[0]}, CCS={peak_prod[1]:.4f}")

    # =========================================================================
    # Load raw matrices and build Z1
    # =========================================================================
    jon_raw_path = os.path.join(ProjDIR, config["data_files"]["jon_exp_raw"])
    jon_raw = pd.read_csv(jon_raw_path, index_col="ROW")
    ours_raw = pd.read_parquet(os.path.join(ProjDIR, "dat/allen-mouse-exp/ExpressionMatrix_raw.parquet"))
    print(f"\nJon raw: {jon_raw.shape}, Ours raw: {ours_raw.shape}")

    # Build Z1 + exp features for both (cached)
    z1_ours, ef_ours = build_z1_and_features(ours_raw, "Ours")
    z1_jon, ef_jon = build_z1_and_features(jon_raw, "Jon")

    # =========================================================================
    # B) Ours + legacy match
    # =========================================================================
    print("\n--- B) Ours + legacy match ---")
    z2_ours_legacy_path = os.path.join(CACHE_DIR, "Z2_Ours_legacy.parquet")
    z2_ours_legacy = compute_z2(
        z1_cache=os.path.join(CACHE_DIR, "Z1_Ours.parquet"),
        expfeat_cache=os.path.join(CACHE_DIR, "ExpFeatures_Ours.csv"),
        z2_out=z2_ours_legacy_path,
        match_dir=MATCH_DIR,
        n_jobs=10,
    )
    print(f"Ours Z2 (legacy): {z2_ours_legacy.shape}")
    bias_ours_legacy = MouseSTR_AvgZ_Weighted(z2_ours_legacy, ASD_Weights)
    ccs_ours_legacy = compute_ccs_profile(bias_ours_legacy, InfoMat)
    peak_ours_legacy = find_local_peak(ccs_ours_legacy)
    print(f"CCS peak (20-60): size {peak_ours_legacy[0]}, CCS={peak_ours_legacy[1]:.4f}")

    # =========================================================================
    # C) Ours + fresh seed-42 match (the main test)
    # =========================================================================
    print("\n--- C) Ours + fresh seed-42 match ---")
    z2_seed42_path = os.path.join(CACHE_DIR, "Z2_Ours_seed42.parquet")

    z2_seed42 = compute_z2(
        z1_cache=os.path.join(CACHE_DIR, "Z1_Ours.parquet"),
        expfeat_cache=os.path.join(CACHE_DIR, "ExpFeatures_Ours.csv"),
        z2_out=z2_seed42_path,
        match_dir=None,  # <-- fresh matching, no legacy files
        seed=42,
        n_jobs=10,
    )
    print(f"Ours Z2 (seed-42): {z2_seed42.shape}")

    bias_seed42 = MouseSTR_AvgZ_Weighted(z2_seed42, ASD_Weights)
    ccs_seed42 = compute_ccs_profile(bias_seed42, InfoMat)
    peak_seed42 = find_local_peak(ccs_seed42)
    print(f"CCS peak (20-60): size {peak_seed42[0]}, CCS={peak_seed42[1]:.4f}")

    # =========================================================================
    # D) Also try seed-42 on Jon's raw (for completeness)
    # =========================================================================
    print("\n--- D) Jon's raw + fresh seed-42 match ---")
    z2_jon_seed42_path = os.path.join(CACHE_DIR, "Z2_Jon_seed42.parquet")

    z2_jon_seed42 = compute_z2(
        z1_cache=os.path.join(CACHE_DIR, "Z1_Jon.parquet"),
        expfeat_cache=os.path.join(CACHE_DIR, "ExpFeatures_Jon.csv"),
        z2_out=z2_jon_seed42_path,
        match_dir=None,
        seed=42,
        n_jobs=10,
    )
    print(f"Jon Z2 (seed-42): {z2_jon_seed42.shape}")

    bias_jon_seed42 = MouseSTR_AvgZ_Weighted(z2_jon_seed42, ASD_Weights)
    ccs_jon_seed42 = compute_ccs_profile(bias_jon_seed42, InfoMat)
    peak_jon_seed42 = find_local_peak(ccs_jon_seed42)
    print(f"CCS peak (20-60): size {peak_jon_seed42[0]}, CCS={peak_jon_seed42[1]:.4f}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CCS PEAK COMPARISON")
    print("=" * 70)
    print(f"{'Pipeline':<40} {'Peak Size':>10} {'CCS':>10}")
    print("-" * 62)
    print(f"{'A) Production (Jon + legacy match)':<40} {peak_prod[0]:>10d} {peak_prod[1]:>10.4f}")
    print(f"{'B) Ours + legacy match':<40} {peak_ours_legacy[0]:>10d} {peak_ours_legacy[1]:>10.4f}")
    print(f"{'C) Ours + seed-42 match':<40} {peak_seed42[0]:>10d} {peak_seed42[1]:>10.4f}")
    print(f"{'D) Jon + seed-42 match':<40} {peak_jon_seed42[0]:>10d} {peak_jon_seed42[1]:>10.4f}")

    # =========================================================================
    # Bias correlation: seed-42 vs production
    # =========================================================================
    print("\n--- Bias correlation (EFFECT) ---")
    common = bias_prod.index.intersection(bias_seed42.index)
    r_C = np.corrcoef(bias_prod.loc[common, "EFFECT"], bias_seed42.loc[common, "EFFECT"])[0, 1]
    r_D = np.corrcoef(bias_prod.loc[common, "EFFECT"], bias_jon_seed42.loc[common, "EFFECT"])[0, 1]
    print(f"Production vs C (Ours+seed42):  r = {r_C:.6f}")
    print(f"Production vs D (Jon+seed42):   r = {r_D:.6f}")

    # Top-46 overlap
    top_prod = set(bias_prod.sort_values("EFFECT", ascending=False).head(46).index)
    top_C = set(bias_seed42.sort_values("EFFECT", ascending=False).head(46).index)
    top_D = set(bias_jon_seed42.sort_values("EFFECT", ascending=False).head(46).index)
    print(f"\nTop-46 overlap:")
    print(f"  Production vs C: {len(top_prod & top_C)}/46")
    print(f"  Production vs D: {len(top_prod & top_D)}/46")
    print(f"  C vs D:          {len(top_C & top_D)}/46")

    # =========================================================================
    # Z2 correlation: seed-42 vs legacy match (how much does matching matter?)
    # =========================================================================
    common_g = z2_ours_legacy.index.intersection(z2_seed42.index)
    common_c = z2_ours_legacy.columns.intersection(z2_seed42.columns)
    v1 = z2_ours_legacy.loc[common_g, common_c].values.flatten()
    v2 = z2_seed42.loc[common_g, common_c].values.flatten()
    valid = ~np.isnan(v1) & ~np.isnan(v2)
    r_match = np.corrcoef(v1[valid], v2[valid])[0, 1]
    print(f"\nZ2 correlation (same raw, legacy vs seed-42 match): r = {r_match:.6f}")

    # =========================================================================
    # Save CCS profiles
    # =========================================================================
    ccs_all = pd.DataFrame({
        "Size": ccs_prod["Size"],
        "Production": ccs_prod["CCS"],
        "Ours_seed42": ccs_seed42["CCS"],
        "Jon_seed42": ccs_jon_seed42["CCS"],
    })
    ccs_all["Ours_legacy"] = ccs_ours_legacy["CCS"]

    out_csv = os.path.join(OUTPUT_DIR, "z2_seed42_ccs_profiles.csv")
    ccs_all.to_csv(out_csv, index=False)
    print(f"\nSaved CCS profiles: {out_csv}")

    # =========================================================================
    # Multiple seeds test — does the peak location vary?
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEED SENSITIVITY TEST (Ours raw, seeds 0-9)")
    print("=" * 70)
    seed_peaks = []
    for seed in range(10):
        z2_path = os.path.join(CACHE_DIR, f"Z2_Ours_seed{seed}.parquet")
        z2_s = compute_z2(
            z1_cache=os.path.join(CACHE_DIR, "Z1_Ours.parquet"),
            expfeat_cache=os.path.join(CACHE_DIR, "ExpFeatures_Ours.csv"),
            z2_out=z2_path,
            match_dir=None,
            seed=seed,
            n_jobs=10,
        )
        bias_s = MouseSTR_AvgZ_Weighted(z2_s, ASD_Weights)
        ccs_s = compute_ccs_profile(bias_s, InfoMat)
        pk = find_local_peak(ccs_s)
        seed_peaks.append({"seed": seed, "peak_size": pk[0], "peak_ccs": pk[1]})
        print(f"  Seed {seed}: peak at {pk[0]}, CCS={pk[1]:.4f}")

    pk_df = pd.DataFrame(seed_peaks)
    print(f"\nPeak sizes across seeds: {sorted(pk_df['peak_size'].unique())}")
    print(f"Most common peak: {pk_df['peak_size'].mode().values[0]}")
    print(f"Peak at 46: {(pk_df['peak_size'] == 46).sum()}/10 seeds")
    print(f"Peak at 47: {(pk_df['peak_size'] == 47).sum()}/10 seeds")

    pk_df.to_csv(os.path.join(OUTPUT_DIR, "z2_seed_sensitivity.csv"), index=False)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'z2_seed_sensitivity.csv')}")


if __name__ == "__main__":
    main()
