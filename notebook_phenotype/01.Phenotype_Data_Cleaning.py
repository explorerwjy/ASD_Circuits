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
import re
import yaml
import numpy as np
import pandas as pd
import pickle as pk

ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, Filt_LGD_Mis, Mut2GeneDF,
)

os.chdir(os.path.join(ProjDIR, "notebook_phenotype"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 01. Phenotype Data Cleaning
#
# Build a unified **mutation-phenotype master table** linking de novo mutations
# (from Zhou et al. 2022 discovery + replication cohorts) to clinical phenotype
# measures from SPARK and SSC datasets.
#
# **Outputs:**
# - `results/phenotype/mutation_phenotype_master.parquet` -- one row per subject
# - `results/phenotype/subject_gene_weights.pkl` -- per-subject gene weight dicts
#
# **Phenotype domains:**
# - RBS-R (repetitive behavior) -- 6 subscales + overall
# - DCDQ (motor coordination) -- 3 subscales + total
# - Vineland (adaptive behavior) -- ABC + 4 domain standards
# - SRS (social responsiveness) -- total + subscale T-scores
# - IQ (cognitive) -- FSIQ, VIQ, NVIQ
# - Developmental milestones (motor, language)

# %% [markdown]
# ## 1. Load and Filter Mutations

# %%
# Load de novo variant files
discov = pd.read_csv(
    os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Discov_DNVs.txt"),
    sep="\t", low_memory=False,
)
rep = pd.read_csv(
    os.path.join(ProjDIR, "dat/Genetics/SPARK/ASD_Rep_DNVs.txt"),
    sep="\t", low_memory=False,
)

print(f"Discovery mutations: {len(discov):,}")
print(f"Replication mutations: {len(rep):,}")

# %%
# Tag cohort -- Rep file has no Cohort column, default to SPARK (all SP IDs)
if "Cohort" not in rep.columns:
    rep["Cohort"] = "SPARK"

# Ensure DNASource column exists in rep (not always present)
if "DNASource" not in rep.columns:
    rep["DNASource"] = "."

# Combine discovery + replication
mut_all = pd.concat([discov, rep], ignore_index=True)
print(f"Combined mutations: {len(mut_all):,}")
print()
print("Mutations by cohort:")
print(mut_all["Cohort"].value_counts().to_string())

# %%
# Load exome-significant genes from Zhou et al. 2022 Table S7
table_s7 = pd.read_excel(
    os.path.join(ProjDIR, "dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx"),
    sheet_name="Table S7", skiprows=2,
)
table_s7["pDenovoWEST_Meta"] = pd.to_numeric(table_s7["pDenovoWEST_Meta"], errors="coerce")
hc_genes = table_s7.loc[
    table_s7["pDenovoWEST_Meta"] <= 1.3e-6, "HGNC"
].values
hc_gene_set = set(hc_genes)

print(f"Exome-wide significant (HC) genes: {len(hc_gene_set)}")

# %%
# Filter to HC genes only
mut_hc = mut_all[mut_all["HGNC"].isin(hc_gene_set)].copy()
print(f"Mutations in HC genes (before LGD/Dmis filter): {len(mut_hc):,}")

# Filter to LGD + damaging missense (REVEL > 0.5)
mut_filt = Filt_LGD_Mis(mut_hc, Dmis=True)
print(f"Mutations after LGD/Dmis filter: {len(mut_filt):,}")

# %%
# Breakdown by cohort
print("Filtered mutations by cohort:")
print(mut_filt["Cohort"].value_counts().to_string())
print()
n_subjects = mut_filt["IID"].nunique()
print(f"Unique subjects with HC LGD/Dmis mutations: {n_subjects}")

# Identify SPARK and SSC subjects
spark_iids = set(mut_filt.loc[mut_filt["IID"].str.startswith("SP"), "IID"])
ssc_iids = set(
    mut_filt.loc[mut_filt["IID"].str.match(r"^\d+\.p\d+$"), "IID"]
)
other_iids = set(mut_filt["IID"]) - spark_iids - ssc_iids

print(f"  SPARK subjects: {len(spark_iids)}")
print(f"  SSC subjects: {len(ssc_iids)}")
print(f"  Other (MSSNG/ASC without phenotype): {len(other_iids)}")

# %% [markdown]
# ## 2. Link SPARK Subjects to Phenotype Data

# %%
SPARK_PHENO_DIR = os.path.join(
    ProjDIR, "dat/Phenotype/SPARKDataRelease_2025-07-14"
)

# Gene annotations for later
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
# --- RBS-R ---
rbsr_raw = pd.read_csv(os.path.join(SPARK_PHENO_DIR, "rbsr-2025-07-14.csv"))
# Exclude invalid responses
rbsr_raw = rbsr_raw[rbsr_raw["rbsr_validity_flag"] != 1]
# Keep latest eval_year per subject
rbsr_raw = rbsr_raw.sort_values("eval_year", ascending=False)
rbsr = rbsr_raw.drop_duplicates(subset="subject_sp_id", keep="first")
rbsr = rbsr[rbsr["subject_sp_id"].isin(spark_iids)]

rbsr_cols = {
    "subject_sp_id": "IID",
    "i_stereotyped_behavior_score": "rbsr_stereotyped",
    "ii_self_injurious_score": "rbsr_selfinjury",
    "iii_compulsive_behavior_score": "rbsr_compulsive",
    "iv_ritualistic_behavior_score": "rbsr_ritualistic",
    "v_sameness_behavior_score": "rbsr_sameness",
    "vi_restricted_behavior_score": "rbsr_restricted",
    "overall_score": "rbsr_total",
}
spark_rbsr = rbsr[list(rbsr_cols.keys())].rename(columns=rbsr_cols)
print(f"SPARK RBS-R: {len(spark_rbsr)} subjects (of {len(spark_iids)} SPARK mutation carriers)")

# %%
# --- DCDQ ---
dcdq_raw = pd.read_csv(os.path.join(SPARK_PHENO_DIR, "dcdq-2025-07-14.csv"))
dcdq_raw = dcdq_raw[dcdq_raw["dcdq_measure_validity_flag"] != 1]
dcdq_raw = dcdq_raw.sort_values("eval_year", ascending=False)
dcdq = dcdq_raw.drop_duplicates(subset="subject_sp_id", keep="first")
dcdq = dcdq[dcdq["subject_sp_id"].isin(spark_iids)]

dcdq_cols = {
    "subject_sp_id": "IID",
    "control_during_movement": "dcdq_control",
    "fine_motor_handwriting": "dcdq_fine",
    "general_coordination": "dcdq_general",
    "total": "dcdq_total",
    "dcd": "dcdq_dcd",
}
spark_dcdq = dcdq[list(dcdq_cols.keys())].rename(columns=dcdq_cols)
print(f"SPARK DCDQ: {len(spark_dcdq)} subjects")

# %%
# --- Vineland-3 ---
vine_raw = pd.read_csv(os.path.join(SPARK_PHENO_DIR, "vineland-3-2025-07-14.csv"))
vine_raw = vine_raw.sort_values("age_at_eval_months", ascending=False)
vine = vine_raw.drop_duplicates(subset="subject_sp_id", keep="first")
vine = vine[vine["subject_sp_id"].isin(spark_iids)]

vine_cols = {
    "subject_sp_id": "IID",
    "abc_standard": "vine_abc",
    "communication_standard": "vine_comm",
    "dls_standard": "vine_dls",
    "soc_standard": "vine_social",
    "motor_standard": "vine_motor",
}
spark_vine = vine[list(vine_cols.keys())].rename(columns=vine_cols)
print(f"SPARK Vineland-3: {len(spark_vine)} subjects")

# %%
# --- SRS-2 School Age ---
srs_raw = pd.read_csv(os.path.join(SPARK_PHENO_DIR, "srs2_school_age-2025-07-14.csv"))
srs_raw = srs_raw[srs_raw["validity_flag"] != 1]
srs_raw = srs_raw.sort_values("age_at_eval_months", ascending=False)
srs = srs_raw.drop_duplicates(subset="subject_sp_id", keep="first")
srs = srs[srs["subject_sp_id"].isin(spark_iids)]

srs_cols = {
    "subject_sp_id": "IID",
    "total_t_score": "srs_total_t",
    "rrb_t_score": "srs_rrb_t",
    "awr_t_score": "srs_awr_t",
    "soc_cog_t_score": "srs_soccog_t",
    "com_t_score": "srs_com_t",
    "mot_t_score": "srs_mot_t",
    "sci_t_score": "srs_sci_t",
}
spark_srs = srs[list(srs_cols.keys())].rename(columns=srs_cols)
print(f"SPARK SRS-2: {len(spark_srs)} subjects")

# %%
# --- Core Descriptive Variables (IQ, sex, language, cognitive impairment) ---
core_raw = pd.read_csv(
    os.path.join(SPARK_PHENO_DIR, "core_descriptive_variables-2025-07-14.csv"),
    low_memory=False,
)
# Exclude invalid ASD diagnoses and known problematic family
core_raw = core_raw[core_raw["asd_validity_flag"] != 1]
core_raw = core_raw[core_raw["family_sf_id"] != "SF0006897"]
core = core_raw.drop_duplicates(subset="subject_sp_id", keep="first")
core = core[core["subject_sp_id"].isin(spark_iids)]

core_cols = {
    "subject_sp_id": "IID",
    "sex": "sex",
    "fsiq": "iq_fsiq",
    "viq": "iq_viq",
    "nviq": "iq_nviq",
    "language_level_latest": "language_level",
    "cognitive_impairment_latest": "cognitive_impairment",
}
spark_core = core[list(core_cols.keys())].rename(columns=core_cols)
print(f"SPARK core descriptive: {len(spark_core)} subjects")

# %%
# --- Background History (developmental milestones, regression) ---
bghx_raw = pd.read_csv(
    os.path.join(SPARK_PHENO_DIR, "background_history_child-2025-07-14.csv"),
    low_memory=False,
)
bghx_raw = bghx_raw[bghx_raw["bghx_validity_flag"] != 1]
bghx_raw = bghx_raw.sort_values("eval_year", ascending=False)
bghx = bghx_raw.drop_duplicates(subset="subject_sp_id", keep="first")
bghx = bghx[bghx["subject_sp_id"].isin(spark_iids)]

bghx_cols = {
    "subject_sp_id": "IID",
    "walked_age_mos": "milestone_walk_mos",
    "used_words_age_mos": "milestone_words_mos",
    "combined_phrases_age_mos": "milestone_phrases_mos",
    "sat_wo_support_age_mos": "milestone_sat_mos",
    "age_onset_mos": "asd_onset_mos",
    "regress_lang_y_n": "regress_lang",
    "regress_other_y_n": "regress_other",
}
spark_bghx = bghx[list(bghx_cols.keys())].rename(columns=bghx_cols)
print(f"SPARK background history: {len(spark_bghx)} subjects")

# %%
# Merge all SPARK phenotype data
spark_pheno = spark_core.copy()
for df in [spark_rbsr, spark_dcdq, spark_vine, spark_srs, spark_bghx]:
    spark_pheno = spark_pheno.merge(df, on="IID", how="outer")

spark_pheno["cohort"] = "SPARK"

# Replace 888 with NaN (SPARK convention for "not applicable" / "never achieved")
numeric_cols = spark_pheno.select_dtypes(include=[np.number]).columns
spark_pheno[numeric_cols] = spark_pheno[numeric_cols].replace(888, np.nan)

print(f"SPARK phenotype table: {spark_pheno.shape}")
print(f"  Subjects with any data: {spark_pheno.dropna(how='all', subset=[c for c in spark_pheno.columns if c not in ['IID', 'cohort']]).shape[0]}")

# %% [markdown]
# ## 3. Link SSC Subjects to Phenotype Data

# %%
SSC_PROBAND_DIR = os.path.join(
    ProjDIR,
    "dat/Phenotype/SSC_Phenotype_Dataset/SSC_V15_Phenotype_DATA/Proband_Data",
)

# %%
# --- RBS-R ---
ssc_rbsr_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "rbs_r.csv"))
ssc_rbsr_raw = ssc_rbsr_raw[ssc_rbsr_raw["individual"].isin(ssc_iids)]

ssc_rbsr_cols = {
    "individual": "IID",
    "i_stereotyped_behavior_score": "rbsr_stereotyped",
    "ii_self_injurious_score": "rbsr_selfinjury",
    "iii_compulsive_behavior_score": "rbsr_compulsive",
    "iv_ritualistic_behavior_score": "rbsr_ritualistic",
    "v_sameness_behavior_score": "rbsr_sameness",
    "vi_restricted_behavior_score": "rbsr_restricted",
    "overall_score": "rbsr_total",
}
ssc_rbsr = ssc_rbsr_raw[list(ssc_rbsr_cols.keys())].rename(columns=ssc_rbsr_cols)
print(f"SSC RBS-R: {len(ssc_rbsr)} subjects (of {len(ssc_iids)} SSC mutation carriers)")

# %%
# --- DCDQ ---
ssc_dcdq_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "dcdq.csv"))
ssc_dcdq_raw = ssc_dcdq_raw[ssc_dcdq_raw["individual"].isin(ssc_iids)]

# Note: SSC has a typo "general_corrdination" (double r)
ssc_dcdq_cols = {
    "individual": "IID",
    "control_during_movement": "dcdq_control",
    "fine_motor_handwriting": "dcdq_fine",
    "general_corrdination": "dcdq_general",  # SSC typo: double r
    "total": "dcdq_total",
}
ssc_dcdq = ssc_dcdq_raw[list(ssc_dcdq_cols.keys())].rename(columns=ssc_dcdq_cols)
print(f"SSC DCDQ: {len(ssc_dcdq)} subjects")

# %%
# --- Vineland-II ---
ssc_vine_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "vineland_ii.csv"))
ssc_vine_raw = ssc_vine_raw[ssc_vine_raw["individual"].isin(ssc_iids)]

ssc_vine_cols = {
    "individual": "IID",
    "composite_standard_score": "vine_abc",
    "communication_standard": "vine_comm",
    "dls_standard": "vine_dls",
    "soc_standard": "vine_social",
    "motor_skills_standard": "vine_motor",
}
ssc_vine = ssc_vine_raw[list(ssc_vine_cols.keys())].rename(columns=ssc_vine_cols)
print(f"SSC Vineland-II: {len(ssc_vine)} subjects")

# %%
# --- SRS Parent ---
ssc_srs_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "srs_parent.csv"))
ssc_srs_raw = ssc_srs_raw[ssc_srs_raw["individual"].isin(ssc_iids)]

ssc_srs_cols = {
    "individual": "IID",
    "t_score": "srs_total_t",
    "mannerisms_t_score": "srs_rrb_t",
    "awareness_t_score": "srs_awr_t",
    "cognition_t_score": "srs_soccog_t",
    "communication_t_score": "srs_com_t",
    "motivation_t_score": "srs_mot_t",
}
ssc_srs = ssc_srs_raw[list(ssc_srs_cols.keys())].rename(columns=ssc_srs_cols)
# SSC SRS-1 does not have a separate SCI composite -- leave srs_sci_t as NaN
print(f"SSC SRS Parent: {len(ssc_srs)} subjects")

# %%
# --- Core Descriptive (IQ, sex) ---
ssc_core_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "ssc_core_descriptive.csv"))
ssc_core_raw = ssc_core_raw[ssc_core_raw["individual"].isin(ssc_iids)]

ssc_core_cols = {
    "individual": "IID",
    "sex": "sex",
    "ssc_diagnosis_full_scale_iq": "iq_fsiq",
    "ssc_diagnosis_verbal_iq": "iq_viq",
    "ssc_diagnosis_nonverbal_iq": "iq_nviq",
}
ssc_core = ssc_core_raw[list(ssc_core_cols.keys())].rename(columns=ssc_core_cols)
# Capitalize sex to match SPARK convention
ssc_core["sex"] = ssc_core["sex"].str.capitalize()
print(f"SSC core descriptive: {len(ssc_core)} subjects")

# %%
# --- Background History (milestones) ---
ssc_bghx_raw = pd.read_csv(os.path.join(SSC_PROBAND_DIR, "ssc_background_hx.csv"))
ssc_bghx_raw = ssc_bghx_raw[ssc_bghx_raw["individual"].isin(ssc_iids)]

ssc_bghx_cols = {
    "individual": "IID",
    "age_walked_alone": "milestone_walk_mos",
    "age_used_words": "milestone_words_mos",
    "age_combined_words_short_sen": "milestone_phrases_mos",
    "age_sat_wo_support": "milestone_sat_mos",
}
ssc_bghx = ssc_bghx_raw[list(ssc_bghx_cols.keys())].rename(columns=ssc_bghx_cols)
print(f"SSC background history: {len(ssc_bghx)} subjects")

# %%
# Merge all SSC phenotype data
ssc_pheno = ssc_core.copy()
for df in [ssc_rbsr, ssc_dcdq, ssc_vine, ssc_srs, ssc_bghx]:
    ssc_pheno = ssc_pheno.merge(df, on="IID", how="outer")

ssc_pheno["cohort"] = "SSC"

print(f"SSC phenotype table: {ssc_pheno.shape}")
print(f"  Subjects with any data: {ssc_pheno.dropna(how='all', subset=[c for c in ssc_pheno.columns if c not in ['IID', 'cohort']]).shape[0]}")

# %% [markdown]
# ## 4. Merge, Compute Gene Weights, and Save

# %%
# Concatenate SPARK + SSC
pheno = pd.concat([spark_pheno, ssc_pheno], ignore_index=True)

# Ensure numeric columns are actually numeric (coerce strings to NaN)
score_cols = [c for c in pheno.columns if c not in ["IID", "cohort", "sex",
              "language_level", "cognitive_impairment", "regress_lang",
              "regress_other"]]
for col in score_cols:
    pheno[col] = pd.to_numeric(pheno[col], errors="coerce")

print(f"Combined phenotype table: {pheno.shape}")
print(f"  SPARK: {(pheno['cohort'] == 'SPARK').sum()}")
print(f"  SSC: {(pheno['cohort'] == 'SSC').sum()}")

# %%
# Compute per-subject gene weights using Mut2GeneDF
subject_gene_weights = {}
for iid in pheno["IID"].values:
    subj_muts = mut_filt[mut_filt["IID"] == iid]
    if len(subj_muts) == 0:
        subject_gene_weights[iid] = {}
        continue
    gw = Mut2GeneDF(subj_muts, gene_col="HGNC",
                    gene_symbol_to_entrez=GeneSymbol2Entrez)
    subject_gene_weights[iid] = gw

# Add mutation summary columns
pheno["n_mutations"] = pheno["IID"].map(
    lambda x: len(mut_filt[mut_filt["IID"] == x])
)
pheno["n_genes"] = pheno["IID"].map(
    lambda x: len(subject_gene_weights.get(x, {}))
)
pheno["total_gene_weight"] = pheno["IID"].map(
    lambda x: sum(subject_gene_weights.get(x, {}).values())
)

print(f"Subjects with mutations: {(pheno['n_mutations'] > 0).sum()}")
print(f"Subjects with gene weights: {(pheno['n_genes'] > 0).sum()}")
print(f"Mean mutations per subject: {pheno['n_mutations'].mean():.2f}")
print(f"Mean genes per subject: {pheno['n_genes'].mean():.2f}")

# %%
# Coverage summary table
phenotype_groups = {
    "RBS-R": ["rbsr_total"],
    "DCDQ": ["dcdq_total"],
    "Vineland": ["vine_abc"],
    "SRS": ["srs_total_t"],
    "IQ (FSIQ)": ["iq_fsiq"],
    "Milestones (words)": ["milestone_words_mos"],
}

coverage_rows = []
for group_name, cols in phenotype_groups.items():
    col = cols[0]
    n_total = pheno[col].notna().sum()
    n_spark = pheno.loc[pheno["cohort"] == "SPARK", col].notna().sum()
    n_ssc = pheno.loc[pheno["cohort"] == "SSC", col].notna().sum()
    coverage_rows.append({
        "Phenotype": group_name,
        "Total": n_total,
        "SPARK": n_spark,
        "SSC": n_ssc,
    })

coverage_df = pd.DataFrame(coverage_rows)
print("\nPhenotype coverage:")
print(coverage_df.to_string(index=False))

# %%
# Validation checks
assert pheno["IID"].is_unique, "Duplicate IIDs found!"

n_any_pheno = pheno.drop(columns=["IID", "cohort", "sex", "n_mutations",
                                   "n_genes", "total_gene_weight",
                                   "language_level", "cognitive_impairment",
                                   "regress_lang", "regress_other"]
                         ).notna().any(axis=1).sum()
print(f"Subjects with at least 1 phenotype measure: {n_any_pheno} / {len(pheno)}")

n_no_pheno = len(pheno) - n_any_pheno
if n_no_pheno > 0:
    print(f"  WARNING: {n_no_pheno} subjects have mutations but no phenotype data")

# %%
# Save outputs
out_dir = os.path.join(ProjDIR, "results/phenotype")
os.makedirs(out_dir, exist_ok=True)

pheno.to_parquet(os.path.join(out_dir, "mutation_phenotype_master.parquet"),
                 index=False)
print(f"Saved: {out_dir}/mutation_phenotype_master.parquet ({len(pheno)} rows)")

with open(os.path.join(out_dir, "subject_gene_weights.pkl"), "wb") as f:
    pk.dump(subject_gene_weights, f)
print(f"Saved: {out_dir}/subject_gene_weights.pkl ({len(subject_gene_weights)} subjects)")

# %%
# Quick summary of the final table
print("\n--- Final Table Summary ---")
print(f"Shape: {pheno.shape}")
print(f"\nCohort breakdown:")
print(pheno["cohort"].value_counts().to_string())
print(f"\nSex breakdown:")
print(pheno["sex"].value_counts().to_string())
print(f"\nColumn dtypes:")
for col in pheno.columns:
    n_valid = pheno[col].notna().sum()
    print(f"  {col:30s} {str(pheno[col].dtype):10s}  ({n_valid:>4d} non-null)")
