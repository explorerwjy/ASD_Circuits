# Phenotype-Specific Brain Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `notebook_phenotype/` with 5 notebooks mapping ASD phenotype dimensions (social, motor, repetitive behavior, adaptive function, language) to brain structure bias patterns using SPARK + SSC data.

**Architecture:** Notebook 01 builds a unified mutation-phenotype master table from SPARK + SSC. Notebooks 02-05 consume this table for stratification (binary splits), continuous mapping (Spearman correlations), data-driven subtypes (PCA/NMF), and SSC cross-validation. All expensive computations cached to `results/phenotype/`.

**Tech Stack:** Python 3.10 (conda env: `gencic`), pandas, numpy, scipy, matplotlib, seaborn, statsmodels, joblib, sklearn. Existing functions from `src/ASD_Circuits.py` and `scripts/script_phenotype_{bootstrap,permutation}.py`.

---

## Task 1: Set Up Directory Structure and Symlink

**Files:**
- Create: `notebook_phenotype/` directory
- Create: `dat/Phenotype` symlink → `/home/jw3514/Work/ASD_Phenotype/dat`
- Create: `results/phenotype/`, `results/phenotype/cache/`, `results/phenotype/stratification/`, `results/phenotype/continuous/`, `results/phenotype/subtypes/`, `results/phenotype/ssc_validation/`, `results/phenotype/figs/`

**Step 1: Create directories**

```bash
conda activate gencic
cd /home/jw3514/Work/ASD_Circuits_CellType
mkdir -p notebook_phenotype
mkdir -p results/phenotype/{cache,stratification,continuous,subtypes,ssc_validation,figs}
```

**Step 2: Create symlink to phenotype data**

```bash
ln -s /home/jw3514/Work/ASD_Phenotype/dat dat/Phenotype
ls -la dat/Phenotype/  # verify SSC_Phenotype_Dataset/ and SPARKDataRelease_2025-07-14/ visible
```

**Step 3: Verify existing data dependencies exist**

```bash
ls dat/Genetics/SPARK/ASD_Discov_DNVs.txt
ls dat/Genetics/SPARK/ASD_Rep_DNVs.txt
ls dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx
ls dat/Genetics/1-s2.0-S0092867419313984-mmc4.xlsx
```

---

## Task 2: Notebook 01 — Data Cleaning (Part 1: Load & Filter Mutations)

**Files:**
- Create: `notebook_phenotype/01.Phenotype_Data_Cleaning.py` (jupytext percent format)

**Step 1: Create the notebook .py file with setup cells**

Write `notebook_phenotype/01.Phenotype_Data_Cleaning.py` with:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 01: Phenotype Data Cleaning & Integration
#
# Build a unified mutation-phenotype table linking SPARK + SSC subjects
# with de novo LGD/Dmis mutations in exome-wide significant ASD genes
# to standardized phenotype scores across instruments.
#
# **Output**: `results/phenotype/mutation_phenotype_master.parquet`

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os, yaml
import numpy as np
import pandas as pd

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
sys.path.insert(1, os.path.join(ProjDIR, "scripts"))
from ASD_Circuits import (
    LoadGeneINFO, Filt_LGD_Mis, Mut2GeneDF, MouseSTR_AvgZ_Weighted,
)

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## Section 1: Load and Filter Mutations
#
# Same pipeline as notebook 06: load SPARK discovery + replication de novo
# variants, filter to 61 exome-wide significant genes, keep LGD + Dmis.

# %%
# Load de novo variants
ASD_Discov = pd.read_csv("../dat/Genetics/SPARK/ASD_Discov_DNVs.txt", delimiter="\t")
ASD_Rep = pd.read_csv("../dat/Genetics/SPARK/ASD_Rep_DNVs.txt", delimiter="\t")
ASD_Muts = pd.concat([ASD_Discov, ASD_Rep], ignore_index=True)
print(f"Total de novo variants: {len(ASD_Muts):,} from {ASD_Muts['IID'].nunique():,} subjects")

# Exome-wide significant genes (Zhou et al. 2022)
Spark_Meta = pd.read_excel(
    "../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
    skiprows=2, sheet_name="Table S7"
)
HighConfGenes = Spark_Meta[Spark_Meta["pDenovoWEST_Meta"] <= 1.3e-6]["HGNC"].values
print(f"Exome-wide significant genes: {len(HighConfGenes)}")

# Filter to HC genes + LGD/Dmis
HighConfMuts = Filt_LGD_Mis(ASD_Muts[ASD_Muts["HGNC"].isin(HighConfGenes)], Dmis=True)
print(f"HC mutations (LGD+Dmis in {len(HighConfGenes)} genes): {len(HighConfMuts):,}")
print(f"  Unique subjects: {HighConfMuts['IID'].nunique():,}")

# Tag cohort for SPARK subjects (Rep file has no Cohort column)
if "Cohort" not in HighConfMuts.columns:
    HighConfMuts["Cohort"] = "SPARK"
HighConfMuts.loc[HighConfMuts["Cohort"].isna(), "Cohort"] = "SPARK"

# Breakdown by cohort
print("\nCohort breakdown:")
for cohort, grp in HighConfMuts.groupby("Cohort"):
    print(f"  {cohort}: {grp['IID'].nunique()} subjects, {len(grp)} mutations")
```

**Step 2: Sync to ipynb**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType
jupytext --to notebook notebook_phenotype/01.Phenotype_Data_Cleaning.py
```

---

## Task 3: Notebook 01 — Data Cleaning (Part 2: Link SPARK Phenotypes)

**Files:**
- Modify: `notebook_phenotype/01.Phenotype_Data_Cleaning.py`

**Step 1: Add SPARK phenotype loading cells**

Append to the .py file:

```python
# %% [markdown]
# ## Section 2: Link SPARK Subjects to Phenotype Data

# %%
SPARK_DIR = "../dat/Phenotype/SPARKDataRelease_2025-07-14"

# Identify SPARK subjects in mutations
spark_muts = HighConfMuts[HighConfMuts["IID"].str.startswith("SP")]
spark_ids = spark_muts["IID"].unique()
print(f"SPARK subjects with HC mutations: {len(spark_ids)}")

# --- RBS-R ---
rbsr_spark = pd.read_csv(f"{SPARK_DIR}/rbsr-2025-07-14.csv")
rbsr_spark = rbsr_spark[rbsr_spark["subject_sp_id"].isin(spark_ids)].copy()
# Drop rows flagged invalid
if "rbsr_validity_flag" in rbsr_spark.columns:
    rbsr_spark = rbsr_spark[rbsr_spark["rbsr_validity_flag"] != 1]
# Keep latest eval per subject
rbsr_spark = rbsr_spark.sort_values("eval_year", ascending=False).drop_duplicates("subject_sp_id")
print(f"RBS-R: {len(rbsr_spark)} SPARK subjects matched")

# --- DCDQ ---
dcdq_spark = pd.read_csv(f"{SPARK_DIR}/dcdq-2025-07-14.csv")
dcdq_spark = dcdq_spark[dcdq_spark["subject_sp_id"].isin(spark_ids)].copy()
if "dcdq_measure_validity_flag" in dcdq_spark.columns:
    dcdq_spark = dcdq_spark[dcdq_spark["dcdq_measure_validity_flag"] != 1]
dcdq_spark = dcdq_spark.sort_values("eval_year", ascending=False).drop_duplicates("subject_sp_id")
print(f"DCDQ: {len(dcdq_spark)} SPARK subjects matched")

# --- Vineland-3 ---
vine_spark = pd.read_csv(f"{SPARK_DIR}/vineland-3-2025-07-14.csv")
vine_spark = vine_spark[vine_spark["subject_sp_id"].isin(spark_ids)].copy()
vine_spark = vine_spark.sort_values("age_at_eval_months", ascending=False).drop_duplicates("subject_sp_id")
print(f"Vineland-3: {len(vine_spark)} SPARK subjects matched")

# --- SRS-2 (school age) ---
srs_spark = pd.read_csv(f"{SPARK_DIR}/srs2_school_age-2025-07-14.csv")
srs_spark = srs_spark[srs_spark["subject_sp_id"].isin(spark_ids)].copy()
if "validity_flag" in srs_spark.columns:
    srs_spark = srs_spark[srs_spark["validity_flag"] != 1]
print(f"SRS-2 school-age: {len(srs_spark)} SPARK subjects matched")

# --- Core descriptive (IQ, milestones, language level) ---
core_spark = pd.read_csv(f"{SPARK_DIR}/core_descriptive_variables-2025-07-14.csv")
core_spark = core_spark[core_spark["subject_sp_id"].isin(spark_ids)].copy()
# Remove flagged invalid ASD diagnoses
core_spark = core_spark[core_spark["asd_validity_flag"] != 1]
# Remove invalid family
core_spark = core_spark[core_spark["family_sf_id"] != "SF0006897"]
print(f"Core descriptive: {len(core_spark)} SPARK subjects matched")

# --- Background history (milestones) ---
bghx_spark = pd.read_csv(f"{SPARK_DIR}/background_history_child-2025-07-14.csv")
bghx_spark = bghx_spark[bghx_spark["subject_sp_id"].isin(spark_ids)].copy()
if "bghx_validity_flag" in bghx_spark.columns:
    bghx_spark = bghx_spark[bghx_spark["bghx_validity_flag"] != 1]
bghx_spark = bghx_spark.sort_values("eval_year", ascending=False).drop_duplicates("subject_sp_id")
print(f"Background history: {len(bghx_spark)} SPARK subjects matched")

# %% [markdown]
# ### SPARK Phenotype Summary

# %%
# Build SPARK phenotype table
spark_pheno = pd.DataFrame({"subject_id": spark_ids, "cohort": "SPARK"}).set_index("subject_id")

# RBS-R subscales + total
rbsr_cols = {
    "i_stereotyped_behavior_score": "rbsr_stereotyped",
    "ii_self_injurious_score": "rbsr_self_injurious",
    "iii_compulsive_behavior_score": "rbsr_compulsive",
    "iv_ritualistic_behavior_score": "rbsr_ritualistic",
    "v_sameness_behavior_score": "rbsr_sameness",
    "vi_restricted_behavior_score": "rbsr_restricted",
    "overall_score": "rbsr_total",
}
rbsr_df = rbsr_spark.set_index("subject_sp_id")[list(rbsr_cols.keys())].rename(columns=rbsr_cols)
spark_pheno = spark_pheno.join(rbsr_df)

# DCDQ subscales + total
dcdq_cols = {
    "control_during_movement": "dcdq_control",
    "fine_motor_handwriting": "dcdq_fine_motor",
    "general_coordination": "dcdq_coordination",
    "total": "dcdq_total",
    "dcd": "dcdq_flag",
}
dcdq_df = dcdq_spark.set_index("subject_sp_id")[list(dcdq_cols.keys())].rename(columns=dcdq_cols)
spark_pheno = spark_pheno.join(dcdq_df)

# Vineland-3 domain standard scores
vine_cols = {
    "abc_standard": "vine_abc",
    "communication_standard": "vine_communication",
    "dls_standard": "vine_dls",
    "soc_standard": "vine_social",
    "motor_standard": "vine_motor",
}
vine_df = vine_spark.set_index("subject_sp_id")[list(vine_cols.keys())].rename(columns=vine_cols)
spark_pheno = spark_pheno.join(vine_df)

# SRS-2 T-scores
srs_cols = {
    "total_t_score": "srs_total_t",
    "rrb_t_score": "srs_rrb_t",
    "awr_t_score": "srs_awareness_t",
    "soc_cog_t_score": "srs_soccog_t",
    "com_t_score": "srs_communication_t",
    "mot_t_score": "srs_motivation_t",
    "sci_t_score": "srs_sci_t",
}
srs_df = srs_spark.set_index("subject_sp_id")[list(srs_cols.keys())].rename(columns=srs_cols)
spark_pheno = spark_pheno.join(srs_df)

# Core descriptive: IQ, language level
core_cols = {
    "sex": "sex",
    "fsiq": "iq_fsiq",
    "viq": "iq_verbal",
    "nviq": "iq_nonverbal",
    "language_level_latest": "language_level",
    "cognitive_impairment_latest": "cognitive_impairment",
}
core_df = core_spark.set_index("subject_sp_id")[list(core_cols.keys())].rename(columns=core_cols)
spark_pheno = spark_pheno.join(core_df)

# Background history: milestones
mile_cols = {
    "walked_age_mos": "milestone_walked_mos",
    "used_words_age_mos": "milestone_words_mos",
    "combined_phrases_age_mos": "milestone_phrases_mos",
    "sat_wo_support_age_mos": "milestone_sat_mos",
    "age_onset_mos": "onset_age_mos",
    "regress_lang_y_n": "regression_language",
    "regress_other_y_n": "regression_other",
}
mile_df = bghx_spark.set_index("subject_sp_id")[list(mile_cols.keys())].rename(columns=mile_cols)
spark_pheno = spark_pheno.join(mile_df)

# Replace 888 with NaN (SPARK convention: 888 = not applicable)
spark_pheno = spark_pheno.replace(888, np.nan)

print(f"\nSPARK phenotype table: {spark_pheno.shape}")
print("\nNon-null counts per phenotype:")
print(spark_pheno.notna().sum().to_string())
```

**Step 2: Sync**

```bash
jupytext --sync notebook_phenotype/01.Phenotype_Data_Cleaning.py
```

---

## Task 4: Notebook 01 — Data Cleaning (Part 3: Link SSC Phenotypes)

**Files:**
- Modify: `notebook_phenotype/01.Phenotype_Data_Cleaning.py`

**Step 1: Add SSC phenotype loading cells**

Append to the .py file:

```python
# %% [markdown]
# ## Section 3: Link SSC Subjects to Phenotype Data
#
# SSC subjects use IID format `NNNNN.p1`. SSC phenotype files use
# `individual` column with the same format. Direct string match.

# %%
SSC_DIR = "../dat/Phenotype/SSC_Phenotype_Dataset/SSC_V15_Phenotype_DATA/Proband_Data"

# Identify SSC probands in mutations
ssc_muts = HighConfMuts[HighConfMuts["IID"].str.match(r"^\d+\.p\d+$")]
ssc_ids = ssc_muts["IID"].unique()
print(f"SSC subjects with HC mutations: {len(ssc_ids)}")

# --- RBS-R ---
rbsr_ssc = pd.read_csv(f"{SSC_DIR}/rbs_r.csv")
rbsr_ssc = rbsr_ssc[rbsr_ssc["individual"].isin(ssc_ids)].copy()
print(f"RBS-R: {len(rbsr_ssc)} SSC subjects matched")

# --- DCDQ ---
dcdq_ssc = pd.read_csv(f"{SSC_DIR}/dcdq.csv")
dcdq_ssc = dcdq_ssc[dcdq_ssc["individual"].isin(ssc_ids)].copy()
print(f"DCDQ: {len(dcdq_ssc)} SSC subjects matched")

# --- Vineland-II ---
vine_ssc = pd.read_csv(f"{SSC_DIR}/vineland_ii.csv")
vine_ssc = vine_ssc[vine_ssc["individual"].isin(ssc_ids)].copy()
print(f"Vineland-II: {len(vine_ssc)} SSC subjects matched")

# --- SRS parent ---
srs_ssc = pd.read_csv(f"{SSC_DIR}/srs_parent.csv")
srs_ssc = srs_ssc[srs_ssc["individual"].isin(ssc_ids)].copy()
print(f"SRS parent: {len(srs_ssc)} SSC subjects matched")

# --- Core descriptive (IQ, demographics) ---
core_ssc = pd.read_csv(f"{SSC_DIR}/ssc_core_descriptive.csv")
core_ssc = core_ssc[core_ssc["individual"].isin(ssc_ids)].copy()
print(f"Core descriptive: {len(core_ssc)} SSC subjects matched")

# %% [markdown]
# ### SSC Phenotype Table

# %%
ssc_pheno = pd.DataFrame({"subject_id": ssc_ids, "cohort": "SSC"}).set_index("subject_id")

# RBS-R (same subscale names as SPARK, map to unified columns)
ssc_rbsr_cols = {
    "i_stereotyped_behavior_score": "rbsr_stereotyped",
    "ii_self_injurious_score": "rbsr_self_injurious",
    "iii_compulsive_behavior_score": "rbsr_compulsive",
    "iv_ritualistic_behavior_score": "rbsr_ritualistic",
    "v_sameness_behavior_score": "rbsr_sameness",
    "vi_restricted_behavior_score": "rbsr_restricted",
    "overall_score": "rbsr_total",
}
ssc_rbsr_df = rbsr_ssc.set_index("individual")[list(ssc_rbsr_cols.keys())].rename(columns=ssc_rbsr_cols)
ssc_pheno = ssc_pheno.join(ssc_rbsr_df)

# DCDQ (note SSC typo: general_corrdination)
ssc_dcdq_cols = {
    "control_during_movement": "dcdq_control",
    "fine_motor_handwriting": "dcdq_fine_motor",
    "general_corrdination": "dcdq_coordination",  # typo in SSC data
    "total": "dcdq_total",
}
dcdq_ssc_sub = dcdq_ssc.set_index("individual")[list(ssc_dcdq_cols.keys())].rename(columns=ssc_dcdq_cols)
ssc_pheno = ssc_pheno.join(dcdq_ssc_sub)

# Vineland-II → unified column names (standard scores comparable across V2/V3)
ssc_vine_cols = {
    "composite_standard_score": "vine_abc",
    "communication_standard": "vine_communication",
    "dls_standard": "vine_dls",
    "soc_standard": "vine_social",
    "motor_skills_standard": "vine_motor",
}
ssc_vine_df = vine_ssc.set_index("individual")[list(ssc_vine_cols.keys())].rename(columns=ssc_vine_cols)
ssc_pheno = ssc_pheno.join(ssc_vine_df)

# SRS parent T-score (SSC SRS v1 T-scores comparable to SRS-2 T-scores)
ssc_srs_cols = {
    "t_score": "srs_total_t",
    "awareness_t_score": "srs_awareness_t",
    "cognition_t_score": "srs_soccog_t",
    "communication_t_score": "srs_communication_t",
    "motivation_t_score": "srs_motivation_t",
    "mannerisms_t_score": "srs_rrb_t",  # SRS mannerisms ≈ SRS-2 RRB
}
srs_ssc_sub = srs_ssc.set_index("individual")[list(ssc_srs_cols.keys())].rename(columns=ssc_srs_cols)
ssc_pheno = ssc_pheno.join(srs_ssc_sub)

# Core descriptive: IQ
ssc_core_cols = {
    "sex": "sex",
    "ssc_diagnosis_full_scale_iq": "iq_fsiq",
    "ssc_diagnosis_verbal_iq": "iq_verbal",
    "ssc_diagnosis_nonverbal_iq": "iq_nonverbal",
}
ssc_core_df = core_ssc.set_index("individual")[list(ssc_core_cols.keys())].rename(columns=ssc_core_cols)
ssc_pheno = ssc_pheno.join(ssc_core_df)

print(f"\nSSC phenotype table: {ssc_pheno.shape}")
print("\nNon-null counts per phenotype:")
print(ssc_pheno.notna().sum().to_string())
```

**Step 2: Sync**

```bash
jupytext --sync notebook_phenotype/01.Phenotype_Data_Cleaning.py
```

---

## Task 5: Notebook 01 — Data Cleaning (Part 4: Merge & Validate)

**Files:**
- Modify: `notebook_phenotype/01.Phenotype_Data_Cleaning.py`

**Step 1: Add merge, validation, and save cells**

Append to the .py file:

```python
# %% [markdown]
# ## Section 4: Merge SPARK + SSC into Master Table

# %%
# Concatenate phenotype tables
master = pd.concat([spark_pheno, ssc_pheno])
print(f"Master table: {master.shape[0]} subjects × {master.shape[1]} phenotype columns")
print(f"  SPARK: {(master['cohort'] == 'SPARK').sum()}")
print(f"  SSC: {(master['cohort'] == 'SSC').sum()}")

# %% [markdown]
# ### Attach mutation info (gene weights per subject)

# %%
# Compute per-subject gene weights
# Group mutations by subject, compute gene weight dict for each
subject_gene_weights = {}
subject_mutation_counts = {}
for iid, grp in HighConfMuts.groupby("IID"):
    if iid in master.index:
        gw = Mut2GeneDF(grp, gene_col="HGNC", gene_symbol_to_entrez=GeneSymbol2Entrez)
        subject_gene_weights[iid] = gw
        subject_mutation_counts[iid] = len(grp)

master["n_mutations"] = pd.Series(subject_mutation_counts)
master["n_genes"] = pd.Series({k: len(v) for k, v in subject_gene_weights.items()})

print(f"\nSubjects with gene weights: {len(subject_gene_weights)}")
print(f"Mutation count distribution:")
print(master["n_mutations"].describe())

# %% [markdown]
# ### Phenotype coverage summary

# %%
# Summary table: phenotype × cohort coverage
pheno_groups = {
    "RBS-R": ["rbsr_total"],
    "DCDQ": ["dcdq_total"],
    "Vineland": ["vine_abc"],
    "SRS": ["srs_total_t"],
    "IQ": ["iq_fsiq"],
    "Milestones": ["milestone_words_mos"],
}

coverage = []
for name, cols in pheno_groups.items():
    col = cols[0]
    for cohort in ["SPARK", "SSC"]:
        sub = master[master["cohort"] == cohort]
        n_total = len(sub)
        n_valid = sub[col].notna().sum() if col in sub.columns else 0
        coverage.append({"Instrument": name, "Cohort": cohort,
                         "N_total": n_total, "N_valid": n_valid,
                         "Pct": f"{100*n_valid/n_total:.1f}%" if n_total > 0 else "N/A"})

coverage_df = pd.DataFrame(coverage)
print(coverage_df.pivot(index="Instrument", columns="Cohort", values=["N_valid", "Pct"]).to_string())

# %% [markdown]
# ### Validation checks

# %%
# 1. No duplicate subject IDs
assert master.index.is_unique, "Duplicate subject IDs found!"

# 2. All subjects have at least one phenotype score
pheno_cols = [c for c in master.columns if c not in ["cohort", "sex", "n_mutations", "n_genes"]]
has_any = master[pheno_cols].notna().any(axis=1)
print(f"Subjects with at least 1 phenotype score: {has_any.sum()} / {len(master)}")
if (~has_any).any():
    print(f"  WARNING: {(~has_any).sum()} subjects have NO phenotype data")

# 3. Phenotype distributions look reasonable (no all-zeros, no extreme outliers)
for col in ["rbsr_total", "dcdq_total", "vine_abc", "srs_total_t"]:
    if col in master.columns:
        vals = master[col].dropna()
        if len(vals) > 0:
            print(f"{col}: N={len(vals)}, mean={vals.mean():.1f}, "
                  f"std={vals.std():.1f}, min={vals.min()}, max={vals.max()}")

# %% [markdown]
# ### Save master table

# %%
# Save
out_path = "../results/phenotype/mutation_phenotype_master.parquet"
master.to_parquet(out_path)
print(f"Saved: {out_path} ({master.shape})")

# Also save the per-subject gene weight dict for downstream use
import pickle
gw_path = "../results/phenotype/subject_gene_weights.pkl"
with open(gw_path, "wb") as f:
    pickle.dump(subject_gene_weights, f)
print(f"Saved: {gw_path} ({len(subject_gene_weights)} subjects)")
```

**Step 2: Sync and run**

```bash
jupytext --sync notebook_phenotype/01.Phenotype_Data_Cleaning.py
cd notebook_phenotype && jupyter nbconvert --to notebook --execute --inplace 01.Phenotype_Data_Cleaning.ipynb
```

**Step 3: Verify outputs**

```bash
ls -la ../results/phenotype/mutation_phenotype_master.parquet
ls -la ../results/phenotype/subject_gene_weights.pkl
```

**Step 4: Commit**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType
git add notebook_phenotype/01.Phenotype_Data_Cleaning.py notebook_phenotype/01.Phenotype_Data_Cleaning.ipynb
git commit -m "Add phenotype data cleaning notebook linking SPARK+SSC mutations to phenotypes"
```

---

## Task 6: Notebook 02 — Phenotype Stratification (Part 1: Framework)

**Files:**
- Create: `notebook_phenotype/02.Phenotype_Stratification.py`

**Step 1: Create notebook with setup and stratification framework**

Write `notebook_phenotype/02.Phenotype_Stratification.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 02: Phenotype Stratification
#
# Binary phenotype splits → separate gene weights → brain bias comparison.
# Direct extension of notebook 06 (IQ) to other phenotype dimensions.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os, yaml, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
sys.path.insert(1, os.path.join(ProjDIR, "scripts"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted,
    Mut2GeneDF, Filt_LGD_Mis,
)
from script_phenotype_bootstrap import bootstrap_phenotype_bias
from script_phenotype_permutation import permutation_test_phenotype

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
ExpZ2Mat = pd.read_parquet(f"../{expr_matrix_path}")

SEED = 42
CACHE_DIR = "../results/phenotype/cache"
FIG_DIR = "../results/phenotype/figs"

# %%
# Load master table and mutations
master = pd.read_parquet("../results/phenotype/mutation_phenotype_master.parquet")
with open("../results/phenotype/subject_gene_weights.pkl", "rb") as f:
    subject_gene_weights = pickle.load(f)

# Load raw mutations for bootstrap/permutation (need mutation-level DataFrame)
ASD_Discov = pd.read_csv("../dat/Genetics/SPARK/ASD_Discov_DNVs.txt", delimiter="\t")
ASD_Rep = pd.read_csv("../dat/Genetics/SPARK/ASD_Rep_DNVs.txt", delimiter="\t")
ASD_Muts = pd.concat([ASD_Discov, ASD_Rep], ignore_index=True)
Spark_Meta = pd.read_excel(
    "../dat/Genetics/41588_2022_1148_MOESM4_ESM.xlsx",
    skiprows=2, sheet_name="Table S7"
)
HighConfGenes = Spark_Meta[Spark_Meta["pDenovoWEST_Meta"] <= 1.3e-6]["HGNC"].values
HighConfMuts = Filt_LGD_Mis(ASD_Muts[ASD_Muts["HGNC"].isin(HighConfGenes)], Dmis=True)

print(f"Master table: {master.shape}")
print(f"HC mutations: {len(HighConfMuts)} from {HighConfMuts['IID'].nunique()} subjects")

# %% [markdown]
# ## Stratification Framework
#
# For each phenotype dimension:
# 1. Split subjects at median into High/Low groups
# 2. Compute gene weights per group from their mutations
# 3. Compute brain bias per group
# 4. Bootstrap CI (1000 resamples)
# 5. Permutation test for group difference (10,000 perms)

# %%
def stratify_and_analyze(pheno_col, pheno_label, split="median",
                         threshold=None, higher_is_worse=True):
    """
    Run full stratification analysis for one phenotype dimension.

    Parameters
    ----------
    pheno_col : str
        Column name in master table
    pheno_label : str
        Human-readable label for plots
    split : str
        "median" or "threshold"
    threshold : float, optional
        Custom threshold (used if split="threshold")
    higher_is_worse : bool
        If True, High group = more severe. Affects labeling.

    Returns
    -------
    dict with keys: high_df, low_df, high_boot, low_boot, perm_pvals
    """
    # Get subjects with valid phenotype scores
    valid = master[master[pheno_col].notna()].copy()
    valid_ids = set(valid.index)
    muts_with_pheno = HighConfMuts[HighConfMuts["IID"].isin(valid_ids)].copy()

    if split == "median":
        threshold = valid[pheno_col].median()
    elif split == "threshold" and threshold is None:
        raise ValueError("Must provide threshold for split='threshold'")

    if higher_is_worse:
        high_ids = valid[valid[pheno_col] > threshold].index
        low_ids = valid[valid[pheno_col] <= threshold].index
        high_label, low_label = f"High {pheno_label}", f"Low {pheno_label}"
    else:
        high_ids = valid[valid[pheno_col] <= threshold].index
        low_ids = valid[valid[pheno_col] > threshold].index
        high_label, low_label = f"Low {pheno_label}", f"High {pheno_label}"

    high_muts = muts_with_pheno[muts_with_pheno["IID"].isin(high_ids)]
    low_muts = muts_with_pheno[muts_with_pheno["IID"].isin(low_ids)]

    print(f"\n{'='*60}")
    print(f"{pheno_label} stratification (threshold={threshold:.1f})")
    print(f"  {high_label}: {len(high_ids)} subjects, {len(high_muts)} mutations")
    print(f"  {low_label}: {len(low_ids)} subjects, {len(low_muts)} mutations")

    # Gene weights
    high_gw = Mut2GeneDF(high_muts, gene_col="HGNC",
                          gene_symbol_to_entrez=GeneSymbol2Entrez)
    low_gw = Mut2GeneDF(low_muts, gene_col="HGNC",
                         gene_symbol_to_entrez=GeneSymbol2Entrez)
    print(f"  {high_label} genes: {len(high_gw)}")
    print(f"  {low_label} genes: {len(low_gw)}")

    # Bias
    high_df = MouseSTR_AvgZ_Weighted(ExpZ2Mat, high_gw)
    low_df = MouseSTR_AvgZ_Weighted(ExpZ2Mat, low_gw)

    # Bootstrap CI
    high_boot = bootstrap_phenotype_bias(
        high_muts, ExpZ2Mat, gene_symbol_to_entrez=GeneSymbol2Entrez,
        n_boot=1000, n_jobs=10, seed=SEED,
        cache_dir=CACHE_DIR, group_name=f"{pheno_col}.high"
    )
    low_boot = bootstrap_phenotype_bias(
        low_muts, ExpZ2Mat, gene_symbol_to_entrez=GeneSymbol2Entrez,
        n_boot=1000, n_jobs=10, seed=SEED,
        cache_dir=CACHE_DIR, group_name=f"{pheno_col}.low"
    )

    # Permutation test
    # Need phenotype column in mutation df for permutation
    muts_with_pheno = muts_with_pheno.copy()
    muts_with_pheno[pheno_col] = muts_with_pheno["IID"].map(
        valid[pheno_col].to_dict()
    )

    perm_cache = f"{CACHE_DIR}/{pheno_col}_perm_pvals.csv"
    perm_pvals = permutation_test_phenotype(
        muts_with_pheno, phenotype_col=pheno_col, threshold=threshold,
        exp_mat=ExpZ2Mat, gene_symbol_to_entrez=GeneSymbol2Entrez,
        n_perm=10000, n_jobs=10, seed=SEED,
        cache_path=perm_cache
    )

    return {
        "high_label": high_label, "low_label": low_label,
        "high_df": high_df, "low_df": low_df,
        "high_boot": high_boot, "low_boot": low_boot,
        "perm_pvals": perm_pvals,
        "threshold": threshold,
        "n_high": len(high_ids), "n_low": len(low_ids),
    }
```

**Step 2: Sync**

```bash
jupytext --to notebook notebook_phenotype/02.Phenotype_Stratification.py
```

---

## Task 7: Notebook 02 — Phenotype Stratification (Part 2: Run Analyses & Plot)

**Files:**
- Modify: `notebook_phenotype/02.Phenotype_Stratification.py`

**Step 1: Add analysis execution and plotting cells**

Append to the .py file:

```python
# %% [markdown]
# ## Section 2: Run Stratification for Each Phenotype
#
# Dimensions analyzed (in order of statistical power):
# 1. RBS-R total (repetitive behavior severity)
# 2. DCDQ total (motor coordination — lower = worse)
# 3. Vineland ABC (adaptive behavior — lower = worse)
# 4. SRS total T-score (social responsiveness — higher = worse)
# 5. Age of first words (language milestone)
# 6. IQ (cognitive — lower = worse, replication of notebook 06)

# %%
# RBS-R: higher = more repetitive behavior (worse)
results_rbsr = stratify_and_analyze("rbsr_total", "RBS-R", higher_is_worse=True)

# %%
# DCDQ: higher = better motor coordination (lower = worse)
results_dcdq = stratify_and_analyze("dcdq_total", "DCDQ Motor", higher_is_worse=False)

# %%
# Vineland ABC: lower = worse adaptive behavior
results_vine = stratify_and_analyze("vine_abc", "Vineland ABC", higher_is_worse=False)

# %%
# SRS total T-score: higher = worse social responsiveness
results_srs = stratify_and_analyze("srs_total_t", "SRS Social", higher_is_worse=True)

# %%
# Age of first words: higher = more delayed
# Filter out 888 (already replaced with NaN in NB01)
results_words = stratify_and_analyze("milestone_words_mos", "Word Delay", higher_is_worse=True)

# %%
# IQ: lower = worse (replication of notebook 06 with combined SPARK+SSC)
results_iq = stratify_and_analyze("iq_fsiq", "IQ", split="threshold",
                                   threshold=70, higher_is_worse=False)

# %% [markdown]
# ## Section 3: Visualization — Regional Bias Comparison
#
# For each phenotype, plot High vs Low group bias across the 46 ASD circuit
# structures, grouped by brain region, with bootstrap CI and permutation p-values.

# %%
# Load ASD circuit structures
from ASD_Circuits import Fil2Dict
import glob

# Load Pareto circuit (index 3, same as main analysis)
pareto_dir = "../results/Circuits"
pareto_files = sorted(glob.glob(f"{pareto_dir}/ASD_Pareto_SI_Size46/Pareto_*.txt"))
if len(pareto_files) > 3:
    circuit_strs_raw = list(Fil2Dict(pareto_files[3]).keys())
else:
    # Fallback: load from config or results
    circuit_strs_raw = list(pd.read_csv(
        "../dat/CircuitSearch/SA_results/ASD_Pareto_SI_Size46.ParetoIdx3.txt",
        header=None)[0])

ASD_Circuit = [s.replace("_", " ") for s in circuit_strs_raw]
CIR_REGIONS_Dict = {}
for s in ASD_Circuit:
    reg = str2reg.get(s, "Other")
    CIR_REGIONS_Dict.setdefault(reg, []).append(s)

REGION_COLORS = {
    "Isocortex": "#1f77b4",
    "OLF": "#ff7f0e",
    "HPF": "#2ca02c",
    "CTXsp": "#d62728",
    "STR": "#9467bd",
    "PAL": "#8c564b",
    "TH": "#e377c2",
    "HY": "#7f7f7f",
    "MB": "#bcbd22",
}
REGION_ORDER = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB"]

# %%
def plot_phenotype_comparison(result, pheno_label, save=True):
    """Plot regional bias comparison for a phenotype stratification."""
    high_df = result["high_df"]
    low_df = result["low_df"]
    high_boot = result["high_boot"]["ALL"]
    low_boot = result["low_boot"]["ALL"]
    perm_pvals = result["perm_pvals"]

    # Build ordered structure list by region
    ordered_strs = []
    region_boundaries = []
    for reg in REGION_ORDER:
        if reg in CIR_REGIONS_Dict:
            start = len(ordered_strs)
            ordered_strs.extend(sorted(CIR_REGIONS_Dict[reg]))
            region_boundaries.append((reg, start, len(ordered_strs)))

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(ordered_strs))
    width = 0.35

    for i, s in enumerate(ordered_strs):
        reg = str2reg.get(s, "Other")
        color = REGION_COLORS.get(reg, "#333333")

        # High group
        h_val = high_df.loc[s, "EFFECT"] if s in high_df.index else 0
        h_se = high_boot.loc[s].std() if s in high_boot.index else 0
        ax.bar(i - width/2, h_val, width, color=color, alpha=0.4,
               edgecolor=color, linewidth=1.5)
        ax.errorbar(i - width/2, h_val, yerr=h_se, color=color,
                     capsize=2, linewidth=1, fmt="none")

        # Low group
        l_val = low_df.loc[s, "EFFECT"] if s in low_df.index else 0
        l_se = low_boot.loc[s].std() if s in low_boot.index else 0
        ax.bar(i + width/2, l_val, width, color=color, alpha=0.85)
        ax.errorbar(i + width/2, l_val, yerr=l_se, color=color,
                     capsize=2, linewidth=1, fmt="none")

        # Significance star
        if s.replace(" ", "_") in perm_pvals.index or s in perm_pvals.index:
            idx = s if s in perm_pvals.index else s.replace(" ", "_")
            p = perm_pvals.loc[idx, "Pvalue"]
            if p < 0.001:
                ax.text(i, max(h_val, l_val) + h_se + 0.02, "***",
                        ha="center", fontsize=8, fontweight="bold")
            elif p < 0.01:
                ax.text(i, max(h_val, l_val) + h_se + 0.02, "**",
                        ha="center", fontsize=8, fontweight="bold")
            elif p < 0.05:
                ax.text(i, max(h_val, l_val) + h_se + 0.02, "*",
                        ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in ordered_strs],
                        rotation=90, fontsize=7)
    ax.set_ylabel("Weighted Avg Z2 Bias (EFFECT)")
    ax.set_title(f"{result['high_label']} (N={result['n_high']}) vs "
                 f"{result['low_label']} (N={result['n_low']})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Region dividers
    for reg, start, end in region_boundaries:
        if start > 0:
            ax.axvline(start - 0.5, color="lightgray", linewidth=0.5, linestyle=":")
        ax.text((start + end) / 2, ax.get_ylim()[0] * 0.95, reg,
                ha="center", fontsize=8, color=REGION_COLORS.get(reg, "gray"))

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="gray", alpha=0.4, edgecolor="gray"),
               Patch(facecolor="gray", alpha=0.85)],
              [result["high_label"], result["low_label"]],
              loc="upper right", fontsize=9)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.tight_layout()

    if save:
        fname = f"{FIG_DIR}/stratification_{pheno_label.replace(' ', '_')}.pdf"
        fig.savefig(fname, transparent=True, dpi=300, bbox_inches="tight")
        print(f"Saved: {fname}")

    plt.show()
    return fig

# %%
# Plot all phenotype comparisons
all_results = {
    "RBS-R": results_rbsr,
    "DCDQ Motor": results_dcdq,
    "Vineland ABC": results_vine,
    "SRS Social": results_srs,
    "Word Delay": results_words,
    "IQ": results_iq,
}

for label, res in all_results.items():
    plot_phenotype_comparison(res, label)

# %% [markdown]
# ## Section 4: Summary — Significant Structures per Phenotype

# %%
summary_rows = []
for label, res in all_results.items():
    pvals = res["perm_pvals"]
    n_sig_05 = (pvals["Pvalue"] < 0.05).sum()
    n_sig_01 = (pvals["Pvalue"] < 0.01).sum()

    # Which structures differ most?
    top_strs = pvals.nsmallest(5, "Pvalue")
    top_names = ", ".join([f"{idx} (p={row['Pvalue']:.4f})"
                           for idx, row in top_strs.iterrows()])

    summary_rows.append({
        "Phenotype": label,
        "N_high": res["n_high"],
        "N_low": res["n_low"],
        "Threshold": f"{res['threshold']:.1f}",
        "Sig_p05": n_sig_05,
        "Sig_p01": n_sig_01,
        "Top_5_structures": top_names,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df[["Phenotype", "N_high", "N_low", "Sig_p05", "Sig_p01"]].to_string(index=False))

# Save
summary_df.to_csv("../results/phenotype/stratification/summary.csv", index=False)
```

**Step 2: Sync and commit**

```bash
jupytext --sync notebook_phenotype/02.Phenotype_Stratification.py
git add notebook_phenotype/02.Phenotype_Stratification.py notebook_phenotype/02.Phenotype_Stratification.ipynb
git commit -m "Add phenotype stratification notebook (binary splits for RBS-R, DCDQ, Vineland, SRS, milestones, IQ)"
```

---

## Task 8: Notebook 03 — Continuous Phenotype-Brain Mapping

**Files:**
- Create: `notebook_phenotype/03.Phenotype_Brain_Mapping.py`

**Step 1: Create notebook**

Write `notebook_phenotype/03.Phenotype_Brain_Mapping.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 03: Continuous Phenotype-Brain Mapping
#
# For each brain structure, quantify how its expression-weighted mutation
# load correlates with phenotype severity. No binary splits — uses
# continuous phenotype scores for maximum statistical power.
#
# **Method**: Per-subject structure bias vector, then Spearman correlation
# between structure bias and phenotype score across subjects.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os, yaml, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ExpZ2Mat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")

SEED = 42
N_PERM = 10000
N_JOBS = 10
CACHE_DIR = "../results/phenotype/cache"
FIG_DIR = "../results/phenotype/figs"
structures = ExpZ2Mat.columns.tolist()

# %%
# Load master table and per-subject gene weights
master = pd.read_parquet("../results/phenotype/mutation_phenotype_master.parquet")
with open("../results/phenotype/subject_gene_weights.pkl", "rb") as f:
    subject_gene_weights = pickle.load(f)

print(f"Subjects: {len(master)}, Structures: {len(structures)}")

# %% [markdown]
# ## Section 1: Compute Per-Subject Structure Bias Matrix
#
# For each subject $i$ with gene weight dict $w_g$:
# $$b_i(s) = \frac{\sum_{g} w_g \cdot Z_2(g,s)}{\sum_{g} w_g}$$
#
# Result: (N_subjects × 213 structures) matrix

# %%
cache_path = f"{CACHE_DIR}/subject_structure_bias_matrix.parquet"

if os.path.exists(cache_path):
    subj_bias_mat = pd.read_parquet(cache_path)
    print(f"Loaded cached subject-structure bias matrix: {subj_bias_mat.shape}")
else:
    rows = {}
    for iid, gw in subject_gene_weights.items():
        bias_df = MouseSTR_AvgZ_Weighted(ExpZ2Mat, gw)
        rows[iid] = bias_df["EFFECT"]
    subj_bias_mat = pd.DataFrame(rows).T
    subj_bias_mat.to_parquet(cache_path)
    print(f"Computed and cached: {subj_bias_mat.shape}")

# %% [markdown]
# ## Section 2: Phenotype-Structure Correlations
#
# For each phenotype × structure pair, compute Spearman $\rho$ and
# permutation p-value (10K permutations of phenotype labels).

# %%
phenotype_cols = {
    "rbsr_total": "RBS-R Total",
    "rbsr_stereotyped": "Stereotyped Behavior",
    "rbsr_self_injurious": "Self-Injurious",
    "rbsr_compulsive": "Compulsive",
    "rbsr_sameness": "Sameness",
    "rbsr_restricted": "Restricted Interests",
    "dcdq_total": "Motor Coordination (DCDQ)",
    "vine_abc": "Adaptive Behavior (Vineland)",
    "vine_communication": "Communication (Vineland)",
    "vine_social": "Socialization (Vineland)",
    "vine_motor": "Motor (Vineland)",
    "srs_total_t": "Social Responsiveness (SRS)",
    "iq_fsiq": "Full-Scale IQ",
    "milestone_words_mos": "Age First Words",
}

def compute_phenotype_structure_corr(pheno_col, n_perm=N_PERM, seed=SEED):
    """Compute Spearman correlation + permutation p-value for one phenotype."""
    valid_ids = master[master[pheno_col].notna()].index
    valid_ids = valid_ids.intersection(subj_bias_mat.index)
    if len(valid_ids) < 20:
        return None

    pheno_vals = master.loc[valid_ids, pheno_col].values
    bias_vals = subj_bias_mat.loc[valid_ids].values  # (N × 213)

    # Observed correlations
    obs_rho = np.array([stats.spearmanr(bias_vals[:, j], pheno_vals).statistic
                        for j in range(bias_vals.shape[1])])

    # Permutation null
    rng = np.random.default_rng(seed)
    null_rhos = np.zeros((n_perm, bias_vals.shape[1]))
    for p in range(n_perm):
        shuffled = rng.permutation(pheno_vals)
        for j in range(bias_vals.shape[1]):
            null_rhos[p, j] = stats.spearmanr(bias_vals[:, j], shuffled).statistic

    # Two-sided p-value
    pvals = np.mean(np.abs(null_rhos) >= np.abs(obs_rho)[None, :], axis=0)
    # Add continuity correction
    pvals = (np.sum(np.abs(null_rhos) >= np.abs(obs_rho)[None, :], axis=0) + 1) / (n_perm + 1)

    result = pd.DataFrame({
        "structure": structures,
        "rho": obs_rho,
        "pvalue": pvals,
        "n_subjects": len(valid_ids),
    })
    result["region"] = result["structure"].map(str2reg)

    # FDR correction (BH)
    from statsmodels.stats.multitest import multipletests
    _, result["qvalue"], _, _ = multipletests(result["pvalue"], method="fdr_bh")

    return result.set_index("structure")

# %%
# Run for all phenotypes (cached)
all_corr_results = {}
for pheno_col, pheno_label in phenotype_cols.items():
    cache_file = f"{CACHE_DIR}/continuous_corr_{pheno_col}.parquet"
    if os.path.exists(cache_file):
        all_corr_results[pheno_col] = pd.read_parquet(cache_file)
        print(f"Loaded cached: {pheno_label} "
              f"(N={all_corr_results[pheno_col]['n_subjects'].iloc[0]})")
    else:
        print(f"Computing: {pheno_label}...")
        result = compute_phenotype_structure_corr(pheno_col)
        if result is not None:
            result.to_parquet(cache_file)
            all_corr_results[pheno_col] = result
            n_sig = (result["qvalue"] < 0.05).sum()
            print(f"  N={result['n_subjects'].iloc[0]}, "
                  f"sig structures (q<0.05): {n_sig}")
        else:
            print(f"  Skipped (too few subjects)")

# %% [markdown]
# ## Section 3: Phenotype × Structure Heatmap
#
# Rows = phenotype dimensions, Columns = brain structures (ordered by region).
# Color = Spearman $\rho$. Stars = FDR < 0.05.

# %%
# Build heatmap matrix (phenotype × structure)
pheno_keys = [k for k in phenotype_cols if k in all_corr_results]
rho_matrix = pd.DataFrame(
    {phenotype_cols[k]: all_corr_results[k]["rho"] for k in pheno_keys}
).T
qval_matrix = pd.DataFrame(
    {phenotype_cols[k]: all_corr_results[k]["qvalue"] for k in pheno_keys}
).T

# Order columns by region
region_order = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB"]
ordered_structs = []
for reg in region_order:
    reg_strs = [s for s in structures if str2reg.get(s) == reg]
    ordered_structs.extend(sorted(reg_strs))
# Add any unassigned
ordered_structs.extend([s for s in structures if s not in ordered_structs])

rho_ordered = rho_matrix[ordered_structs]
qval_ordered = qval_matrix[ordered_structs]

# Plot
fig, ax = plt.subplots(figsize=(24, len(pheno_keys) * 0.6 + 2))
sns.heatmap(rho_ordered, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
            ax=ax, xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.5})

# Mark significant cells
for i in range(qval_ordered.shape[0]):
    for j in range(qval_ordered.shape[1]):
        if qval_ordered.iloc[i, j] < 0.05:
            ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                    fontsize=6, color="black", fontweight="bold")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_title("Phenotype-Structure Correlation Map (Spearman ρ)")
fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/phenotype_structure_heatmap.pdf",
            transparent=True, dpi=300, bbox_inches="tight")
print(f"Saved heatmap")
plt.show()

# %% [markdown]
# ## Section 4: Top Structures per Phenotype

# %%
for pheno_col, pheno_label in phenotype_cols.items():
    if pheno_col not in all_corr_results:
        continue
    res = all_corr_results[pheno_col]
    n_sig = (res["qvalue"] < 0.05).sum()
    top5 = res.nsmallest(5, "pvalue")
    print(f"\n{pheno_label} (N={res['n_subjects'].iloc[0]}, sig q<0.05: {n_sig}):")
    for s, row in top5.iterrows():
        star = "*" if row["qvalue"] < 0.05 else ""
        print(f"  {s}: ρ={row['rho']:.3f}, p={row['pvalue']:.4f}, "
              f"q={row['qvalue']:.4f} {star}")

# Save all results
for pheno_col in all_corr_results:
    out = f"../results/phenotype/continuous/{pheno_col}_corr.csv"
    all_corr_results[pheno_col].to_csv(out)
print(f"\nSaved all correlation results to results/phenotype/continuous/")

# %% [markdown]
# ## Section 5: Confound Check
#
# Partial correlations controlling for total mutation count, sex, and cohort.

# %%
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def partial_correlation_structure(pheno_col, confounds=["n_mutations"]):
    """Partial Spearman: rank-transform then regress out confounds."""
    valid_ids = master[master[pheno_col].notna()].index
    valid_ids = valid_ids.intersection(subj_bias_mat.index)
    sub = master.loc[valid_ids].copy()

    # Rank-transform phenotype
    pheno_ranks = stats.rankdata(sub[pheno_col].values)

    # Encode confounds
    conf_df = pd.DataFrame(index=valid_ids)
    if "n_mutations" in confounds:
        conf_df["n_mutations"] = sub["n_mutations"]
    if "sex" in confounds and "sex" in sub.columns:
        conf_df["sex_male"] = (sub["sex"] == "Male").astype(int)
    if "cohort" in confounds:
        conf_df["cohort_spark"] = (sub["cohort"] == "SPARK").astype(int)
    conf_df = conf_df.fillna(0)
    X_conf = add_constant(conf_df.values)

    # Residualize phenotype
    pheno_resid = OLS(pheno_ranks, X_conf).fit().resid

    # Residualize each structure's bias
    bias_vals = subj_bias_mat.loc[valid_ids].values
    partial_rhos = []
    for j in range(bias_vals.shape[1]):
        bias_ranks = stats.rankdata(bias_vals[:, j])
        bias_resid = OLS(bias_ranks, X_conf).fit().resid
        r, p = stats.pearsonr(pheno_resid, bias_resid)
        partial_rhos.append({"structure": structures[j], "partial_rho": r, "pvalue": p})

    return pd.DataFrame(partial_rhos).set_index("structure")

# %%
# Compare raw vs partial for top phenotypes
for pheno_col in ["rbsr_total", "dcdq_total", "vine_abc", "srs_total_t"]:
    if pheno_col not in all_corr_results:
        continue
    raw = all_corr_results[pheno_col]["rho"]
    partial = partial_correlation_structure(
        pheno_col, confounds=["n_mutations", "sex", "cohort"]
    )["partial_rho"]

    common = raw.index.intersection(partial.index)
    r, p = stats.spearmanr(raw[common], partial[common])
    print(f"{phenotype_cols[pheno_col]}: raw vs partial ρ correlation = {r:.3f} (p={p:.2e})")
```

**Step 2: Sync and commit**

```bash
jupytext --sync notebook_phenotype/03.Phenotype_Brain_Mapping.py
git add notebook_phenotype/03.Phenotype_Brain_Mapping.py notebook_phenotype/03.Phenotype_Brain_Mapping.ipynb
git commit -m "Add continuous phenotype-brain mapping notebook (Spearman correlations + heatmap)"
```

---

## Task 9: Notebook 04 — Data-Driven Phenotype Subtypes

**Files:**
- Create: `notebook_phenotype/04.Phenotype_Subtypes.py`

**Step 1: Create notebook**

Write `notebook_phenotype/04.Phenotype_Subtypes.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 04: Data-Driven Phenotype Subtypes (Exploratory)
#
# PCA/NMF on phenotype matrix to discover phenotype dimensions,
# then map each component to brain bias patterns.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os, yaml, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.impute import SimpleImputer

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ExpZ2Mat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")

SEED = 42
FIG_DIR = "../results/phenotype/figs"

# %%
master = pd.read_parquet("../results/phenotype/mutation_phenotype_master.parquet")
with open("../results/phenotype/subject_gene_weights.pkl", "rb") as f:
    subject_gene_weights = pickle.load(f)

# Load subject-structure bias matrix (from NB03)
subj_bias_mat = pd.read_parquet("../results/phenotype/cache/subject_structure_bias_matrix.parquet")

# %% [markdown]
# ## Section 1: Build Phenotype Matrix
#
# Select subjects with at least 3 phenotype domains completed.

# %%
# Phenotype columns for PCA (summary scores only — no subscales to avoid collinearity)
pca_cols = ["rbsr_total", "dcdq_total", "vine_abc", "srs_total_t",
            "iq_fsiq", "milestone_words_mos"]
pca_labels = ["RBS-R", "DCDQ Motor", "Vineland ABC", "SRS Social",
              "IQ", "Word Delay"]

# Count available phenotypes per subject
available = master[pca_cols].notna().sum(axis=1)
print("Phenotypes available per subject:")
print(available.value_counts().sort_index())

# Require at least 3 phenotypes
min_pheno = 3
eligible = master[available >= min_pheno].index
eligible = eligible.intersection(subj_bias_mat.index)
print(f"\nSubjects with >= {min_pheno} phenotypes AND bias data: {len(eligible)}")

# %% [markdown]
# ## Section 2: PCA on Phenotype Matrix

# %%
pheno_mat = master.loc[eligible, pca_cols].copy()

# Impute missing values with column median
imputer = SimpleImputer(strategy="median")
pheno_imputed = pd.DataFrame(
    imputer.fit_transform(pheno_mat),
    index=pheno_mat.index, columns=pheno_mat.columns
)

# Standardize
scaler = StandardScaler()
pheno_scaled = pd.DataFrame(
    scaler.fit_transform(pheno_imputed),
    index=pheno_imputed.index, columns=pheno_imputed.columns
)

# PCA
pca = PCA(n_components=min(len(pca_cols), len(eligible)), random_state=SEED)
pca_scores = pca.fit_transform(pheno_scaled)
pca_scores_df = pd.DataFrame(
    pca_scores, index=eligible,
    columns=[f"PC{i+1}" for i in range(pca_scores.shape[1])]
)

print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("Cumulative:", np.round(np.cumsum(pca.explained_variance_ratio_), 3))

# %%
# Plot loadings
loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=pca_labels,
    columns=[f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})" for i in range(4)]
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(loadings, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("PCA Loadings: Phenotype Dimensions")
fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/pca_loadings.pdf", transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 3: Map PC Scores to Brain Bias
#
# For each PC, correlate PC scores with per-structure bias across subjects.

# %%
n_pcs = min(4, pca_scores.shape[1])
structures = ExpZ2Mat.columns.tolist()

pc_brain_corr = {}
for pc_idx in range(n_pcs):
    pc_name = f"PC{pc_idx+1}"
    pc_vals = pca_scores_df[pc_name].values
    bias_vals = subj_bias_mat.loc[eligible].values

    rhos = []
    pvals = []
    for j in range(bias_vals.shape[1]):
        r, p = stats.spearmanr(bias_vals[:, j], pc_vals)
        rhos.append(r)
        pvals.append(p)

    from statsmodels.stats.multitest import multipletests
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    pc_brain_corr[pc_name] = pd.DataFrame({
        "structure": structures,
        "rho": rhos,
        "pvalue": pvals,
        "qvalue": qvals,
    }).set_index("structure")

    n_sig = (qvals < 0.05).sum()
    print(f"{pc_name}: {n_sig} structures at q<0.05")

# %% [markdown]
# ### PC-Brain Heatmap

# %%
rho_mat = pd.DataFrame({k: v["rho"] for k, v in pc_brain_corr.items()}).T

# Order by region
ordered = []
for reg in ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB"]:
    reg_strs = [s for s in structures if str2reg.get(s) == reg]
    ordered.extend(sorted(reg_strs))
ordered.extend([s for s in structures if s not in ordered])

fig, ax = plt.subplots(figsize=(24, n_pcs * 0.8 + 1))
sns.heatmap(rho_mat[ordered], cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
            ax=ax, xticklabels=True,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.5})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
ax.set_title("PC-Brain Correlation Map")
fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/pc_brain_heatmap.pdf", transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 4: Robustness — SPARK vs SSC Consistency

# %%
for cohort in ["SPARK", "SSC"]:
    sub = master.loc[eligible]
    cohort_ids = sub[sub["cohort"] == cohort].index
    if len(cohort_ids) < 20:
        print(f"{cohort}: only {len(cohort_ids)} subjects, skipping")
        continue
    pheno_sub = pheno_scaled.loc[cohort_ids]
    pca_sub = PCA(n_components=min(4, len(pca_cols)), random_state=SEED)
    pca_sub.fit(pheno_sub)
    print(f"{cohort} (N={len(cohort_ids)}):")
    print(f"  Variance explained: {np.round(pca_sub.explained_variance_ratio_, 3)}")
    # Compare loadings
    loadings_sub = pca_sub.components_[:min(3, pca_sub.n_components_)]
    for i in range(loadings_sub.shape[0]):
        # Align sign
        sign = np.sign(np.dot(loadings_sub[i], pca.components_[i]))
        r = np.corrcoef(loadings_sub[i] * sign, pca.components_[i])[0, 1]
        print(f"  PC{i+1} loading correlation with full PCA: r={r:.3f}")

# Save
for pc_name, df in pc_brain_corr.items():
    df.to_csv(f"../results/phenotype/subtypes/{pc_name}_brain_corr.csv")
pca_scores_df.to_parquet("../results/phenotype/subtypes/pca_scores.parquet")
```

**Step 2: Sync and commit**

```bash
jupytext --sync notebook_phenotype/04.Phenotype_Subtypes.py
git add notebook_phenotype/04.Phenotype_Subtypes.py notebook_phenotype/04.Phenotype_Subtypes.ipynb
git commit -m "Add data-driven phenotype subtypes notebook (PCA on phenotype matrix, map to brain)"
```

---

## Task 10: Notebook 05 — SSC Cross-Validation

**Files:**
- Create: `notebook_phenotype/05.SSC_Validation.py`

**Step 1: Create notebook**

Write `notebook_phenotype/05.SSC_Validation.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 05: SSC Cross-Validation
#
# Replicate SPARK-derived phenotype-brain findings using SSC's
# gold-standard clinician-administered data. Also explores SSC-unique
# instruments (ADI-R, ADOS item-level, SRS-teacher).

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys, os, yaml, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, STR2Region, MouseSTR_AvgZ_Weighted,
    Mut2GeneDF, Filt_LGD_Mis,
)

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
str2reg = STR2Region()

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
ExpZ2Mat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")

SEED = 42
FIG_DIR = "../results/phenotype/figs"
SSC_DIR = "../dat/Phenotype/SSC_Phenotype_Dataset/SSC_V15_Phenotype_DATA/Proband_Data"

# %%
master = pd.read_parquet("../results/phenotype/mutation_phenotype_master.parquet")
subj_bias_mat = pd.read_parquet("../results/phenotype/cache/subject_structure_bias_matrix.parquet")

ssc_master = master[master["cohort"] == "SSC"]
spark_master = master[master["cohort"] == "SPARK"]
print(f"SSC subjects: {len(ssc_master)}")
print(f"SPARK subjects: {len(spark_master)}")

structures = ExpZ2Mat.columns.tolist()

# %% [markdown]
# ## Section 1: Replication — Core Phenotype-Brain Correlations
#
# Compare SPARK-only vs SSC-only Spearman correlations for shared instruments.

# %%
shared_phenos = ["rbsr_total", "dcdq_total", "vine_abc", "srs_total_t", "iq_fsiq"]
pheno_labels = ["RBS-R", "DCDQ Motor", "Vineland ABC", "SRS Social", "IQ"]

replication_results = {}
for pheno_col, label in zip(shared_phenos, pheno_labels):
    rho_spark = []
    rho_ssc = []
    for cohort, sub, rho_list in [("SPARK", spark_master, rho_spark),
                                   ("SSC", ssc_master, rho_ssc)]:
        valid = sub[sub[pheno_col].notna()].index.intersection(subj_bias_mat.index)
        if len(valid) < 10:
            rho_list.extend([np.nan] * len(structures))
            continue
        pheno_vals = sub.loc[valid, pheno_col].values
        bias_vals = subj_bias_mat.loc[valid].values
        for j in range(len(structures)):
            r, _ = stats.spearmanr(bias_vals[:, j], pheno_vals)
            rho_list.append(r)

    replication_results[pheno_col] = {
        "spark_rho": np.array(rho_spark),
        "ssc_rho": np.array(rho_ssc),
    }

    # Cross-cohort correlation
    mask = ~(np.isnan(rho_spark) | np.isnan(rho_ssc))
    if mask.sum() > 10:
        r, p = stats.spearmanr(np.array(rho_spark)[mask], np.array(rho_ssc)[mask])
        n_spark = spark_master[pheno_col].notna().sum()
        n_ssc = ssc_master[pheno_col].notna().sum()
        print(f"{label}: SPARK(N={n_spark}) vs SSC(N={n_ssc}) "
              f"cross-cohort ρ = {r:.3f} (p={p:.2e})")

# %% [markdown]
# ### Replication Scatter Plots

# %%
fig, axes = plt.subplots(1, len(shared_phenos), figsize=(4 * len(shared_phenos), 4))
for idx, (pheno_col, label) in enumerate(zip(shared_phenos, pheno_labels)):
    ax = axes[idx]
    res = replication_results[pheno_col]
    mask = ~(np.isnan(res["spark_rho"]) | np.isnan(res["ssc_rho"]))
    if mask.sum() < 10:
        ax.text(0.5, 0.5, f"{label}\nInsufficient data", ha="center", va="center")
        continue

    ax.scatter(res["spark_rho"][mask], res["ssc_rho"][mask], alpha=0.3, s=10)
    r, p = stats.spearmanr(res["spark_rho"][mask], res["ssc_rho"][mask])
    ax.set_xlabel("SPARK ρ")
    ax.set_ylabel("SSC ρ")
    ax.set_title(f"{label}\nr={r:.2f}, p={p:.1e}")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.patch.set_alpha(0)

fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/ssc_replication_scatter.pdf",
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 2: SSC-Unique Instruments
#
# Explore ADI-R and SRS-teacher (not available in SPARK).

# %%
# ADI-R: clinician-administered autism diagnostic interview
adi_r = pd.read_csv(f"{SSC_DIR}/ssc_core_descriptive.csv")
adi_cols = {
    "adi_r_soc_a_total": "ADI-R Social (A)",
    "adi_r_b_comm_verbal_total": "ADI-R Communication Verbal (B)",
    "adi_r_rrb_c_total": "ADI-R RRB (C)",
}

ssc_ids = ssc_master.index.intersection(subj_bias_mat.index)
adi_data = adi_r.set_index("individual").loc[
    adi_r.set_index("individual").index.intersection(ssc_ids)
]

for adi_col, label in adi_cols.items():
    vals = adi_data[adi_col].dropna()
    common = vals.index.intersection(subj_bias_mat.index)
    if len(common) < 10:
        print(f"{label}: only {len(common)} subjects, skipping")
        continue

    pheno_vals = vals.loc[common].values
    bias_vals = subj_bias_mat.loc[common].values

    rhos = []
    pvals = []
    for j in range(len(structures)):
        r, p = stats.spearmanr(bias_vals[:, j], pheno_vals)
        rhos.append(r)
        pvals.append(p)

    n_sig = sum(1 for p in pvals if p < 0.05)
    top_idx = np.argmin(pvals)
    print(f"{label} (N={len(common)}): {n_sig} structures p<0.05, "
          f"top={structures[top_idx]} (ρ={rhos[top_idx]:.3f}, p={pvals[top_idx]:.4f})")

# %%
# SRS Teacher report (SSC only)
srs_teacher = pd.read_csv(f"{SSC_DIR}/srs_teacher.csv")
srs_teacher = srs_teacher.set_index("individual")
teacher_common = srs_teacher.index.intersection(ssc_ids)
teacher_common = teacher_common.intersection(subj_bias_mat.index)

if len(teacher_common) >= 10:
    t_vals = srs_teacher.loc[teacher_common, "t_score"].dropna()
    t_common = t_vals.index.intersection(subj_bias_mat.index)
    pheno_v = t_vals.loc[t_common].values
    bias_v = subj_bias_mat.loc[t_common].values

    # Compare parent vs teacher SRS
    p_vals_parent = srs_teacher.loc[t_common].index.map(
        lambda x: ssc_master.loc[x, "srs_total_t"] if x in ssc_master.index else np.nan
    )
    valid_both = ~(np.isnan(p_vals_parent) | np.isnan(pheno_v))
    if valid_both.sum() > 10:
        r_pt, _ = stats.spearmanr(
            np.array(p_vals_parent)[valid_both],
            pheno_v[valid_both]
        )
        print(f"\nSRS Parent-Teacher correlation: r={r_pt:.3f} (N={valid_both.sum()})")

    # Teacher SRS → brain correlations
    rhos_t = [stats.spearmanr(bias_v[:, j], pheno_v).statistic for j in range(len(structures))]
    n_sig_t = sum(1 for r in rhos_t if abs(r) > 0.2)
    print(f"SRS Teacher (N={len(t_common)}): {n_sig_t} structures with |ρ|>0.2")

# Save
pd.DataFrame(replication_results).to_pickle(
    "../results/phenotype/ssc_validation/replication_results.pkl"
)
print("\nSaved SSC validation results")
```

**Step 2: Sync and commit**

```bash
jupytext --sync notebook_phenotype/05.SSC_Validation.py
git add notebook_phenotype/05.SSC_Validation.py notebook_phenotype/05.SSC_Validation.ipynb
git commit -m "Add SSC cross-validation notebook (replication + ADI-R/SRS-teacher)"
```

---

## Task 11: Run Notebook 01, Review Results, Adjust Downstream

**Step 1: Execute notebook 01**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType/notebook_phenotype
conda activate gencic
jupyter nbconvert --to notebook --execute --inplace 01.Phenotype_Data_Cleaning.ipynb --ExecutePreprocessor.timeout=600
```

**Step 2: Verify output**

```bash
python -c "
import pandas as pd
m = pd.read_parquet('../results/phenotype/mutation_phenotype_master.parquet')
print(f'Shape: {m.shape}')
print(f'Cohorts: {m[\"cohort\"].value_counts().to_dict()}')
print(f'Non-null per column:')
print(m.notna().sum().sort_values(ascending=False).head(20))
"
```

**Step 3: Adjust notebooks 02-05 if column names or sample sizes differ from expectations**

Review the actual coverage numbers. If any phenotype has < 30 subjects after filtering, skip it in the stratification notebook. If column names differ, update the mapping dicts.

---

## Task 12: Run Notebooks 02-05 Sequentially

**Step 1: Run notebook 02 (stratification — most compute-intensive due to permutations)**

```bash
jupyter nbconvert --to notebook --execute --inplace 02.Phenotype_Stratification.ipynb --ExecutePreprocessor.timeout=7200
```

**Step 2: Run notebook 03 (continuous mapping)**

```bash
jupyter nbconvert --to notebook --execute --inplace 03.Phenotype_Brain_Mapping.ipynb --ExecutePreprocessor.timeout=7200
```

**Step 3: Run notebook 04 (subtypes)**

```bash
jupyter nbconvert --to notebook --execute --inplace 04.Phenotype_Subtypes.ipynb --ExecutePreprocessor.timeout=600
```

**Step 4: Run notebook 05 (SSC validation)**

```bash
jupyter nbconvert --to notebook --execute --inplace 05.SSC_Validation.ipynb --ExecutePreprocessor.timeout=600
```

**Step 5: Final commit**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType
git add notebook_phenotype/*.py notebook_phenotype/*.ipynb
git commit -m "Execute all phenotype notebooks, cache results"
```

---

## Task 13: Review and Iterate

After running all notebooks:

1. **Check coverage table** from NB01: are sample sizes sufficient?
2. **Check stratification summary** from NB02: which phenotypes show significant structure-level differences?
3. **Check heatmap** from NB03: are there phenotype-specific brain patterns?
4. **Check PCA loadings** from NB04: do components make sense (social vs motor vs RRB)?
5. **Check replication** from NB05: do SPARK findings hold in SSC?

If results are promising, potential next steps:
- Extract top figures for publication
- Add cell-type level analysis (cluster Z2 matrix)
- Run phenotype-specific circuit search (SA with phenotype-stratified gene weights)
