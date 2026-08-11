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
# # Notebook 02 — Structure-Level Bias and Recovery (PD / HD validation)
#
# Structure-level arm of the GENCIC PD / striatal-degeneration circuit validation, written in
# response to Reviewer 2. This notebook is the clean, sectioned, reproducible consolidation of
# the exploratory analysis in `notebooks_disease_validation/reference/` (`threenull.py`,
# `top_str.py`, `summary.py`, `ccs_sib.py`, `ccs_matched.py`, `union.py` — see
# `reference/README.md` for the full script-to-notebook mapping). It does not re-derive any
# science: every number below is either read directly from pipeline outputs already on disk
# (`results/STR_ISH/`, read-only) or recomputed deterministically from them.
#
# **Scope.** Structure-level bias, recovery under four null models, top-ranked structures,
# circuit connectivity (CCS) under size-matched nulls, and a gene-set union analysis. The
# cell-type arm is notebook 03 and the pathology / circuit-refinement arm (including the blinded,
# anatomy-matched pathology evaluation) is notebook 04 — both out of scope here.
#
# **Contents**
# 1. Setup
# 2. Pre-registered gene sets and ground truth
# 3. Structure-level bias (6 pre-registered sets + the post-hoc high-confidence tier)
# 4. Recovery statistics under four null models
# 5. Top-ranked structures per PD gene set
# 6. CCS profiles under size-matched nulls
# 7. Gene-set union analysis
# 8. Consolidated summary table
# 9. Verification

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # 1. Setup

# %%
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from joblib import Parallel, delayed

sys.path.insert(1, "../src")
from ASD_Circuits import (
    LoadGeneINFO,
    STR2Region,
    Fil2Dict,
    MouseSTR_AvgZ_Weighted,
    MouseCT_AvgZ_Weighted,
    ScoreCircuit_SI_Joint,
)
from disease_validation import (
    load_gene_sets,
    load_ground_truth,
    recovery_stats,
    recovery_null_aurocs,
    empirical_p,
)
from plot import REGION_COLORS

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

SEED = 42
N_JOBS = 10
np.random.seed(SEED)
# No stochastic draws happen anywhere in this notebook: every null simulation was generated
# upstream, once, by the seeded Snakemake bias pipeline (script_generate_geneweights.py). This
# notebook only ever reads those simulations back from disk. SEED is set here in defence of depth
# and to follow house convention, not because anything below consumes it.

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

STR_EXPR_MATRIX = f"../{config['analysis_types']['STR_ISH']['expr_matrix']}"
CT_EXPR_MATRIX = f"../{config['analysis_types']['CT_Z2']['expr_matrix']}"
INFOMAT_PATH = f"../{config['data_files']['infomat_ipsi']}"
RANKSCORE_PATH = f"../{config['data_files']['rankscore_ipsi']}"
CROSS_PLATFORM_CORR_PATH = f"../{config['data_files']['gene_cross_platform_corr']}"

# results/STR_ISH and results/CT_Z2 hold the null distributions behind a manuscript under review:
# READ ONLY, everywhere in this notebook. The only results/ tree this notebook writes to is
# results/PD_HD_validation/ (figures/tables/cache), created fresh below if missing.
STR_RESULTS_DIR = "../results/STR_ISH"
CT_RESULTS_DIR = "../results/CT_Z2"
OUT_DIR = "../results/PD_HD_validation"
FIG_DIR = f"{OUT_DIR}/figures"
TABLE_DIR = f"{OUT_DIR}/tables"
CACHE_DIR = f"{OUT_DIR}/cache"
for d in (FIG_DIR, TABLE_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

ANNO = STR2Region()
HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

SUBCLASS_RE = re.compile(r"^\d+\s+")


def subclass_of_cluster(cluster_ids):
    """Cluster ID ('0460 CT SUB Glut_2') -> bare subclass name ('CT SUB Glut').

    Matches the convention frozen in config/disease_validation_ground_truth.yaml and checked by
    tests/test_disease_validation_data.py::test_frozen_cell_type_subclasses_map_to_clusters.
    """
    return pd.Series([SUBCLASS_RE.sub("", c).rsplit("_", 1)[0] for c in cluster_ids], index=cluster_ids)


def clusters_in_subclasses(cluster_ids, subclass_names):
    sub = subclass_of_cluster(cluster_ids)
    return sub.index[sub.isin(subclass_names)].tolist()


print(f"STR_ISH expression matrix: {STR_EXPR_MATRIX}")
print(f"CT_Z2 expression matrix:   {CT_EXPR_MATRIX}")
print(f"Writable output tree:      {OUT_DIR}")

# %% [markdown]
# # 2. Pre-registered gene sets and ground truth
#
# `config/disease_validation_genesets.yaml` and `config/disease_validation_ground_truth.yaml`
# were frozen together, before any bias was computed. The commit hash printed below is the
# citation for the rebuttal.

# %%
GENESETS_YAML = "../config/disease_validation_genesets.yaml"
GROUND_TRUTH_YAML = "../config/disease_validation_ground_truth.yaml"

gene_sets = load_gene_sets(GENESETS_YAML)   # the 6 pre-registered sets, expanded from the pooled YAML
GT = load_ground_truth(GROUND_TRUTH_YAML)
GT_KEY = dict(GT["notes"]["gene_sets_to_ground_truth"])  # {'PD_Primary': 'parkinson', ...}

preregistration_commit = subprocess.run(
    ["git", "log", "-1", "--format=%H", "--",
     "config/disease_validation_genesets.yaml", "config/disease_validation_ground_truth.yaml"],
    cwd="..", capture_output=True, text=True,
).stdout.strip()
print(f"Pre-registration commit (frozen BEFORE any bias was computed): {preregistration_commit}")

assert set(gene_sets) == {"PD_Primary", "PD_Sens_DA", "PD_Sens_Atypical",
                          "PD_GWAS_L2G", "HD_HTT", "StriatalDegeneration"}, \
    "the frozen pre-registration must define exactly these six gene sets"

# %% [markdown]
# ## 2.1 The post-hoc high-confidence tier
#
# `PD_HighConf` / `PD_HighConf_DA` are **not** part of the pre-registration above. They were
# added one day later (literature-derived, disclosed as post-hoc) by applying a stricter
# inclusion rule — ClinGen "Definitive" or undisputed established Mendelian PD, excluding
# disputed / contested / non-nigrostriatal genes — to a literature table
# (`results/tables/PD_gene_literature_evidence.csv`) that was itself built before any expression
# result was computed. They are subsets of the same PD gene space, evaluated against the same
# frozen `parkinson` ground truth; they are not a new ground truth, and (correctly) do not appear
# in `GT_KEY` because they were never pre-registered.

# %%
highconf_commit = subprocess.run(
    ["git", "log", "-1", "--format=%H %s", "--", "config/config.STR.yaml"],
    cwd="..", capture_output=True, text=True,
).stdout.strip()
print(f"Commit that adds the post-hoc PD_HighConf tier: {highconf_commit}")

GT_KEY = {**GT_KEY, "PD_HighConf": "parkinson", "PD_HighConf_DA": "parkinson"}
ALL_SETS = ["PD_Primary", "PD_Sens_DA", "PD_Sens_Atypical", "PD_GWAS_L2G", "HD_HTT",
            "StriatalDegeneration", "PD_HighConf", "PD_HighConf_DA"]
PREREGISTERED = set(gene_sets)

geneset_meta = pd.DataFrame({
    "gene_set": ALL_SETS,
    "n_genes": [len(pd.read_csv(f"../dat/Genetics/GeneWeights/{s}.gw", header=None)) for s in ALL_SETS],
    "disease": ["HD" if GT_KEY[s] == "striatal" else "PD" for s in ALL_SETS],
    "ground_truth": [GT_KEY[s] for s in ALL_SETS],
    "pre_registered": [s in PREREGISTERED for s in ALL_SETS],
}).set_index("gene_set")
geneset_meta

# %% [markdown]
# # 3. Structure-level bias across the pre-registered sets + the high-confidence tier
#
# Bias is read directly from the Snakemake bias pipeline's output
# (`results/STR_ISH/{gene_set}_bias_addP_random.csv`), never recomputed.
#
# **Key point, relied on for the rest of the notebook:** `EFFECT` is the weighted-average Z2 bias
# of the *real* gene set on each structure. It does not depend on which null model was configured
# for that pipeline run — the null only determines the population of *simulated* gene sets used
# to compute `P-value` / `q-value`. Demonstrated concretely below: for the same gene set, `EFFECT`
# is byte-identical between the random-null and sibling-null bias files, while `q-value` differs.
# **A comparison of "recovery under null A vs null B" using EFFECT alone would therefore be
# comparing nothing.** Section 4 uses the statistic that actually differs between nulls: the rank
# of the observed AUROC against 10,000 null *gene sets*, not the EFFECT value itself.

# %%
BIAS = {s: pd.read_csv(f"{STR_RESULTS_DIR}/{s}_bias_addP_random.csv", index_col=0) for s in ALL_SETS}

# Concrete demonstration of the EFFECT-is-null-independent point. PD_Sens_DA is the only
# pre-registered set with q<0.10 hits under every null (Section 4), which makes it the clearest
# example that only q-value moves.
_sib = pd.read_csv(f"{STR_RESULTS_DIR}/PD_Sens_DA_bias_addP_sibling.csv", index_col=0)
_max_deffect = (BIAS["PD_Sens_DA"]["EFFECT"] - _sib.loc[BIAS["PD_Sens_DA"].index, "EFFECT"]).abs().max()
_top_struct = BIAS["PD_Sens_DA"]["EFFECT"].idxmax()
print(f"PD_Sens_DA: max|EFFECT(random-null file) - EFFECT(sibling-null file)| = {_max_deffect:.2e}")
print(f"  top structure {_top_struct!r}: q-value(random)={BIAS['PD_Sens_DA'].loc[_top_struct, 'q-value']:.4f}"
      f"  vs  q-value(sibling)={_sib.loc[_top_struct, 'q-value']:.4f}   <- only THIS differs")
assert _max_deffect == 0.0, "EFFECT must be null-independent -- see disease_validation.py docstring"

overview = pd.DataFrame({
    "n_genes": geneset_meta["n_genes"],
    "top_structure": [BIAS[s]["EFFECT"].idxmax() for s in ALL_SETS],
    "top_EFFECT": [round(BIAS[s]["EFFECT"].max(), 3) for s in ALL_SETS],
    "n_q<0.10 (random null)": [int((BIAS[s]["q-value"] < 0.10).sum()) for s in ALL_SETS],
})
overview

# %% [markdown]
# # 4. Recovery statistics under four null models
#
# Four null models test different questions about the **same** observed AUROC (it does not move
# across nulls — Section 3):
#
# - **random (uniform)** — null gene sets drawn uniformly from all expressed genes.
# - **expression-matched (EM)** — null gene sets drawn to match the real set's per-decile
#   expression composition (robustness check: is the signal just "these are expressed genes"?).
# - **sibling (uniform)** — null gene sets drawn uniformly from the SPARK sibling (unaffected)
#   mutation pool.
# - **sibling (mutability = ASD)** — null gene sets drawn from the sibling pool weighted by
#   mutability: the *exact* null procedure used for the published `ASD_All` result.
#
# For each null, `p_*` = `empirical_p(observed_AUROC, recovery_null_aurocs(null_bias_df,
# ground_truth))` — the fraction of the 10,000 null *gene sets* whose own ground-truth AUROC is
# `>=` the real gene set's.
#
# `PD_HighConf` / `PD_HighConf_DA` have no sibling-uniform null registered (`config/config.STR.yaml`
# only registers `random`, `_EM`, and `_SibMut` for the post-hoc tier), so `p_sibling_uniform` /
# `q10_sibling_uniform` are `NaN` for those two rows **by design**.
#
# `HD_HTT` (n=1 gene) and `StriatalDegeneration` (n=8 genes) trip `recovery_null_aurocs`'s NaN
# guard under every null: some of the 10,000 null gene sets are so small that a background
# structure ends up with an entirely-NaN weighted average for that particular draw, and the
# function refuses to silently return a p-value built on propagated NaN (see its docstring). Both
# sets are non-significant under `p_mannwhitney` regardless — a reported limitation, not a
# suppressed result.

# %%
NULL_VARIANTS = [
    ("p_random", "{s}", "random"),
    ("p_exprmatched", "{s}_EM", "random"),
    ("p_sibling_uniform", "{s}", "sibling"),
    ("p_sibling_mutability", "{s}_SibMut", "sibling"),
]


def four_null_recovery_row(s):
    gt_core = GT["structures"][GT_KEY[s]]["core"]
    st = recovery_stats(BIAS[s], gt_core)
    row = {
        "n_genes": int(geneset_meta.loc[s, "n_genes"]),
        "ground_truth": GT_KEY[s],
        "n_gt_present": st["n_ground_truth"],
        "AUROC": st["auroc"],
        "p_mannwhitney": st["p_mannwhitney"],
        "median_rank": st["median_rank"],
        "precision_at_20": st["precision_at_20"],
    }
    for label, pat, kind in NULL_VARIANTS:
        geneset_name = pat.format(s=s)
        bias_f = f"{STR_RESULTS_DIR}/{geneset_name}_bias_addP_{kind}.csv"
        null_f = f"{STR_RESULTS_DIR}/null_bias/{geneset_name}_null_bias_{kind}.parquet"
        qcol = label.replace("p_", "q10_")
        if not os.path.exists(bias_f):
            row[label], row[qcol] = np.nan, np.nan
            continue
        d = pd.read_csv(bias_f, index_col=0)
        row[qcol] = int((d["q-value"] < 0.10).sum())
        if not os.path.exists(null_f):
            row[label] = np.nan
            continue
        try:
            row[label] = empirical_p(st["auroc"], recovery_null_aurocs(pd.read_parquet(null_f), gt_core))
        except ValueError:
            row[label] = np.nan  # NaN guard fired -- see markdown above, both offending sets are ns anyway
    return row


recovery_table = pd.DataFrame({s: four_null_recovery_row(s) for s in ALL_SETS}).T
recovery_table.index.name = "gene_set"
recovery_table = recovery_table.round(4)
recovery_table

# %% [markdown]
# ## 4.1 Headline negative result: the pre-registered composite fails
#
# `PD_HighConf_DA` is the tightest, best-curated, non-circular-adjacent PD tier (19 genes: the 14
# ClinGen-definitive/established-Mendelian genes plus the 5 dopamine-synthesis/transport markers).
# Under the sibling-mutability null — the null that mirrors the published ASD procedure — its
# recovery of the 13 pre-registered PD-core structures as a single composite **fails**
# (p ≈ 0.11, not significant at any conventional threshold).
#
# This is not a failure of the framework. The pre-registered core list conflates two anatomically
# distinct things: structures where dopaminergic *neurons* degenerate (SNc, VTA) and structures
# that are *denervated* downstream of that loss but whose own neurons survive (striatum, pallidum,
# STN). GENCIC's structure-level bias tracks gene-expressing neurons, not axon terminal loss, so
# it should not — and does not — treat those two classes the same way. Caudoputamen ranks ~190/213
# in `PD_Primary` (the non-circular headline set): **that is neuropathologically correct**. The
# striatum loses its dopaminergic input in PD, but the medium spiny neurons that make up the
# structure are not the ones that degenerate.

# %%
cp_rank = int(BIAS["PD_Primary"]["EFFECT"].rank(ascending=False)["Caudoputamen"])
cp_effect = BIAS["PD_Primary"].loc["Caudoputamen", "EFFECT"]
print(f"Caudoputamen in PD_Primary: rank {cp_rank}/213, EFFECT={cp_effect:.3f} (strongly depleted, "
      "consistent with denervation-without-neuron-loss)")

p_highconf_da_sibmut = recovery_table.loc["PD_HighConf_DA", "p_sibling_mutability"]
q10_highconf_da_sibmut = recovery_table.loc["PD_HighConf_DA", "q10_sibling_mutability"]
print(f"\nPD_HighConf_DA (n={int(geneset_meta.loc['PD_HighConf_DA', 'n_genes'])}) vs the 13-structure "
      f"pre-registered PD-core composite:")
print(f"  AUROC = {recovery_table.loc['PD_HighConf_DA', 'AUROC']:.3f}")
print(f"  sibling-mutability null p_geneset = {p_highconf_da_sibmut:.4f}  <- FAILS to reach p<0.05")
print(f"  structures at q<0.10 under that same sibling-mutability null: {int(q10_highconf_da_sibmut)}")

assert 0.10 < p_highconf_da_sibmut < 0.13, (
    f"headline negative result changed: PD_HighConf_DA sibling-mutability p was "
    f"{p_highconf_da_sibmut}, expected in the open interval (0.10, 0.13)")

# %% [markdown]
# # 5. Top-ranked structures per PD gene set
#
# Top-20 structures by `EFFECT` for every PD-labelled gene set (the two Mendelian tiers, the
# GWAS tier, and the two post-hoc high-confidence tiers), annotated with the random-null and
# sibling-mutability-null q-values, the major brain region (`STR2Region()`), and whether the
# structure is in the pre-registered ground truth (PD core, or Braak-early — only partially
# testable; three of its four structures are absent from the 213-structure atlas, see
# `config/disease_validation_ground_truth.yaml`).

# %%
PD_SETS = ["PD_Primary", "PD_Sens_DA", "PD_Sens_Atypical", "PD_GWAS_L2G", "PD_HighConf", "PD_HighConf_DA"]
PD_CORE = set(GT["structures"]["parkinson"]["core"])
PD_BRAAK = set(GT["structures"]["parkinson"]["braak_early"])


def top_n_table(s, n=20):
    d = BIAS[s].sort_values("EFFECT", ascending=False).head(n).copy()
    sib_f = f"{STR_RESULTS_DIR}/{s}_SibMut_bias_addP_sibling.csv"
    d["q_sibling_mutability"] = (pd.read_csv(sib_f, index_col=0).loc[d.index, "q-value"]
                                 if os.path.exists(sib_f) else np.nan)
    d["ground_truth"] = ["PD core" if st in PD_CORE else ("Braak-early" if st in PD_BRAAK else "")
                         for st in d.index]
    return d[["EFFECT", "q-value", "q_sibling_mutability", "REGION", "ground_truth"]].rename(
        columns={"q-value": "q_random"})


TOP20 = {s: top_n_table(s, 20) for s in PD_SETS}

top20_export = pd.concat(TOP20, names=["gene_set", "structure"]).reset_index()
top20_export.to_csv(f"{TABLE_DIR}/PD_HD_validation_top20_structures.csv", index=False)
print(f"Full top-20 tables (6 sets x 20 structures) -> {TABLE_DIR}/PD_HD_validation_top20_structures.csv")

pd.concat({s: TOP20[s].head(5) for s in PD_SETS}, names=["gene_set", "structure"])

# %% [markdown]
# ## 5.1 Figure: top-20 structures per PD gene set, coloured by region

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for ax, s in zip(axes.flat, PD_SETS):
    d = TOP20[s].iloc[::-1]  # reverse so rank 1 plots at the top of the horizontal bar chart
    colors = [REGION_COLORS.get(r, "#999999") for r in d["REGION"]]
    ylabels = [f"{st}  [{gt}]" if gt else st for st, gt in zip(d.index, d["ground_truth"])]
    ax.barh(range(len(d)), d["EFFECT"], color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_title(f"{s}  (n={int(geneset_meta.loc[s, 'n_genes'])} genes)", fontsize=11)
    ax.set_xlabel("EFFECT (weighted Z2 structure bias)", fontsize=9)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_facecolor("none")

seen = set()
region_order = [r for tbl in TOP20.values() for r in tbl["REGION"] if not (r in seen or seen.add(r))]
handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS.get(r, "#999999")) for r in region_order]
fig.legend(handles, region_order, loc="lower center", ncol=len(region_order), fontsize=8,
          frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Top-20 structures by structure-level bias (EFFECT), coloured by brain region\n"
            "[PD core] / [Braak-early] = pre-registered ground truth", y=1.01, fontsize=12)
fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/top20_structures_per_PD_geneset.png",
           transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5.2 The consistent source/target pattern, and the PD_HighConf_DA top 3

# %%
for s in PD_SETS:
    top3 = BIAS[s]["EFFECT"].nlargest(3).index.tolist()
    print(f"{s:16s} top 3: {top3}")

top3_highconf_da = BIAS["PD_HighConf_DA"]["EFFECT"].nlargest(3).index.tolist()
assert top3_highconf_da == ["Ventral_tegmental_area", "Dorsal_nucleus_raphe",
                            "Substantia_nigra_compact_part"], \
    f"PD_HighConf_DA top-3 structures changed: {top3_highconf_da}"

# %% [markdown]
# Every dopamine-marker-containing PD set (`PD_Sens_DA`, `PD_HighConf_DA`) puts the same three
# midbrain structures on top — Ventral tegmental area, dorsal raphe nucleus, and substantia nigra
# pars compacta — all dopaminergic/monoaminergic *source* structures. The non-marker sets
# (`PD_Primary`, `PD_Sens_Atypical`, `PD_HighConf`) surface a more diffuse mix (raphe, pontine and
# cingulate structures) because they lack the dopamine-synthesis genes that concentrate the signal
# at the source nuclei. In no PD set does a striatal *target* structure reach the top 5 — the
# same source-high / target-low asymmetry documented for Caudoputamen in Section 4.1.

# %% [markdown]
# # 6. CCS profiles under size-matched nulls
#
# Circuit Connectivity Score (`ScoreCircuit_SI_Joint`, Shannon-information connectivity of the
# top-N structures by bias) for each gene set at six circuit sizes, against **three** nulls:
#
# 1. **sibling-mutability** — CCS of the top-N structures from each of the 10,000 sibling-
#    mutability null *gene sets* (the same null gene sets used in Section 4's headline column,
#    scored for connectivity instead of recovery). Exact ASD procedure, size- and
#    composition-matched to the real gene set.
# 2. **random-uniform (matched)** — same idea, uniform-random null gene sets, size-matched.
# 3. **ASD 61-gene sibling band** (`dat/allen-mouse-conn/RankScores/RankScore.Ipsi.Cont.npy`) —
#    included **only as a labelled comparison, never as a primary null**. This array is built from
#    the *published* ASD analysis's sibling-mutability null at the ASD gene-set size (61 genes);
#    comparing an 8-40-gene PD/HD circuit against it is not size-matched. An earlier pass of this
#    analysis used this band as the *only* null and found it to be **anti-conservative**, not
#    conservative as first assumed (it made several PD sets look significant that are not,
#    once compared against their own size-matched null). It is shown below purely so a reader who
#    already knows the published ASD figure can see how different the true, matched null is —
#    never to compute a p-value that goes in the manuscript.
#
# `HD_HTT` (n=1 gene, no meaningful "top-N circuit" null) and the post-hoc high-confidence tier
# (out of scope for `ccs_sib.py`/`ccs_matched.py` — its structure-level story is already told in
# Section 5) are not included here, matching the reference scripts' exact gene-set list.

# %%
CCS_SETS = ["PD_Primary", "PD_Sens_DA", "PD_Sens_Atypical", "PD_GWAS_L2G", "StriatalDegeneration"]
SIZES = [100, 60, 46, 30, 20, 10]
ORDER = np.argsort(SIZES)          # ascending-size reordering, used only for plotting
SIZES_ASC = np.array(SIZES)[ORDER]

Info = pd.read_csv(INFOMAT_PATH, index_col=0)
ASD_band = np.load(RANKSCORE_PATH)                 # (10000, 195): ASD 61-gene band, NOT size-matched
ASD_band_topNs = np.arange(200, 5, -1)
ASD_band_idx = {N: int(np.where(ASD_band_topNs == N)[0][0]) for N in SIZES}


def ccs_profile(sorted_structs):
    return [ScoreCircuit_SI_Joint(sorted_structs[:N], Info) for N in SIZES]


NULL_KEYS = [f"{s}__{lbl}" for s in CCS_SETS for lbl in ("sibmut", "random")]
CCS_CACHE = f"{CACHE_DIR}/ccs_null_profiles.npz"
null_profiles = {}
if os.path.exists(CCS_CACHE):
    _cache = np.load(CCS_CACHE)
    null_profiles = {k: _cache[k] for k in _cache.files}
    if set(null_profiles) != set(NULL_KEYS):
        print("Cache key mismatch (gene-set/null list changed) -- recomputing.")
        null_profiles = {}
    else:
        print(f"Loaded cached null CCS profiles from {CCS_CACHE}")

if not null_profiles:
    for s in CCS_SETS:
        for null_lbl, null_f in [
            ("sibmut", f"{STR_RESULTS_DIR}/null_bias/{s}_SibMut_null_bias_sibling.parquet"),
            ("random", f"{STR_RESULTS_DIR}/null_bias/{s}_null_bias_random.parquet"),
        ]:
            nb = pd.read_parquet(null_f)
            prof = Parallel(n_jobs=N_JOBS)(
                delayed(ccs_profile)(nb[c].sort_values(ascending=False).index.values) for c in nb.columns
            )
            null_profiles[f"{s}__{null_lbl}"] = np.array(prof)
    np.savez_compressed(CCS_CACHE, **null_profiles)
    print(f"Computed null CCS profiles for {len(CCS_SETS)} gene sets x 2 nulls x 10,000 sims -> {CCS_CACHE}")

# EFFECT is null-independent (Section 3), so the random-null bias file's structure ORDER is the
# same real-gene-set order that every null was configured against.
observed = {s: np.array(ccs_profile(BIAS[s].sort_values("EFFECT", ascending=False).index.values))
           for s in CCS_SETS}

# %%
ccs_rows = []
for s in CCS_SETS:
    o = observed[s]
    for j, N in enumerate(SIZES):
        row = {"gene_set": s, "N": N, "observed_CCS": round(float(o[j]), 4)}
        for null_lbl in ("sibmut", "random"):
            null = null_profiles[f"{s}__{null_lbl}"]
            row[f"p_{null_lbl}"] = round(empirical_p(o[j], null[:, j]), 4)
        row["p_ASDband_labelled_only"] = round(empirical_p(o[j], ASD_band[:, ASD_band_idx[N]]), 4)
        ccs_rows.append(row)
ccs_table = pd.DataFrame(ccs_rows)
ccs_table.to_csv(f"{TABLE_DIR}/PD_HD_validation_CCS_profiles.csv", index=False)
ccs_table

# %% [markdown]
# `PD_Primary` and `StriatalDegeneration` are not significant at any circuit size under either
# matched null. `PD_GWAS_L2G` reaches nominal significance under the sibling-mutability null at
# most sizes — but Section 5 already showed its top-ranked structures are pontine/medullary
# reticular formation (`Pontine_reticular_nucleus`, `Red_nucleus`,
# `Magnocellular_reticular_nucleus`), none of which are in the PD core or Braak-early ground
# truth. Densely interconnected regions score high CCS regardless of disease relevance — CCS
# measures connectivity, not disease relevance, and that dissociation is stated plainly here
# rather than left implicit.

# %%
gwas_top5 = TOP20["PD_GWAS_L2G"].head(5)
gwas_top5_in_gt = set(gwas_top5.index) & (PD_CORE | PD_BRAAK)
print("PD_GWAS_L2G top 5 structures:", gwas_top5.index.tolist())
print("...of which in the PD ground truth (core or Braak-early):", gwas_top5_in_gt or "NONE")
assert gwas_top5_in_gt == set(), \
    "PD_GWAS_L2G's top-5 structures were expected to be disease-irrelevant reticular/brainstem nuclei"

# %% [markdown]
# ## 6.1 Figure: CCS profile vs matched null bands

# %%
fig, axes = plt.subplots(1, 5, figsize=(23, 4.3), sharey=False)
NULL_STYLE = [("random", "#7f8c8d", "random-uniform (size-matched)"),
             ("sibmut", "#3498db", "sibling-mutability (=ASD, size-matched)")]
for ax, s in zip(axes, CCS_SETS):
    for null_lbl, color, label in NULL_STYLE:
        null = null_profiles[f"{s}__{null_lbl}"][:, ORDER]      # (10000, 6) ascending-N columns
        lo, med, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)
        ax.fill_between(SIZES_ASC, lo, hi, color=color, alpha=0.25, label=f"{label}, 95% band")
        ax.plot(SIZES_ASC, med, color=color, lw=1, ls="--")
    asd_lo = np.array([np.percentile(ASD_band[:, ASD_band_idx[N]], 2.5) for N in SIZES_ASC])
    asd_hi = np.array([np.percentile(ASD_band[:, ASD_band_idx[N]], 97.5) for N in SIZES_ASC])
    ax.fill_between(SIZES_ASC, asd_lo, asd_hi, facecolor="none", edgecolor="#e74c3c",
                    hatch="//", alpha=0.6, label="ASD 61-gene band (NOT size-matched -- reference only)")
    ax.plot(SIZES_ASC, observed[s][ORDER], color="black", marker="o", lw=1.5, label="observed")
    ax.set_title(f"{s}\n(n={int(geneset_meta.loc[s, 'n_genes'])} genes)", fontsize=10)
    ax.set_xlabel("circuit size N", fontsize=9)
    ax.set_facecolor("none")
axes[0].set_ylabel("CCS (Shannon-information connectivity score)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
          bbox_to_anchor=(0.5, -0.1))
fig.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/CCS_profile_vs_matched_nulls.png",
           transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 7. Gene-set union analysis
#
# Does pooling all four PD tiers strengthen the signal? Structure- and cell-type-level recovery
# for the individual tiers, the full union, and Mendelian-only sub-unions (with and without the
# five circular dopamine markers). Cell-type recovery uses `MouseCT_AvgZ_Weighted` with DN
# weights (`weight_DN = weight_ISH * max(V2-V3 Spearman r, 0)^2`), exactly as the published
# cell-type arm does; DN weights are never used at the structure level (repo convention).

# %%
def gw_entrez(name):
    return set(int(x) for x in Fil2Dict(f"../dat/Genetics/GeneWeights/{name}.gw"))


Z2 = pd.read_parquet(STR_EXPR_MATRIX)
CT = pd.read_parquet(CT_EXPR_MATRIX)
v2v3 = pd.read_csv(CROSS_PLATFORM_CORR_PATH, index_col="Genes")["V2_V3_CT_Corr"]

CT_GT_CLUSTERS = {
    disease: clusters_in_subclasses(CT.columns, GT["cell_type_subclasses"][disease]["core"])
    for disease in ("parkinson", "striatal")
}

P, DA, AT, GW = (gw_entrez("PD_Primary"), gw_entrez("PD_Sens_DA"),
                gw_entrez("PD_Sens_Atypical"), gw_entrez("PD_GWAS_L2G"))
# The dopamine-marker tier, derived (not hardcoded) as the set difference between PD_Sens_DA and
# PD_Primary -- PD_Sens_DA's YAML definition is exactly PD_Primary's pool plus pd_dopamine_markers.
DAonly = DA - P

union_sets = {
    "PD_Primary (Mendelian, no DA markers)": P,
    "PD_Sens_Atypical (+ disputed/atypical)": AT,
    "PD_GWAS_L2G (common-variant tier alone)": GW,
    "UNION all four tiers": P | DA | AT | GW,
    "UNION all four, minus DA markers": (P | DA | AT | GW) - DAonly,
    "UNION Mendelian only (Primary+DA+Atypical)": P | DA | AT,
    "UNION Mendelian, no DA markers (=Atypical)": (P | DA | AT) - DAonly,
}

parkinson_core = GT["structures"]["parkinson"]["core"]
dopa_clusters = CT_GT_CLUSTERS["parkinson"]

union_rows = []
for label, gs in union_sets.items():
    gs_w = {g: 1.0 for g in gs if g in Z2.index}
    s_str = recovery_stats(MouseSTR_AvgZ_Weighted(Z2, gs_w), parkinson_core)
    dn_w = {g: 1.0 * (max(v2v3.loc[g], 0.0) ** 2) for g in gs if g in CT.index and g in v2v3.index}
    s_ct = recovery_stats(MouseCT_AvgZ_Weighted(CT, dn_w), dopa_clusters)
    union_rows.append({
        "combination": label, "n_genes": len(gs_w),
        "STR_AUROC": round(s_str["auroc"], 3), "STR_p_mannwhitney": s_str["p_mannwhitney"],
        "CT_AUROC": round(s_ct["auroc"], 3), "CT_p_mannwhitney": s_ct["p_mannwhitney"],
    })
union_table = pd.DataFrame(union_rows).set_index("combination")
union_table.to_csv(f"{TABLE_DIR}/PD_HD_validation_union_analysis.csv")
union_table

# %%
mendelian_only_str = union_table.loc["UNION Mendelian only (Primary+DA+Atypical)", "STR_AUROC"]
union_all_str = union_table.loc["UNION all four tiers", "STR_AUROC"]
mendelian_only_ct = union_table.loc["UNION Mendelian only (Primary+DA+Atypical)", "CT_AUROC"]
union_all_ct = union_table.loc["UNION all four tiers", "CT_AUROC"]
shared = sorted(Entrez2Symbol.get(g, str(g)) for g in (P | DA | AT) & GW)

print(f"Structure AUROC: Mendelian-only union {mendelian_only_str:.3f} -> "
     f"{union_all_str:.3f} once the GWAS tier is added ({union_all_str - mendelian_only_str:+.3f})")
print(f"Cell-type AUROC: Mendelian-only union {mendelian_only_ct:.3f} -> "
     f"{union_all_ct:.3f} once the GWAS tier is added ({union_all_ct - mendelian_only_ct:+.3f})")
print(f"Genes shared between the Mendelian tiers and the 40-gene GWAS tier: {shared} "
     f"({len(shared)} of 40)")

assert union_all_str < mendelian_only_str and union_all_ct < mendelian_only_ct, \
    "adding the GWAS tier was expected to DILUTE recovery at both levels"

# %% [markdown]
# Adding the common-variant (GWAS) tier to the union **dilutes** the signal at both levels, not
# just marginally: structure AUROC drops and cell-type AUROC drops much more sharply (0.98 for
# the DA-marker-containing Mendelian union down to 0.81). Only 3 of the 40 GWAS genes overlap the
# Mendelian tiers at all (`LRRK2`, `MAPT`, `SNCA`), so pooling tiers adds 37 largely independent,
# non-nigrostriatal-selective genes that pull the average toward the population mean. This is
# direct evidence against a "more genes = more power" intuition for this framework: gene-set
# curation quality matters more than gene-set size.

# %% [markdown]
# # 8. Consolidated summary table
#
# One row per gene set, pulling together the structure-level recovery (Section 4) and cell-type
# recovery (read the same way, from `results/CT_Z2/`, read-only) into the table used for the
# rebuttal response letter and supplementary materials. This is the only disk write in this
# notebook outside `results/PD_HD_validation/`'s figures/tables/cache subtrees.

# %%
def ct_recovery_row(s):
    gt_clusters = CT_GT_CLUSTERS[GT_KEY[s]]
    bias_f = f"{CT_RESULTS_DIR}/{s}_bias_addP_random.csv"
    if not os.path.exists(bias_f):
        return {}
    d = pd.read_csv(bias_f, index_col=0)
    st = recovery_stats(d, gt_clusters)
    row = {"CT_n_clusters": st["n_ground_truth"], "CT_AUROC": st["auroc"],
          "CT_p_mannwhitney": st["p_mannwhitney"], "CT_median_rank": st["median_rank"]}
    null_f = f"{CT_RESULTS_DIR}/null_bias/{s}_null_bias_random.parquet"
    if os.path.exists(null_f):
        try:
            row["CT_p_geneset"] = empirical_p(st["auroc"], recovery_null_aurocs(pd.read_parquet(null_f), gt_clusters))
        except ValueError:
            row["CT_p_geneset"] = np.nan  # NaN guard -- same small-gene-set cause as Section 4
    else:
        row["CT_p_geneset"] = np.nan
    return row


ct_block = pd.DataFrame({s: ct_recovery_row(s) for s in ALL_SETS}).T
ct_block.index.name = "gene_set"

summary_df = recovery_table.join(ct_block).round(4)
summary_df = summary_df.reset_index()

SUMMARY_CSV = f"{TABLE_DIR}/PD_HD_validation_summary.csv"
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Consolidated summary ({summary_df.shape[0]} gene sets x {summary_df.shape[1]} columns) -> {SUMMARY_CSV}")
summary_df

# %% [markdown]
# # 9. Verification
#
# Assertions against the pre-registration and against the pipeline outputs actually written to
# disk (re-read fresh here, not taken from in-memory variables computed above) — the notebook's
# contract with the numbers reported in the rebuttal. If any upstream pipeline output changes,
# this section fails loudly rather than silently reporting stale numbers.

# %%
_summary_on_disk = pd.read_csv(SUMMARY_CSV).set_index("gene_set")
_top20_on_disk = pd.read_csv(f"{TABLE_DIR}/PD_HD_validation_top20_structures.csv")
_highconf_da_bias_on_disk = pd.read_csv(
    f"{STR_RESULTS_DIR}/PD_HighConf_DA_bias_addP_random.csv", index_col=0
).sort_values("EFFECT", ascending=False)

# 1. PD_HighConf_DA has 19 genes.
n_genes_highconf_da = int(_summary_on_disk.loc["PD_HighConf_DA", "n_genes"])
assert n_genes_highconf_da == 19, f"PD_HighConf_DA gene count changed: {n_genes_highconf_da}"

# 2. Its sibling-mutability recovery p is between 0.10 and 0.13.
p_sibmut_on_disk = float(_summary_on_disk.loc["PD_HighConf_DA", "p_sibling_mutability"])
assert 0.10 < p_sibmut_on_disk < 0.13, \
    f"PD_HighConf_DA sibling-mutability recovery p changed: {p_sibmut_on_disk}"

# 3. It has 15 structures at q<0.10 under the sibling null (sibling-mutability is the only
#    sibling-flavoured null registered for the post-hoc tier -- Section 4).
q10_sibmut_on_disk = int(_summary_on_disk.loc["PD_HighConf_DA", "q10_sibling_mutability"])
assert q10_sibmut_on_disk == 15, \
    f"PD_HighConf_DA structures at q<0.10 (sibling-mutability null) changed: {q10_sibmut_on_disk}"

# 4. VTA / dorsal raphe / SNc are its top 3 structures.
top3_on_disk = _highconf_da_bias_on_disk.index[:3].tolist()
assert top3_on_disk == ["Ventral_tegmental_area", "Dorsal_nucleus_raphe", "Substantia_nigra_compact_part"], \
    f"PD_HighConf_DA top-3 structures changed: {top3_on_disk}"

# Bonus checks: internal consistency of the tables actually written to disk.
assert set(_summary_on_disk.index) == set(ALL_SETS), "summary table is missing a gene set"
assert set(_top20_on_disk["gene_set"].unique()) == set(PD_SETS), "top-20 export is missing a PD gene set"
assert (_top20_on_disk.groupby("gene_set").size() == 20).all(), "every gene set must contribute 20 rows"

print("All verification assertions passed:")
print(f"  PD_HighConf_DA: n_genes={n_genes_highconf_da}, "
     f"p_sibling_mutability={p_sibmut_on_disk:.4f}, q10_sibling_mutability={q10_sibmut_on_disk}")
print(f"  PD_HighConf_DA top 3: {top3_on_disk}")
print(f"\nOutputs written under {OUT_DIR}/:")
for sub in ("figures", "tables", "cache"):
    for fn in sorted(os.listdir(f"{OUT_DIR}/{sub}")):
        print(f"  {sub}/{fn}")
