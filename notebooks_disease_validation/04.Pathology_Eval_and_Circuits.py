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
# # Notebook 04 — Blinded Pathology Evaluation and Pareto Circuit Analysis
#
# Two independent validation arms for the PD / striatal-degeneration rebuttal (Reviewer 2):
#
# - **Arm 1 (Section 2)** — does an independent, blinded literature review of PD neuropathology
#   confirm that the structures our model ranks highest are genuine sites of PD-related
#   degeneration, using an anatomy-matched design that removes the confounds present in two
#   earlier attempts?
# - **Arm 2 (Section 3)** — when the simulated-annealing circuit search is calibrated the same
#   way the published ASD circuit was (accept the same ~20% bias sacrifice for a CCS gain), does
#   it retain the expected dopaminergic core (SNc, VTA) and monoaminergic co-degeneration sites
#   (raphe nuclei), rather than drifting to anatomically nonspecific, densely-connected regions?
#
# **Scope / constraints for this notebook:**
# - `results/` is **read-only** here, with the single exception of `results/PD_HD_validation/`.
#   Every path this notebook reads under `results/CircuitSearch/` and `results/STR_ISH/` was
#   produced by the SA circuit search and bias pipelines in earlier tasks (hours of compute) —
#   this notebook never re-runs Snakemake, SA search, or the bias pipeline.
# - Every number below is **recomputed from the files on disk**, not copied from prior notes —
#   the assertion cells in Section 5 will fail if any upstream file changes.
# - Working reference scripts that originally produced these numbers live in
#   `notebooks_disease_validation/reference/` (`unblind2.py`, `consensus2.py`, `refine.py`,
#   `knee.py`, `pareto_all.py`, `da_final.py`); this notebook supersedes them as the reproducible
#   source, but is checked against them throughout.

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # 1. Setup

# %%
import os
import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import cohen_kappa_score

sys.path.insert(1, "../src")
from ASD_Circuits import ScoreCircuit_SI_Joint, STR2Region
from disease_validation import load_ground_truth, recovery_stats, recovery_null_aurocs, empirical_p
from plot import REGION_COLORS, pretty_pval_allstyle, format_pval

SEED = 42
rng = np.random.default_rng(SEED)

FIG_DIR = "../results/PD_HD_validation/figures"
TAB_DIR = "../results/PD_HD_validation/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

anno = STR2Region()
GT = load_ground_truth("../config/disease_validation_ground_truth.yaml")

with open("../config/circuit_config_disease.yaml") as f:
    disease_cfg = yaml.safe_load(f)
SIZES = disease_cfg["circuit_sizes"]
INFO_MAT_PATH = "../" + disease_cfg["info_mat"]
CIRCUIT_OUT_DIR = "../" + disease_cfg["output_dir"]

InfoMat = pd.read_csv(INFO_MAT_PATH, index_col=0)
print(f"Circuit sizes under search: {SIZES}")
print(f"Datasets registered for the circuit search: {list(disease_cfg['Input_str_bias'])}")
for name, spec in disease_cfg["Input_str_bias"].items():
    print(f"  {name:22s} {spec['description']}")

# %% [markdown]
# # 2. Arm 1 — Blinded Structure-Pathology Evaluation
#
# ## 2.1 Methodology and its failures
#
# Three formulations of "does the model's structure ranking track real PD neuropathology" were
# tried, in this order. All three are part of the analytic record and are reported here — not
# just the one that worked.

# %% [markdown]
# ### 2.1.1 Attempt 1 — pre-registered composite AUROC over the 13-structure PD circuit (FAILED)
#
# The original pre-registration (`config/disease_validation_ground_truth.yaml`,
# `structures.parkinson.core`, 13 structures spanning nigrostriatal, pallidal, subthalamic,
# thalamic-relay and motor-cortical nodes) asked a single composite question: as a block, do
# these 13 structures rank above the rest of the atlas on `PD_HighConf_DA` bias? This is exactly
# `recovery_stats()` / `recovery_null_aurocs()` from `src/disease_validation.py`, using the
# sibling-mutability null — the same null model the published ASD result uses — recomputed below
# from the bias file and null-bias parquet already on disk.

# %%
core13 = GT["structures"]["parkinson"]["core"]
bias_sibmut = pd.read_csv("../results/STR_ISH/PD_HighConf_DA_SibMut_bias_addP_sibling.csv", index_col=0)
null_sibmut = pd.read_parquet("../results/STR_ISH/null_bias/PD_HighConf_DA_SibMut_null_bias_sibling.parquet")

s13 = recovery_stats(bias_sibmut, core13)
null_aurocs_13 = recovery_null_aurocs(null_sibmut, core13)
p_geneset_13 = empirical_p(s13["auroc"], null_aurocs_13)

print(f"13-structure composite AUROC = {s13['auroc']:.3f}")
print(f"  structure-permutation p_mannwhitney = {s13['p_mannwhitney']:.4f}")
print(f"  gene-set null p ({len(null_aurocs_13)} sims)  = {p_geneset_13:.4f}")
print("  -> FAILS to reach significance under the pre-registered gene-set null.")

assert s13["n_missing"] == 0, "a pre-registered core structure is missing from the atlas"
assert p_geneset_13 > 0.05, "attempt 1 is recorded as a failure to reach p<0.05; recompute disagrees"

# %% [markdown]
# ### 2.1.2 Attempt 2 — blinded binary test on mixed-anatomy decoys (FAILED)
#
# The next attempt had two independent raters, blind to model rank, classify each of 20
# top-ranked structures and 20 decoys **drawn from anywhere in the 213-structure atlas** as
# PD-pathology-positive or not, then asked whether "ever PD-affected" was more common in the
# top 20. Result: **6/20 top-ranked vs 6/20 decoys, Fisher exact p = 0.63** — no enrichment.
#
# By Braak stage 5–6 (the stage at which most PD brains come to autopsy), Lewy pathology has been
# reported in the large majority of catecholaminergic and many non-catecholaminergic nuclei
# through the brainstem, diencephalon and even neocortex. "Has PD pathology ever been reported
# here" therefore has a base rate close to ceiling almost everywhere in the brain, which strips
# the binary question of the resolution needed to separate a genuinely disease-preferential
# ranking from anatomy.
#
# ### 2.1.3 Attempt 2b — naive anatomical refinement (CONFOUNDED)
#
# Reasoning that the failure above might reflect PD-literature density varying by gross
# anatomical class rather than by structure identity, the same 20-vs-20 mixed decoy set was
# re-analyzed restricting "hit" to structures in brainstem/midbrain/diencephalic regions
# (`reference/refine.py`, `REFINED CRITERION`). This *appeared* to recover a signal, but the
# comparison was **confounded by construction**: the top-20 set was already 20/20
# brainstem/diencephalic, so the anatomical filter removed nothing from it, while the mixed
# decoy set was only 6/20 in those regions, so the filter removed 14/20 decoys. The two groups
# being compared after filtering were no longer selected the same way — the apparent enrichment
# is an artifact of the filter interacting differently with the two groups, not evidence about
# pathology.
#
# The raw per-rater annotation files from attempts 2 and 2b (`blind_key.csv`, `blind_rater1.csv`,
# `blind_rater2.csv`) were not retained beyond that round of blinding, so these two outcomes are
# reported from the project ledger and `reference/refine.py` rather than recomputed here; they are
# included because a rebuttal that reports only the design that worked, without the two that
# didn't, misrepresents how the result was obtained.
#
# ### 2.1.4 Attempt 3 — anatomy-matched design (this section; the valid test)
#
# Attempt 2b's confound is fixed at the source: decoys are now drawn **from the same anatomical
# classes as the top-20 set before any pathology classification happens**, so anatomy can no
# longer explain an enrichment. Concretely: 20 top-ranked structures (all brainstem/diencephalic
# under `PD_HighConf_DA`) vs. 30 decoys drawn from ranks 21–198 in the *same* regions
# (Midbrain / Pons / Medulla / Hypothalamus). Two raters, blind to rank, independently classified
# all 50 structures as `established` / `probable` / `downstream-only` / `no-evidence`, each with a
# verified PMID and rationale. This is `results/tables/PD_structure_blinded_pathology_eval.csv`,
# loaded below.

# %% [markdown]
# ## 2.2 Load the anatomy-matched blinded evaluation

# %%
arm1 = pd.read_csv("../results/tables/PD_structure_blinded_pathology_eval.csv")

n_top20 = int((arm1.group == "TOP20").sum())
n_decoy = int((arm1.group == "DECOY_BS").sum())
assert n_top20 == 20 and n_decoy == 30, "the anatomy-matched design is 20 top-ranked vs 30 decoys"
assert set(arm1.region.unique()) <= set(REGION_COLORS), "an eval region is missing a REGION_COLORS entry"

print(f"{len(arm1)} structures rated: {n_top20} TOP20, {n_decoy} DECOY_BS (anatomy-matched)")
print("Region composition (both groups drawn from the same brainstem/diencephalic classes):")
print(pd.crosstab(arm1.group, arm1.region).to_string())

hit = lambda s: s.isin(["established", "probable"])

# %% [markdown]
# ## 2.3 Inter-rater agreement

# %%
raw_agreement = (arm1.classification_rater1 == arm1.classification_rater2).mean()
kappa = cohen_kappa_score(arm1.classification_rater1, arm1.classification_rater2)
print(f"Raw agreement: {raw_agreement:.2f}   Cohen's kappa: {kappa:.3f}")

assert 0.55 <= kappa <= 0.65, f"kappa {kappa:.3f} outside the expected 0.55-0.65 band"

# %% [markdown]
# ## 2.4 Per-rater enrichment and rank test
#
# For each rater independently: (a) a 2x2 enrichment of "established/probable" hits in TOP20 vs
# DECOY_BS (one-sided Fisher exact), and (b) the rank test — do hit structures have a lower
# (= more implicated) `true_rank` than non-hit structures (one-sided Mann-Whitney U)?

# %%
rater_rows = []
for col, label in [("classification_rater1", "rater1"), ("classification_rater2", "rater2")]:
    sel = hit(arm1[col])
    a, c = int((arm1.group.eq("TOP20") & sel).sum()), int((arm1.group.eq("DECOY_BS") & sel).sum())
    orr, p_fisher = fisher_exact([[a, n_top20 - a], [c, n_decoy - c]], alternative="greater")
    u, p_rank = mannwhitneyu(arm1.loc[sel, "true_rank"], arm1.loc[~sel, "true_rank"], alternative="less")
    rater_rows.append(dict(
        rater=label, top20_hits=a, top20_n=n_top20, decoy_hits=c, decoy_n=n_decoy,
        odds_ratio=orr, p_fisher=p_fisher, p_rank=p_rank,
        median_rank_hit=arm1.loc[sel, "true_rank"].median(),
        median_rank_nonhit=arm1.loc[~sel, "true_rank"].median(),
    ))
rater_stats = pd.DataFrame(rater_rows)
for _, r in rater_stats.iterrows():
    print(f"{r.rater}: TOP20 {r.top20_hits:.0f}/{r.top20_n:.0f}  DECOY {r.decoy_hits:.0f}/{r.decoy_n:.0f}  "
          f"OR={r.odds_ratio:.2f}  {pretty_pval_allstyle(r.p_fisher)} | "
          f"rank {pretty_pval_allstyle(r.p_rank)} (median {r.median_rank_hit:.0f} vs {r.median_rank_nonhit:.0f})")

# %% [markdown]
# ## 2.5 Consensus — the headline result
#
# **Strict** = both raters call established/probable; **lenient** = either rater does. These are
# cross-checked against the `consensus_hit` / `either_hit` columns already in the source CSV
# (computed independently when the ratings were merged) before being used for anything downstream.

# %%
strict = hit(arm1.classification_rater1) & hit(arm1.classification_rater2)
lenient = hit(arm1.classification_rater1) | hit(arm1.classification_rater2)
assert (strict == arm1.consensus_hit).all(), "recomputed strict-consensus disagrees with consensus_hit column"
assert (lenient == arm1.either_hit).all(), "recomputed lenient-consensus disagrees with either_hit column"

consensus_rows = []
for label, sel in [("strict", strict), ("lenient", lenient)]:
    a, c = int((arm1.group.eq("TOP20") & sel).sum()), int((arm1.group.eq("DECOY_BS") & sel).sum())
    orr, p_fisher = fisher_exact([[a, n_top20 - a], [c, n_decoy - c]], alternative="greater")
    u, p_rank = mannwhitneyu(arm1.loc[sel, "true_rank"], arm1.loc[~sel, "true_rank"], alternative="less")
    consensus_rows.append(dict(
        rater="consensus_" + label, top20_hits=a, top20_n=n_top20, decoy_hits=c, decoy_n=n_decoy,
        odds_ratio=orr, p_fisher=p_fisher, p_rank=p_rank,
        median_rank_hit=arm1.loc[sel, "true_rank"].median(),
        median_rank_nonhit=arm1.loc[~sel, "true_rank"].median(),
    ))
    print(f"{label:8s}: TOP20 {a}/{n_top20}  DECOY {c}/{n_decoy}  OR={orr:.2f}  {pretty_pval_allstyle(p_fisher)} | "
          f"RANK TEST {pretty_pval_allstyle(p_rank)} (median {arm1.loc[sel,'true_rank'].median():.0f} vs "
          f"{arm1.loc[~sel,'true_rank'].median():.0f})")

consensus_stats = pd.DataFrame(consensus_rows)
p_rank_strict = consensus_stats.loc[consensus_stats.rater == "consensus_strict", "p_rank"].iloc[0]
p_rank_lenient = consensus_stats.loc[consensus_stats.rater == "consensus_lenient", "p_rank"].iloc[0]

print("\nConsensus (strict) hits by rank:")
for _, x in arm1[strict].sort_values("true_rank").iterrows():
    print(f"   rank {x.true_rank:3.0f}  {x.structure:<42s} {x.group}")

arm1_stats = pd.concat([rater_stats, consensus_stats], ignore_index=True)
arm1_stats.to_csv(f"{TAB_DIR}/PD_blinded_pathology_stats_summary.csv", index=False)

assert p_rank_strict < 0.01, f"strict consensus rank-test p={p_rank_strict:.4g} not < 0.01"
assert p_rank_lenient < 0.01, f"lenient consensus rank-test p={p_rank_lenient:.4g} not < 0.01"

# %% [markdown]
# ## 2.6 Figure (a) — rank distribution: PD-affected vs unaffected structures

# %%
fig, ax = plt.subplots(figsize=(4.6, 5.0), dpi=100)

cats = [("PD-affected\n(consensus, strict)", strict), ("Unaffected", ~strict)]
box_data = [arm1.loc[sel, "true_rank"].values for _, sel in cats]
bp = ax.boxplot(box_data, tick_labels=[c[0] for c in cats], widths=0.5,
                 showfliers=False, patch_artist=True, zorder=2)
for patch, color in zip(bp["boxes"], ["#c0392b", "#7f8c8d"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.22)
for median in bp["medians"]:
    median.set_color("black")

for i, (_, sel) in enumerate(cats, start=1):
    sub = arm1.loc[sel]
    x = i + rng.normal(loc=0.0, scale=0.06, size=len(sub))
    colors = [REGION_COLORS[r] for r in sub["region"]]
    ax.scatter(x, sub["true_rank"], color=colors, s=42, edgecolor="white",
               linewidth=0.6, zorder=3)

ax.set_ylabel("True rank in PD_HighConf_DA bias ranking\n(1 = most implicated, 198 = least)")
ax.invert_yaxis()
ax.set_title("Blinded pathology consensus vs. bias rank\n"
             f"strict {pretty_pval_allstyle(p_rank_strict)}   |   lenient {pretty_pval_allstyle(p_rank_lenient)}",
             fontsize=11)
ax.grid(axis="y", alpha=0.25)

region_handles = [Patch(facecolor=REGION_COLORS[r], label=r) for r in sorted(arm1.region.unique())]
ax.legend(handles=region_handles, title="Region", bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, fontsize=9)

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/04a_blinded_rank_distribution.png",
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 3. Arm 2 — Pareto Circuit Analysis
#
# ## 3.1 Load the Pareto fronts
#
# `Snakefile.circuit` / `scripts/workflow/create_pareto_front.py` wrote one Pareto-front CSV per
# `(dataset, size)` under `results/CircuitSearch/{dataset}/pareto_fronts/`. Each file has one
# `baseline` row (top-N structures by bias, no SA search) and many `optimized` rows (best SA
# circuit found at each of a grid of bias-floor constraints). These are read only — nothing here
# re-runs the search.

# %%
pareto = {}
for dataset in disease_cfg["Input_str_bias"]:
    for size in SIZES:
        f = f"{CIRCUIT_OUT_DIR}/{dataset}/pareto_fronts/{dataset}_size_{size}_pareto_front.csv"
        df = pd.read_csv(f)
        assert (df.circuit_type == "baseline").sum() == 1, f"{f}: expected exactly one baseline row"
        assert int(df.loc[df.circuit_type == "baseline", "n_structures"].iloc[0]) == size
        pareto[(dataset, size)] = df
        print(f"{dataset:22s} size {size:>2d}: {len(df):3d} Pareto points "
              f"({(df.circuit_type=='optimized').sum()} optimized)")

# %% [markdown]
# ## 3.2 The two SA datasets are identical by construction
#
# `PD_HighConf_DA` and `PD_HighConf_DA_SibMut` are the same 19 genes scored under two different
# null models (uniform-random vs. mutability-weighted sibling — the exact procedure the published
# ASD result uses). The null model only changes the *p-value/q-value* columns of the bias table;
# simulated annealing consumes only the `EFFECT` column, which is identical between the two bias
# files by construction (the null never touches the observed gene set's own bias). So the SA
# search — which never sees a null — must return byte-identical circuits for both datasets. This
# is checked explicitly, not assumed:

# %%
for size in SIZES:
    a = pareto[("PD_HighConf_DA", size)].sort_values("bias_limit", na_position="first").reset_index(drop=True)
    b = pareto[("PD_HighConf_DA_SibMut", size)].sort_values("bias_limit", na_position="first").reset_index(drop=True)
    same_struct = (a["structures"].values == b["structures"].values).all()
    same_score = np.allclose(a["circuit_score"].values, b["circuit_score"].values)
    same_bias = np.allclose(a["mean_bias"].values, b["mean_bias"].values)
    assert same_struct and same_score and same_bias, f"size {size}: the two SA datasets diverge"
    print(f"size {size:>2d}: PD_HighConf_DA == PD_HighConf_DA_SibMut  "
          f"(structures {same_struct}, CCS {same_score}, bias {same_bias})")

print("\n=> The two SA runs are one experiment, not two. Everything below reports 4 circuits"
      " (one per size), not 8; PD_HighConf_DA is used as the representative dataset.")

# %% [markdown]
# ## 3.3 Baseline ranking, and an independent check of the Pareto front
#
# The `PD_HighConf_DA` bias ranking (sorted by `EFFECT`, descending) is what both `true_rank` in
# Arm 1 and the circuit-search baseline are built from. As a check that the Pareto front on disk
# really is what `ScoreCircuit_SI_Joint` + this ranking produce (not merely self-consistent with
# whatever the SA search happened to write), the baseline CCS for one size is recomputed here
# directly from the bias ranking and the connectivity `InfoMat` and compared bit-for-bit against
# the stored Pareto front.

# %%
bias_pd = pd.read_csv("../results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv", index_col=0)
bias_pd = bias_pd.sort_values("EFFECT", ascending=False)
bias_rank = pd.Series(np.arange(1, len(bias_pd) + 1), index=bias_pd.index)

check_size = SIZES[1]  # size 13
recomputed_ccs = ScoreCircuit_SI_Joint(list(bias_pd.index[:check_size]), InfoMat)
stored_baseline = pareto[("PD_HighConf_DA", check_size)]
stored_baseline = stored_baseline.loc[stored_baseline.circuit_type == "baseline"].iloc[0]
assert abs(recomputed_ccs - stored_baseline.circuit_score) < 1e-9, \
    "recomputed baseline CCS does not match the stored Pareto front"
print(f"size {check_size}: baseline CCS recomputed from bias ranking + InfoMat = {recomputed_ccs:.6f}, "
      f"stored = {stored_baseline.circuit_score:.6f} -> exact match")

# %% [markdown]
# ## 3.4 The ASD-matched operating point
#
# SA does not simply maximise CCS — it produces a whole trade-off curve between mean structure
# bias (how strongly the circuit is implicated by genetics) and CCS (how densely interconnected
# the circuit is). The published ASD circuit search accepted a **~20% reduction in mean bias for
# a ~86% gain in CCS**. That reference point is not a hardcoded number here — it is recovered
# below from the actual published Pareto front
# (`results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv`) using
# exactly the same selection rule applied to the PD circuits in Section 3.5: the optimized point
# whose bias sacrifice is closest to −20%.

# %%
def select_at_bias_sacrifice(front, target_pct=-20.0):
    """Pareto point whose bias sacrifice (vs. baseline) is closest to target_pct.

    Mirrors reference/pareto_all.py's selection rule: baseline is top-N-by-bias with no SA
    search; 'optimized' rows are the best SA circuit at each bias-floor constraint. Returns
    (baseline_row, selected_row); selected_row carries dBias_pct / dCCS_pct.
    """
    base = front.loc[front.circuit_type == "baseline"].iloc[0]
    opt = front.loc[front.circuit_type == "optimized"].copy()
    opt["dBias_pct"] = 100 * (opt.mean_bias - base.mean_bias) / base.mean_bias
    opt["dCCS_pct"] = 100 * (opt.circuit_score - base.circuit_score) / base.circuit_score
    sel = opt.iloc[(opt.dBias_pct - target_pct).abs().argmin()]
    return base, sel


asd_front = pd.read_csv(
    "../results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv")
asd_base, asd_sel = select_at_bias_sacrifice(asd_front, target_pct=-20.0)
print(f"ASD reference (published, size 46): baseline CCS {asd_base.circuit_score:.3f} "
      f"bias {asd_base.mean_bias:.3f}  ->  selected CCS {asd_sel.circuit_score:.3f} "
      f"bias {asd_sel.mean_bias:.3f}")
print(f"  i.e. ASD accepted bias {asd_sel.dBias_pct:+.1f}% for CCS {asd_sel.dCCS_pct:+.1f}%")

assert -25 < asd_sel.dBias_pct < -15, "the ASD reference point is not close to the claimed ~20% sacrifice"
assert asd_sel.dCCS_pct > 70, "the ASD reference CCS gain is not close to the claimed ~86%"

# %% [markdown]
# ## 3.5 Applying the same criterion to the four PD circuit sizes
#
# `select_at_bias_sacrifice` is applied, unchanged, to each of the four `PD_HighConf_DA` Pareto
# fronts.

# %%
pd_summary_rows = []
for size in SIZES:
    base, sel = select_at_bias_sacrifice(pareto[("PD_HighConf_DA", size)], target_pct=-20.0)
    st = sel.structures.split(",")
    pd_summary_rows.append(dict(
        dataset="PD_HighConf_DA", size=size,
        base_CCS=base.circuit_score, base_bias=base.mean_bias,
        sel_CCS=sel.circuit_score, sel_bias=sel.mean_bias,
        dBias_pct=sel.dBias_pct, dCCS_pct=sel.dCCS_pct,
        VTA="Ventral_tegmental_area" in st,
        SNc="Substantia_nigra_compact_part" in st,
        n_raphe=sum("raphe" in s.lower() for s in st),
        top5_kept=sum(s in st for s in bias_pd.index[:5]),
        structures=st,
    ))
pd_summary = pd.DataFrame(pd_summary_rows)

print(f"{'size':>4s} {'baseCCS':>8s} {'selCCS':>8s} {'dBias%':>7s} {'dCCS%':>7s} "
      f"{'VTA':>5s} {'SNc':>5s} {'raphe':>6s} {'top5kept':>9s}")
for _, x in pd_summary.iterrows():
    print(f"{x['size']:>4d} {x.base_CCS:>8.3f} {x.sel_CCS:>8.3f} {x.dBias_pct:>7.1f} {x.dCCS_pct:>7.1f} "
          f"{'YES' if x.VTA else 'no':>5s} {'YES' if x.SNc else 'no':>5s} {x.n_raphe:>6d} {x.top5_kept:>8d}/5")

assert pd_summary["VTA"].all(), "VTA absent from the ASD-matched circuit at at least one size"
assert pd_summary["SNc"].all(), "SNc absent from the ASD-matched circuit at at least one size"
assert pd_summary["n_raphe"].between(3, 4).all(), "raphe count outside the expected 3-4 per circuit"

# Cross-check against the snapshot pareto_all.py already wrote (results/tables/, read-only) —
# confirms this notebook's recomputation from raw Pareto fronts reproduces the project record.
_prior = pd.read_csv("../results/tables/PD_circuit_pareto_summary.csv")
_prior = _prior[_prior.dataset == "PD_HighConf_DA"].sort_values("size").reset_index(drop=True)
_mine = pd_summary.sort_values("size").reset_index(drop=True)
assert np.allclose(_prior["sel_CCS"].values, _mine["sel_CCS"].values, atol=1e-6)
assert np.allclose(_prior["dBias"].values, _mine["dBias_pct"].values, atol=1e-6)
print("\nCross-check vs. results/tables/PD_circuit_pareto_summary.csv (pareto_all.py): exact match.")

# %% [markdown]
# ## 3.6 Reading only the extreme end of the front gives the opposite conclusion
#
# At the most permissive bias floor — the point that maximizes CCS with no regard for bias — SA
# drifts away from the dopaminergic core toward densely-interconnected reticular formation. This
# is a real feature of the Pareto front, not a search failure — but it is the **extreme end** of
# the trade-off, not the operating point a reader would actually choose (Section 3.5). Reporting
# only this end of the front would give the opposite conclusion from the ASD-matched operating
# point.

# %%
for size in SIZES:
    opt = pareto[("PD_HighConf_DA", size)]
    opt = opt.loc[opt.circuit_type == "optimized"]
    extreme = opt.loc[opt.circuit_score.idxmax()]
    st = set(extreme.structures.split(","))
    reticular = sorted(s for s in st if "reticular" in s.lower())
    has_vta = "Ventral_tegmental_area" in st
    has_snc = "Substantia_nigra_compact_part" in st
    print(f"size {size:>2d}: extreme point CCS={extreme.circuit_score:.3f} bias={extreme.mean_bias:.3f}  "
          f"VTA={'yes' if has_vta else 'NO'}  SNc={'yes' if has_snc else 'NO'}  "
          f"reticular formation nuclei: {len(reticular)} ({', '.join(s[:28] for s in reticular[:3])}{'...' if len(reticular)>3 else ''})")
    assert not has_vta and not has_snc, f"size {size}: expected the extreme point to drop VTA/SNc"

print("\n=> At the extreme end SNc/VTA drop out at every size; at the ASD-matched point (3.5)"
      " they are retained at every size. Both must be read together.")

# %% [markdown]
# ## 3.7 Figure (b) — Pareto fronts with baseline and the ASD-matched operating point

# %%
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0), dpi=100)

for ax, size in zip(axes.flat, SIZES):
    front = pareto[("PD_HighConf_DA", size)]
    opt = front.loc[front.circuit_type == "optimized"].sort_values("mean_bias")
    base, sel = select_at_bias_sacrifice(front, target_pct=-20.0)
    extreme = opt.loc[opt.circuit_score.idxmax()]

    ax.plot(opt.circuit_score, opt.mean_bias, "-", color="#542788", lw=1.5, zorder=2, label="Pareto front")
    ax.scatter([base.circuit_score], [base.mean_bias], marker="D", s=50, color="black",
               zorder=5, label="Baseline (top-N by bias)")
    ax.scatter([sel.circuit_score], [sel.mean_bias], marker="x", s=80, color="red", linewidth=2.2,
               zorder=6, label="ASD-matched (~20% bias sacrifice)")
    ax.scatter([extreme.circuit_score], [extreme.mean_bias], marker="o", s=45, facecolor="none",
               edgecolor="#888888", linewidth=1.3, zorder=5, label="Extreme end (max CCS)")

    ax.set_title(f"size {size}", fontsize=11)
    ax.set_xlabel("Circuit Connectivity Score", fontsize=9)
    ax.set_ylabel("Mean structure bias", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle("PD_HighConf_DA: connectivity-bias trade-off per circuit size", fontsize=12)

fig.patch.set_alpha(0)
for ax in axes.flat:
    ax.patch.set_alpha(0)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
fig.savefig(f"{FIG_DIR}/04b_pareto_fronts_operating_points.png",
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3.8 Circuit membership across sizes and the consensus core
#
# Union of all structures appearing in any of the four ASD-matched circuits, and the subset
# common to **all four** — the consensus core.

# %%
circuit_sets = {int(row["size"]): set(row.structures) for _, row in pd_summary.iterrows()}
all_structures = sorted(set().union(*circuit_sets.values()), key=lambda s: bias_rank[s])
core_consensus = sorted(set.intersection(*circuit_sets.values()), key=lambda s: bias_rank[s])

membership = pd.DataFrame(
    {size: [s in circuit_sets[size] for s in all_structures] for size in SIZES},
    index=all_structures,
)
membership.insert(0, "rank", [int(bias_rank[s]) for s in all_structures])
membership.insert(1, "region", [anno.get(s, "?") for s in all_structures])
membership.insert(2, "in_consensus_core", [s in core_consensus for s in all_structures])

print(f"{len(all_structures)} structures appear in at least one ASD-matched circuit; "
      f"{len(core_consensus)} are common to all four sizes:\n")
for s in core_consensus:
    print(f"   rank {bias_rank[s]:>3d}  bias {bias_pd.loc[s,'EFFECT']:6.3f}  {s:<44s} {anno.get(s,'?')}")

assert len(core_consensus) == 9, f"consensus core has {len(core_consensus)} structures, expected 9"

# Cross-check against results/tables/PD_circuit_core_consensus.csv (pareto_all.py, read-only).
_prior_core = pd.read_csv("../results/tables/PD_circuit_core_consensus.csv")
assert set(_prior_core.structure) == set(core_consensus), "core consensus differs from the project record"
print("\nCross-check vs. results/tables/PD_circuit_core_consensus.csv (pareto_all.py): exact match.")

membership.to_csv(f"{TAB_DIR}/PD_circuit_membership_by_size.csv")
pd_summary.drop(columns="structures").to_csv(f"{TAB_DIR}/PD_circuit_pareto_summary.csv", index=False)
pd.DataFrame({"structure": core_consensus, "rank": [int(bias_rank[s]) for s in core_consensus],
              "bias": [bias_pd.loc[s, "EFFECT"] for s in core_consensus],
              "region": [anno.get(s, "?") for s in core_consensus]}
             ).to_csv(f"{TAB_DIR}/PD_circuit_core_consensus.csv", index=False)

# %% [markdown]
# ## 3.9 Figure (c) — consensus circuit membership across sizes

# %%
fig, ax = plt.subplots(figsize=(5.4, 0.22 * len(all_structures) + 1.0), dpi=100)

rgba = np.ones((len(all_structures), len(SIZES), 4))
for i, s in enumerate(all_structures):
    color = np.array(plt.matplotlib.colors.to_rgba(REGION_COLORS[anno.get(s, "?")]))
    for j, size in enumerate(SIZES):
        rgba[i, j] = color if membership.loc[s, size] else [0.85, 0.85, 0.85, 0.12]

ax.imshow(rgba, aspect="auto")
ax.set_xticks(range(len(SIZES)))
ax.set_xticklabels([str(s) for s in SIZES], fontsize=10)
ax.set_xlabel("Circuit size (n)", fontsize=10)
ax.set_yticks(range(len(all_structures)))
labels = ax.set_yticklabels(
    [("* " if s in core_consensus else "  ") + s.replace("_", " ") for s in all_structures],
    fontsize=8)
for lbl, s in zip(labels, all_structures):
    if s in core_consensus:
        lbl.set_fontweight("bold")

ax.set_xticks(np.arange(-0.5, len(SIZES), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(all_structures), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
ax.set_title("Circuit membership at the ASD-matched operating point\n"
              "(* = consensus core, present at all four sizes)", fontsize=10)

present_regions = sorted({anno.get(s, "?") for s in all_structures})
region_handles = [Patch(facecolor=REGION_COLORS[r], label=r) for r in present_regions]
ax.legend(handles=region_handles, title="Region", bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, fontsize=8)

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/04c_circuit_consensus_membership.png",
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 4. Synthesis for the Rebuttal
#
# **Arm 1.** Two earlier blinded/quasi-blinded formulations of the structure-pathology question
# failed or were confounded (Section 2.1) — reported in full, not omitted. The anatomy-matched
# design removes the confound by drawing decoys from the same brainstem/diencephalic classes as
# the top-ranked set before any pathology classification. Under that design, independent raters
# show substantial agreement (κ ≈ 0.59), and both the strict (both-rater) and lenient
# (either-rater) consensus definitions show top-ranked structures are enriched for literature-
# documented PD pathology and, more importantly, rank significantly higher than unaffected
# structures on the model's own bias ranking (rank test p < 0.01 both ways; headline consensus
# hits include VTA, SNc, dorsal raphe and PAG at ranks 1–7 of 213).
#
# **Arm 2.** The two SA datasets (`PD_HighConf_DA`, `PD_HighConf_DA_SibMut`) search over the same
# `EFFECT` values and are confirmed identical circuit-for-circuit — this is one experiment,
# reported as 4 circuits (one per size), not 8. Calibrated to the same ~20% bias-sacrifice
# operating point the published ASD circuit used, the PD circuit search retains SNc and VTA at
# every size (11/13/15/20) along with 3–4 raphe nuclei, and a 9-structure consensus core is common
# to all four sizes. Reading only the most permissive end of the Pareto front — where SA is free
# to maximize CCS without regard to bias — gives the opposite impression (SNc/VTA drop out in
# favor of anatomically nonspecific reticular formation); that is the extreme of the trade-off
# curve, not the operating point used for any claim in this rebuttal.

# %% [markdown]
# # 5. Verification Gate
#
# The four numbers this rebuttal's headline claims rest on, restated explicitly so a change to
# any upstream file causes a loud failure here rather than a silently stale number downstream.

# %%
assert 0.55 <= kappa <= 0.65, f"Cohen's kappa {kappa:.3f} outside [0.55, 0.65]"
assert p_rank_strict < 0.01, f"consensus (strict) rank-test p={p_rank_strict:.4g} not < 0.01"
assert p_rank_lenient < 0.01, f"consensus (lenient) rank-test p={p_rank_lenient:.4g} not < 0.01"
assert pd_summary["VTA"].all() and pd_summary["SNc"].all(), \
    "VTA/SNc not retained at all 4 ASD-matched circuit sizes"
assert len(core_consensus) == 9, f"consensus core has {len(core_consensus)} structures, expected 9"

print("VERIFIED:")
print(f"  Cohen's kappa               = {kappa:.3f}  (in [0.55, 0.65])")
print(f"  consensus rank-test p       strict={format_pval(p_rank_strict)}  lenient={format_pval(p_rank_lenient)}  (both < 0.01)")
print(f"  VTA & SNc retained at sizes {list(pd_summary['size'])}: True")
print(f"  consensus core size         = {len(core_consensus)}")
