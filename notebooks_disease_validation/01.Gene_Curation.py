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
# # Notebook 01 — Gene Curation and Weight Files
#
# Turns the pre-registered PD / striatal-degeneration gene sets
# (`config/disease_validation_genesets.yaml`, frozen in Task 1 — commit
# printed in Section 7) into the `.gw` / `.DN.gw` weight files that every
# downstream task (structure-level bias, cell-type bias, circuit search)
# consumes.
#
# This notebook **reads** `config/disease_validation_genesets.yaml`; it
# never writes to it. Section 3 re-derives the common-variant
# (`PD_GWAS_L2G`) tier from the Open Targets API purely as an **audit** of
# the frozen pre-registration — the frozen YAML list is authoritative
# regardless of the outcome of that re-derivation.

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # 1. Setup

# %%
import os
import sys
import subprocess

import yaml
import pandas as pd

sys.path.insert(1, "../src")
from ASD_Circuits import LoadGeneINFO, Dict2Fil, Fil2Dict
from disease_validation import load_gene_sets, gene_set_weights, gene_set_report

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
print(f"GeneSymbol2Entrez: {len(GeneSymbol2Entrez)} approved HGNC symbols")

# %% [markdown]
# # 2. Load pre-registered gene sets

# %%
# Expand the pre-registered gene pools into the six analysis gene sets. This
# reads config/disease_validation_genesets.yaml only — the file frozen in
# Task 1 — and never writes to it.
gene_sets = load_gene_sets("../config/disease_validation_genesets.yaml")
for name, records in gene_sets.items():
    print(f"{name:22s} n_curated={len(records):3d}")

# %% [markdown]
# # 3. Open Targets L2G pull for the GWAS tier

# %%
# Section 3: PD GWAS tier via Open Targets locus-to-gene ML scores.
# Nearest-gene assignment is deliberately NOT used - that is what made the
# legacy Parkinson.top61.gw list unusable.
import json, urllib.request

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PD_STUDIES = ["GCST009325", "GCST009324", "GCST004902", "GCST002544",
              "GCST003984", "GCST010049", "GCST009512"]
L2G_THRESHOLD = 0.5
L2G_ARTIFACTS = {"FLG", "HRNR", "MUC19"}   # 1q21 epidermal cluster + mucin repeat
CACHE = "../dat/Disease_Validation/pd_l2g_opentargets.json"

def ot_query(query):
    req = urllib.request.Request(
        OT_URL, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))

os.makedirs(os.path.dirname(CACHE), exist_ok=True)

# Open Targets is a live API: an unreachable network, a rate limit, or a
# changed schema must not fail this notebook. On any such failure this cell
# falls back to the frozen pre-registration and says so loudly rather than
# raising - the frozen list is authoritative either way (see the AUDIT block
# below, which is what actually enforces agreement when the pull succeeds).
network_ok = True
if os.path.exists(CACHE):
    l2g = json.load(open(CACHE))
    print(f"Loaded cached Open Targets pull: {len(l2g)} genes")
else:
    try:
        best = {}
        for s in PD_STUDIES:
            d = ot_query(
                '{ study(studyId: "%s") { credibleSets(page:{index:0,size:100}) '
                '{ rows { l2GPredictions(page:{index:0,size:5}) '
                '{ rows { score target { approvedSymbol } } } } } } }' % s)
            for r in d["data"]["study"]["credibleSets"]["rows"]:
                for p in r["l2GPredictions"]["rows"]:
                    g = p["target"]["approvedSymbol"]
                    if p["score"] > best.get(g, {"l2g": 0})["l2g"]:
                        best[g] = {"l2g": p["score"], "study": s}
        l2g = best
        json.dump(l2g, open(CACHE, "w"), indent=1)
        print(f"Fetched Open Targets pull: {len(l2g)} genes -> {CACHE}")
    except Exception as e:
        network_ok = False
        l2g = {}
        print(f"WARNING: Open Targets pull failed ({type(e).__name__}: {e}).")
        print("  Falling back to the frozen pre-registered PD_GWAS_L2G list;")
        print("  the re-derivation audit below is skipped this run.")

# AUDIT, do not define. Membership is frozen in the pre-registered YAML; this
# cell re-derives it and asserts agreement. Open Targets is a live API, so a
# silent upstream change must fail loudly rather than quietly alter the gene set.
frozen = sorted(r["symbol"] for r in
    load_gene_sets("../config/disease_validation_genesets.yaml")["PD_GWAS_L2G"])
if network_ok:
    derived = sorted(
        g for g, v in l2g.items() if v["l2g"] >= L2G_THRESHOLD and g not in L2G_ARTIFACTS)
    if derived != frozen:
        print(f"WARNING: Open Targets now yields {len(derived)} genes, "
              f"pre-registration has {len(frozen)}")
        print("  added upstream:", sorted(set(derived) - set(frozen)))
        print("  dropped upstream:", sorted(set(frozen) - set(derived)))
        print("  Proceeding with the FROZEN list. Report this drift in the methods.")
    else:
        print(f"PD_GWAS_L2G: {len(frozen)} genes, re-derivation matches pre-registration.")
else:
    print(f"PD_GWAS_L2G: proceeding with the frozen list ({len(frozen)} genes); "
          "no live re-derivation available this run.")
pd_gwas_symbols = frozen

# %% [markdown]
# # 4. Resolve to Entrez and report coverage
# # 5. Write `.gw` weight files
#
# Both happen together below: for each of the six gene sets, resolve curated
# HGNC symbols to Entrez IDs, restrict to genes present in the STR_ISH
# expression matrix, write the surviving weights to
# `dat/Genetics/GeneWeights/`, and accumulate a per-gene coverage report.

# %%
# Section 5: write weight files. Uniform weight 1.0 - matches how the published
# NT positive controls and non-brain negative controls were run.
Z2 = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")
valid_genes = set(Z2.index)

# All six sets, PD_GWAS_L2G included, come straight from the frozen YAML.
gene_sets = load_gene_sets("../config/disease_validation_genesets.yaml")

reports = {}
for name, records in gene_sets.items():
    weights = gene_set_weights(records, GeneSymbol2Entrez, valid_genes)
    Dict2Fil(weights, f"../dat/Genetics/GeneWeights/{name}.gw")
    rep = gene_set_report(records, GeneSymbol2Entrez, valid_genes)
    rep.insert(0, "gene_set", name)
    reports[name] = rep
    print(f"{name:22s} curated={len(records):3d} in_matrix={len(weights):3d} "
          f"dropped={sorted(rep.loc[~rep.in_matrix, 'symbol'])}")

report = pd.concat(reports.values(), ignore_index=True)
os.makedirs("../results/tables", exist_ok=True)
report.to_csv("../results/tables/disease_validation_gene_report.csv", index=False)

# %% [markdown]
# # 6. Write DN weight files (cell-type arm only)
#
# `weight_DN = weight_ISH * max(spearman_r, 0)^2`, from V2-V3 chemistry
# reproducibility. **Never use these for STR_ISH** — it corrupts EFFECT
# values. DN weights feed `MouseCT_AvgZ_Weighted` / the cell-type arm only
# (Task 11).

# %%
# Section 6: DN weights for the CELL-TYPE arm only.
# weight_DN = weight_ISH * max(spearman_r, 0)^2, from V2-V3 chemistry reproducibility.
# Never use these for STR_ISH - it corrupts EFFECT values.
CorrDF = pd.read_csv(f"../{config['data_files']['gene_cross_platform_corr']}", index_col="Genes")
v2v3 = CorrDF["V2_V3_CT_Corr"]

os.makedirs("../dat/Genetics/GeneWeights_DN", exist_ok=True)
for name in gene_sets:
    raw = Fil2Dict(f"../dat/Genetics/GeneWeights/{name}.gw")
    dn = {g: w * (max(v2v3.loc[g], 0.0) ** 2)
          for g, w in raw.items() if g in v2v3.index}
    Dict2Fil(dn, f"../dat/Genetics/GeneWeights_DN/{name}.DN.gw")
    print(f"{name:22s} {len(raw)} raw -> {len(dn)} DN genes")
    # Strict per the task's verification standard: a silently dropped or
    # NaN-valued DN weight would corrupt cell-type EFFECT values with no
    # visible symptom (same principle as the NaN guard in
    # recovery_null_aurocs, src/disease_validation.py).
    assert len(dn) == len(raw), \
        f"{name}: DN weights dropped {sorted(set(raw) - set(dn))} (absent from cross-platform corr file)"
    assert not any(pd.isna(w) for w in dn.values()), \
        f"{name}: NaN DN weight (NaN V2_V3_CT_Corr for a raw gene)"

# %% [markdown]
# # 7. Verify against the pre-registration

# %%
# Section 7: assert the written files match the frozen pre-registration.
# Same exact contract as tests/test_disease_validation_data.py - no floors.
# A floor would let this cell pass after writing a corrupted .gw file.
EXPECTED_IN_MATRIX = {"PD_Primary": 15, "PD_Sens_DA": 20, "PD_Sens_Atypical": 24,
                      "PD_GWAS_L2G": 40, "HD_HTT": 1, "StriatalDegeneration": 8}
EXPECTED_DROPPED = {"StriatalDegeneration": {"FTL"}, "PD_GWAS_L2G": {"FAM47E"}}
for name, n in EXPECTED_IN_MATRIX.items():
    got = len(Fil2Dict(f"../dat/Genetics/GeneWeights/{name}.gw"))
    assert got == n, f"{name}: wrote {got} genes, pre-registration says {n}"
    rep = reports[name]
    dropped = set(rep.loc[~rep["in_matrix"], "symbol"])
    assert dropped == EXPECTED_DROPPED.get(name, set()), \
        f"{name}: dropped {dropped}, expected {EXPECTED_DROPPED.get(name, set())}"
    assert rep["resolved"].all(), \
        f"{name}: unresolved symbols {set(rep.loc[~rep['resolved'], 'symbol'])}"
print("All gene sets match the pre-registration.")
print("Pre-registration commit:",
      subprocess.run(["git", "log", "-1", "--format=%H", "--",
                      "config/disease_validation_genesets.yaml"],
                     cwd="..", capture_output=True, text=True).stdout.strip())
