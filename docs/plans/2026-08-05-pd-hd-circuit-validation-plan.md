# PD / Striatal-Degeneration Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the GENCIC/CCS framework by showing it recovers the nigrostriatal circuit from curated Parkinson's disease genes, with pre-registered ground truth and adversarial controls, in response to reviewer 2.

**Architecture:** New pure-function module `src/disease_validation.py` holds every testable unit (gene-set loading, expression-matched null sampling, recovery statistics). The existing Snakemake bias pipeline is extended with one new null model rather than forked. Three jupytext-paired notebooks in `notebooks_disease_validation/` orchestrate curation, structure-level analysis, and cell-type/MERFISH analysis. Pre-registration artifacts live in tracked `config/` because `dat/` is gitignored.

**Tech Stack:** Python 3.10 (conda env `gencic`), pandas, numpy, scipy.stats, PyYAML, Snakemake 7.x, pytest 8.4.1, jupytext.

**Spec:** `docs/plans/2026-08-05-pd-hd-circuit-validation-design.md`

## Global Constraints

- Conda env `gencic` must be active for every command: `conda activate gencic`.
- Random seed base is **42**. No unseeded `np.random` calls; use `np.random.default_rng(seed)`.
  Null generation derives a per-gene-set seed (`42 + zlib.crc32(geneset.encode()) % 10000`) so
  two gene sets of equal size do not draw byte-identical nulls — PD sets and their negative
  controls must not share Monte Carlo noise. Human ruling 2026-08-05, following CLAUDE.md's
  `base_seed + job_index` convention over a flat 42. Everything else (notebooks, permutation
  tests, samplers called directly) still uses 42.
- All data paths relative to project root or loaded from `config/`. Never reference `/home/jw3514/Work/ASD_Circuits/` or `/mnt/data0/`.
- **DN gene weights are for cell-type (CT) analysis ONLY.** Using them for STR_ISH produces corrupted results.
- Notebooks are jupytext-paired `.py:percent`. **Never use NotebookEdit** — edit the `.py`, then `jupytext --sync <file>.py`. First cell is always `%load_ext autoreload` / `%autoreload 2`.
- Figures: `transparent=True, dpi=300, bbox_inches='tight'`, saved to `results/figures/`. Region colors imported from `src/plot.REGION_COLORS` — never redefined locally.
- Use `color=` not `c=` in `ax.scatter()`. Never `list(some_set)` for ordered labels.
- Default parallelism `n_jobs=10`.
- Structure bias DataFrames are indexed by structure name with columns `EFFECT`, `Rank`, `REGION`. Cell-type bias DataFrames are indexed by cluster ID with columns `EFFECT`, `Rank`.
- Gene weight `.gw` files are headerless CSV: `EntrezID,weight`.
- Commit both `.py` and `.ipynb` for every notebook.

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/disease_validation.py` | All pure functions: gene-set/ground-truth loading, expression-matched sampling, recovery statistics, sensitivity curves. No I/O side effects beyond reading the files it is given. |
| `tests/test_disease_validation.py` | Unit tests for the above. |
| `tests/test_disease_validation_data.py` | Data-contract tests: every curated symbol resolves, every ground-truth structure exists, set sizes match the spec. |
| `config/disease_validation_genesets.yaml` | Curated gene lists. **Tracked — this is the pre-registration.** |
| `config/disease_validation_ground_truth.yaml` | Pre-registered structures and cell types. **Tracked.** |
| `config/circuit_config_disease.yaml` | SA circuit-search config for the six sets. |
| `notebooks_disease_validation/01.Gene_Curation.py/.ipynb` | Builds `.gw` + `.DN.gw` files, fetches Open Targets L2G, writes pre-registration YAMLs. |
| `notebooks_disease_validation/02.STR_Bias_and_Circuits.py/.ipynb` | Structure-level bias, both nulls, recovery metrics, sensitivity, negative controls, CCS, SA circuits. |
| `notebooks_disease_validation/03.CellType_and_MERFISH.py/.ipynb` | Cluster-level bias + recovery, MERFISH concordance. |

**Modified:**

| Path | Change |
|---|---|
| `scripts/script_generate_geneweights.py` | Add expression-decile-matched sampling to `RandomGenes()`. |
| `Snakefile.bias:21-26` | Add `get_null_mode()`; pass `--null_mode` and `--ExpMatch` to the weight generator. |
| `config/config.STR.yaml` | Register 12 gene-set entries (6 sets × uniform + expression-matched). |
| `config/config.SC.DN.yaml` | Register 6 DN gene-set entries. |
| `src/plot.py` | Add `plot_recovery_forest`, `plot_nested_subset_curve`, `plot_circuit_vs_anatomy`. |
| `notebooks_mouse_str/10.Positive_Control_Circuits.py:177-184` | Remove the cell that rewrites `Parkinson.gw` from a hardcoded 5-gene list (spec §11 risk 5). |
| `DATA_MANIFEST.yaml` | Entries for all new data files. |

---

### Task 1: Pre-registration artifacts and loaders

**Files:**
- Create: `config/disease_validation_genesets.yaml`
- Create: `config/disease_validation_ground_truth.yaml`
- Create: `src/disease_validation.py`
- Test: `tests/test_disease_validation_data.py`

**Interfaces:**
- Consumes: `ASD_Circuits.LoadGeneINFO()` → `(HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol)`; `ASD_Circuits.STR2Region()` → `dict[str, str]`.
- Produces: `load_gene_sets(path) -> dict[str, list[dict]]`, `load_ground_truth(path) -> dict[str, dict[str, list[str]]]`, `gene_set_weights(records, symbol2entrez, valid_genes) -> dict[int, float]`.

- [ ] **Step 1: Write `config/disease_validation_genesets.yaml`**

Each record carries `symbol`, `tier`, `syndrome`, `justification`. `tier` is `primary` or `sensitivity`. Gene sets are assembled from tiers by `include` lists so genes are never duplicated across sets.

```yaml
# Pre-registered disease gene sets for GENCIC validation (reviewer 2).
# Spec: docs/plans/2026-08-05-pd-hd-circuit-validation-design.md
# Inclusion criterion: the syndrome must involve anatomical DEGENERATION of the
# target circuit, not merely functional dysfunction of it.
# Circular expression-marker genes are excluded from every primary tier.
version: 1
frozen: '2026-08-05'

gene_pools:
  pd_core:
    - {symbol: SNCA,     syndrome: 'PARK1/4 dominant PD',        justification: 'Definitive dominant PD, Lewy/nigral degeneration'}
    - {symbol: LRRK2,    syndrome: 'PARK8 dominant PD',          justification: 'Definitive dominant PD with nigrostriatal degeneration'}
    - {symbol: VPS35,    syndrome: 'PARK17 dominant PD',         justification: 'Definitive dominant late-onset PD'}
    # LoadGeneINFO() maps approved HGNC symbols only - no aliases, no prev_symbol.
    # GeneSymbol2Entrez['GBA1'] is None; the approved symbol here is GBA (2629).
    # hgnc_symbol overrides the lookup key wherever the current symbol is an alias.
    - {symbol: GBA1, hgnc_symbol: GBA, syndrome: 'Gaucher/PD risk',  justification: 'Largest-effect PD risk gene, Lewy/nigral phenotype'}
    - {symbol: CHCHD2,   syndrome: 'PARK22 dominant PD',         justification: 'Accepted rare dominant PD gene'}
    - {symbol: RAB39B,   syndrome: 'X-linked parkinsonism-ID',   justification: 'Reported Lewy/nigral pathology'}
    - {symbol: PRKN,     syndrome: 'PARK2 recessive EOPD',       justification: 'Definitive recessive EOPD with SN dopaminergic loss'}
    - {symbol: PINK1,    syndrome: 'PARK6 recessive EOPD',       justification: 'Definitive recessive EOPD'}
    - {symbol: PARK7,    syndrome: 'PARK7 recessive EOPD',       justification: 'Definitive recessive EOPD'}
    - {symbol: ATP13A2,  syndrome: 'PARK9 Kufor-Rakeb',          justification: 'Recessive juvenile parkinsonism, nigrostriatal involvement'}
    - {symbol: PLA2G6,   syndrome: 'PARK14',                     justification: 'Neurodegenerative dystonia-parkinsonism, nigrostriatal'}
    - {symbol: FBXO7,    syndrome: 'PARK15 pallidopyramidal',    justification: 'Established recessive parkinsonism, dopaminergic deficit'}
    - {symbol: DNAJC6,   syndrome: 'PARK19 juvenile',            justification: 'Juvenile parkinsonism with dopaminergic neurodegeneration'}
    - {symbol: SYNJ1,    syndrome: 'PARK20 juvenile',            justification: 'Established recessive juvenile parkinsonism'}
    - {symbol: VPS13C,   syndrome: 'PARK23 recessive EOPD',      justification: 'Established recessive early-onset parkinsonism'}
  pd_dopamine_markers:
    - {symbol: TH,       syndrome: 'TH deficiency',              justification: 'CIRCULAR: defining dopaminergic-neuron marker; nondegenerative'}
    - {symbol: SLC6A3,   syndrome: 'DAT deficiency',             justification: 'CIRCULAR: defining nigrostriatal terminal marker; nondegenerative'}
    - {symbol: DDC,      syndrome: 'AADC deficiency',            justification: 'Biochemical monoamine deficiency, no degeneration'}
    - {symbol: GCH1,     syndrome: 'Dopa-responsive dystonia',   justification: 'No nigral degeneration'}
    - {symbol: SPR,      syndrome: 'BH4-pathway deficiency',     justification: 'Biochemical dopamine deficiency, not degenerative'}
  pd_atypical:
    - {symbol: DNAJC13,  syndrome: 'Disputed dominant PD',       justification: 'Disputed candidate, not consensus primary'}
    - {symbol: LRP10,    syndrome: 'Disputed PD/DLB',            justification: 'Disputed candidate'}
    - {symbol: DCTN1,    syndrome: 'Perry syndrome',             justification: 'True nigral degeneration but TDP-43 atypical parkinsonism'}
    - {symbol: MAPT,     syndrome: 'FTDP-17 / PSP',              justification: 'Tau parkinsonism with nigral involvement, not PD'}
    - {symbol: POLG,     syndrome: 'Mitochondrial parkinsonism', justification: 'Nigrostriatal deficit but multisystem'}
    - {symbol: TWNK,     syndrome: 'mtDNA maintenance disease',  justification: 'Parkinsonism, multisystem'}
    - {symbol: SPG11,    syndrome: 'Complicated HSP',            justification: 'Juvenile parkinsonism with dopaminergic deficit'}
    - {symbol: PTRHD1,   syndrome: 'Parkinsonism-ID',            justification: 'Atypical, limited neuropathology'}
    - {symbol: DNAJC12,  syndrome: 'Monoamine cofactor defect',  justification: 'Treatable, nondegenerative'}
  striatal_degeneration:
    - {symbol: HTT,      syndrome: 'Huntington disease',         justification: 'CAG expansion, caudate/putamen MSN degeneration'}
    - {symbol: JPH3,     syndrome: 'HDL2',                       justification: 'HD phenocopy with striatal/cortical degeneration'}
    - {symbol: TBP,      syndrome: 'SCA17 / HDL4',               justification: 'HD-like chorea with basal ganglia degeneration'}
    - {symbol: VPS13A,   syndrome: 'Chorea-acanthocytosis',      justification: 'Marked caudate/striatal atrophy'}
    - {symbol: XK,       syndrome: 'McLeod syndrome',            justification: 'Chorea with caudate/putamen degeneration'}
    - {symbol: FTL,      syndrome: 'Neuroferritinopathy',        justification: 'Marked striatal/pallidal degeneration; ABSENT from Z2 matrix'}
    - {symbol: ATN1,     syndrome: 'DRPLA',                      justification: 'WEAK: dentatorubral-pallidoluysian, not striatal-selective'}
    - {symbol: PRNP,     syndrome: 'HDL1',                       justification: 'Very rare prion HD phenocopy'}
    - {symbol: C9orf72,  syndrome: 'HD phenocopy',               justification: 'WEAK: commonest HD phenocopy but FTLD/ALS pathology'}

  # PD_GWAS_L2G membership is FROZEN HERE, not generated at runtime, so it is
  # covered by the pre-registration commit. Derived 2026-08-05 from Open Targets
  # locus-to-gene ML score >= 0.5 pooled over seven PD GWAS (GCST009325,
  # GCST009324, GCST004902, GCST002544, GCST003984, GCST010049, GCST009512),
  # minus the artifact loci FLG/HRNR/MUC19. Notebook 01 re-derives and ASSERTS
  # equality against this list; it never redefines it.
  # NOTE: written as a YAML flow sequence of mappings - one item per line would
  # be 41 lines of noise. Validate with `yaml.safe_load` in the Task 1 test.
  pd_gwas_l2g: [
    {symbol: LRRK2}, {symbol: SNCA}, {symbol: TMEM175}, {symbol: GPNMB},
    {symbol: BST1}, {symbol: MCCC1}, {symbol: ACMSD}, {symbol: RIT2},
    {symbol: STK39}, {symbol: SH3GL2}, {symbol: CTSB}, {symbol: ITPKB},
    {symbol: SV2C}, {symbol: DLG2}, {symbol: MAPT}, {symbol: RAB29},
    {symbol: SIPA1L2}, {symbol: NUCKS1}, {symbol: INPP5F}, {symbol: TMEM163},
    {symbol: FGF20}, {symbol: CAMK2D}, {symbol: HIP1R}, {symbol: BAG3},
    {symbol: GALC}, {symbol: TOX3}, {symbol: KLHL7}, {symbol: IGSF9B},
    {symbol: SLC45A3}, {symbol: MAP4K4}, {symbol: GPR65}, {symbol: FAM47E},
    {symbol: PRICKLE1}, {symbol: PLEKHH1}, {symbol: NDUFAF2}, {symbol: ITGA8},
    {symbol: TMEM229B}, {symbol: PM20D1}, {symbol: ERCC8}, {symbol: PKP2},
    {symbol: SLC50A1},
  ]

gene_sets:
  PD_Primary:          {include: [pd_core], description: 'Mendelian PD with nigrostriatal degeneration; no dopamine markers'}
  PD_Sens_DA:          {include: [pd_core, pd_dopamine_markers], description: 'PD_Primary + dopamine synthesis/transport genes'}
  PD_Sens_Atypical:    {include: [pd_core, pd_atypical], description: 'PD_Primary + disputed and parkinsonism-plus genes'}
  PD_GWAS_L2G:         {include: [pd_gwas_l2g], description: 'Common-variant tier via Open Targets L2G >= 0.5; NOT nearest-gene'}
  HD_HTT:              {include: [], explicit: [HTT], description: 'Huntington disease proper; expected negative'}
  StriatalDegeneration: {include: [striatal_degeneration], description: 'Mendelian striatal degeneration; NOT a Huntington validation'}

excluded:
  refuted_pd: [TMEM230, UCHL1, HTRA2, EIF4G1, GIGYF2, NR4A2, PODXL, RIC3, PSAP]
  pd_not_nigrostriatal: [SLC30A10, ATP1A3, PRKRA, WDR45, PANK2, C19orf12, ARSA, VPS16]
  chorea_functional_not_degenerative: [NKX2-1, FRRS1L, ADCY5, GPR88]
  chorea_circular_striatal_markers: [PDE10A, PDE2A, PDE8B]
  l2g_artifact_loci: [FLG, HRNR, MUC19]
```

- [ ] **Step 2: Write `config/disease_validation_ground_truth.yaml`**

```yaml
# Pre-registered ground truth. Frozen BEFORE any bias is computed.
# Verified 2026-08-05: every structure name exists in STR2Region(),
# AllenMouseBrain_Z2bias.parquet, and InfoMat.Ipsi.csv.
version: 1
frozen: '2026-08-05'

structures:
  parkinson:
    core:
      - Substantia_nigra_compact_part
      - Substantia_nigra_reticular_part
      - Ventral_tegmental_area
      - Caudoputamen
      - Fundus_of_striatum
      - Nucleus_accumbens
      - Globus_pallidus_external_segment
      - Globus_pallidus_internal_segment
      - Subthalamic_nucleus
      - Ventral_anterior_lateral_complex_of_the_thalamus
      - Ventral_medial_nucleus_of_the_thalamus
      - Primary_motor_area
      - Secondary_motor_area
    braak_early:
      - Dorsal_nucleus_raphe
      - Main_olfactory_bulb
      - Anterior_olfactory_nucleus
      - Pedunculopontine_nucleus
  striatal:
    core:
      - Caudoputamen
      - Nucleus_accumbens
      - Fundus_of_striatum
      - Globus_pallidus_external_segment
      - Globus_pallidus_internal_segment
      - Substantia_nigra_reticular_part
      - Subthalamic_nucleus
    late_stage:
      - Primary_motor_area
      - Secondary_motor_area

cell_type_subclasses:
  parkinson:
    core: ['SNc-VTA-RAmb Foxa1 Dopa']
  striatal:
    core: ['STR D1 Gaba', 'STR D2 Gaba', 'STR D1 Sema5a Gaba', 'ACB-BST-FS D1 Gaba']
  basal_ganglia_context:
    - 'GPe-SI Sox6 Cyp26b1 Gaba'
    - 'GPi Tbr1 Cngb3 Gaba-Glut'
    - 'SNr Six3 Gaba'
    - 'SNr-VTA Pax5 Npas1 Gaba'
    - 'STN-PSTN Pitx2 Glut'

# Structures NOT available in the 213-structure atlas; Braak stages 1-2 are
# therefore only partially testable. Stated in the manuscript, not worked around.
unavailable: [Locus_coeruleus, Dorsal_motor_nucleus_of_the_vagus_nerve, Zona_incerta]

notes:
  gene_sets_to_ground_truth:
    PD_Primary: parkinson
    PD_Sens_DA: parkinson
    PD_Sens_Atypical: parkinson
    PD_GWAS_L2G: parkinson
    HD_HTT: striatal
    StriatalDegeneration: striatal
```

- [ ] **Step 3: Write the failing data-contract test**

```python
# tests/test_disease_validation_data.py
import sys, os
import pandas as pd
import pytest

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))
from disease_validation import (load_gene_sets, load_ground_truth,
                                gene_set_weights, gene_set_report)
from ASD_Circuits import LoadGeneINFO, STR2Region

GENESETS = "config/disease_validation_genesets.yaml"
GROUND_TRUTH = "config/disease_validation_ground_truth.yaml"
Z2 = "dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet"

EXPECTED_SIZES = {
    "PD_Primary": 15,
    "PD_Sens_DA": 20,
    "PD_Sens_Atypical": 24,
    "PD_GWAS_L2G": 41,
    "HD_HTT": 1,
    "StriatalDegeneration": 9,
}
# FTL is absent from the Z2 matrix (spec section 3.6).
# Verified 2026-08-05: all 41 PD_GWAS_L2G symbols resolve; exactly one
# (FAM47E) is absent from the Z2 matrix, so 40 are usable.
EXPECTED_IN_MATRIX = {**EXPECTED_SIZES, "StriatalDegeneration": 8, "PD_GWAS_L2G": 40}
EXPECTED_DROPPED = {"StriatalDegeneration": {"FTL"}, "PD_GWAS_L2G": {"FAM47E"}}


@pytest.fixture(scope="module")
def gene_info():
    return LoadGeneINFO()


@pytest.fixture(scope="module")
def z2_genes():
    return set(pd.read_parquet(Z2).index)


def test_gene_set_sizes_match_spec():
    sets = load_gene_sets(GENESETS)
    assert {k: len(v) for k, v in sets.items()} == EXPECTED_SIZES


def test_every_symbol_resolves_to_entrez(gene_info):
    """LoadGeneINFO maps approved HGNC symbols only - aliases must carry
    an explicit hgnc_symbol override or they silently vanish from the set."""
    _, _, sym2entrez, _ = gene_info
    sets = load_gene_sets(GENESETS)
    unresolved = {
        name: [r["symbol"] for r in recs
               if sym2entrez.get(r.get("hgnc_symbol", r["symbol"])) is None]
        for name, recs in sets.items()
    }
    assert all(not v for v in unresolved.values()), unresolved


def test_gba1_alias_resolves_to_entrez_2629(gene_info):
    """Regression guard: GeneSymbol2Entrez['GBA1'] is None in this repo."""
    _, _, sym2entrez, _ = gene_info
    assert sym2entrez.get("GBA1") is None, "repo changed; revisit the alias override"
    rec = [r for r in load_gene_sets(GENESETS)["PD_Primary"] if r["symbol"] == "GBA1"]
    assert len(rec) == 1 and rec[0]["hgnc_symbol"] == "GBA"
    assert int(sym2entrez[rec[0]["hgnc_symbol"]]) == 2629


def test_matrix_coverage_matches_expectation(gene_info, z2_genes):
    """Exact counts and exact dropped sets - a floor would let genes vanish silently.

    Dropped symbols come from gene_set_report(), which already tracks resolution
    and matrix membership per gene. Recomputing them inline would raise on an
    unresolved symbol instead of reporting it, which is the opposite of what a
    data-contract test should do.
    """
    _, _, sym2entrez, _ = gene_info
    sets = load_gene_sets(GENESETS)
    for name, recs in sets.items():
        w = gene_set_weights(recs, sym2entrez, z2_genes)
        assert len(w) == EXPECTED_IN_MATRIX[name], (name, len(w))
        rep = gene_set_report(recs, sym2entrez, z2_genes)
        dropped = set(rep.loc[~rep["in_matrix"], "symbol"])
        assert dropped == EXPECTED_DROPPED.get(name, set()), (name, dropped)
        assert rep["resolved"].all(), (name, set(rep.loc[~rep["resolved"], "symbol"]))


def test_ground_truth_structures_exist_in_infomat():
    """Circuit search consumes InfoMat.Ipsi.csv; ground truth must be in it too."""
    im = pd.read_csv("dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv",
                     index_col=0)
    gt = load_ground_truth(GROUND_TRUTH)
    for disease, groups in gt["structures"].items():
        for group, names in groups.items():
            missing = [n for n in names if n not in im.index]
            assert not missing, f"{disease}/{group} missing from InfoMat: {missing}"


def test_frozen_cell_type_subclasses_map_to_clusters():
    """A renamed subclass would silently yield an empty ground-truth set."""
    import re
    ct = pd.read_parquet("dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    subclasses = {re.sub(r"^\d+\s+", "", c).rsplit("_", 1)[0] for c in ct.columns}
    gt = load_ground_truth(GROUND_TRUTH)
    wanted = set()
    for disease, groups in gt["cell_type_subclasses"].items():
        wanted |= set(groups["core"]) if isinstance(groups, dict) else set(groups)
    missing = wanted - subclasses
    assert not missing, f"subclasses absent from the CT matrix: {missing}"


def test_pd_gwas_excludes_artifact_loci():
    syms = {r["symbol"] for r in load_gene_sets(GENESETS)["PD_GWAS_L2G"]}
    assert not (syms & {"FLG", "HRNR", "MUC19"})


def test_ground_truth_structures_exist_in_atlas(z2_genes):
    gt = load_ground_truth(GROUND_TRUTH)
    atlas = set(pd.read_parquet(Z2).columns) & set(STR2Region())
    for disease, groups in gt["structures"].items():
        for group, names in groups.items():
            missing = [n for n in names if n not in atlas]
            assert not missing, f"{disease}/{group}: {missing}"


def test_no_circular_marker_in_any_primary_set():
    circular = {"TH", "SLC6A3", "DDC", "GCH1", "SPR",
                "PDE10A", "GPR88", "ADCY5", "RASD2", "DRD2"}
    sets = load_gene_sets(GENESETS)
    for name in ("PD_Primary", "HD_HTT", "StriatalDegeneration"):
        syms = {r["symbol"] for r in sets[name]}
        assert not (syms & circular), f"{name} contains circular markers"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'disease_validation'`

- [ ] **Step 5: Implement the loaders**

```python
# src/disease_validation.py
"""Pure functions for the PD / striatal-degeneration validation of GENCIC.

Spec: docs/plans/2026-08-05-pd-hd-circuit-validation-design.md
Every function here is side-effect free apart from reading files it is handed.
"""
import numpy as np
import pandas as pd
import yaml

DEFAULT_SEED = 42


def load_gene_sets(path):
    """Expand the pooled YAML curation into {set_name: [record, ...]}.

    Each record is the raw dict from the YAML (symbol/syndrome/justification)
    with a 'tier' key naming the pool it came from.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    pools = {
        name: [dict(r, tier=name) for r in recs]
        for name, recs in cfg["gene_pools"].items()
    }
    by_symbol = {r["symbol"]: r for recs in pools.values() for r in recs}
    out = {}
    for name, spec in cfg["gene_sets"].items():
        records, seen = [], set()
        for pool in spec.get("include", []):
            for r in pools[pool]:
                if r["symbol"] not in seen:
                    seen.add(r["symbol"])
                    records.append(r)
        for sym in spec.get("explicit", []):
            if sym not in seen:
                seen.add(sym)
                records.append(by_symbol.get(sym, {"symbol": sym, "tier": "explicit"}))
        out[name] = records
    return out


def load_ground_truth(path):
    with open(path) as f:
        return yaml.safe_load(f)


def lookup_symbol(record):
    """The key to look up in GeneSymbol2Entrez.

    LoadGeneINFO() indexes approved HGNC symbols only - not alias_symbol, not
    prev_symbol. A record whose display symbol is an alias (e.g. GBA1, current
    symbol for what this repo's HGNC table still calls GBA) must declare
    hgnc_symbol or it will be silently dropped from the gene set.
    """
    return record.get("hgnc_symbol", record["symbol"])


def gene_set_weights(records, symbol2entrez, valid_genes, weight=1.0):
    """Map curation records to {entrez: weight}, dropping genes absent from the matrix.

    valid_genes is any container of entrez ids (the expression matrix index).
    """
    valid = set(int(g) for g in valid_genes)
    out = {}
    for r in records:
        e = symbol2entrez.get(lookup_symbol(r))
        if e is None:
            continue
        e = int(e)
        if e in valid:
            out[e] = weight
    return out


def gene_set_report(records, symbol2entrez, valid_genes):
    """Per-gene resolution status, for the methods table. Returns a DataFrame."""
    valid = set(int(g) for g in valid_genes)
    rows = []
    for r in records:
        e = symbol2entrez.get(lookup_symbol(r))
        rows.append({
            "symbol": r["symbol"],
            "hgnc_symbol": lookup_symbol(r),
            "entrez": int(e) if e is not None else None,
            "tier": r.get("tier", ""),
            "syndrome": r.get("syndrome", ""),
            "justification": r.get("justification", ""),
            "resolved": e is not None,
            "in_matrix": e is not None and int(e) in valid,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation_data.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit — this is the pre-registration**

```bash
git add config/disease_validation_genesets.yaml config/disease_validation_ground_truth.yaml \
        src/disease_validation.py tests/test_disease_validation_data.py
git commit -m "Pre-register PD / striatal-degeneration gene sets and ground truth

Frozen before any bias is computed. Cite this commit hash in the rebuttal."
git rev-parse HEAD   # record this hash in notebook 02
```

---

### Task 2: Expression-decile-matched null sampler

**Files:**
- Modify: `src/disease_validation.py`
- Test: `tests/test_disease_validation.py`

**Interfaces:**
- Consumes: `dat/allen-mouse-exp/ExpMatchFeatures.csv` (columns `EXP`, `Rank`, `quantile`, indexed by entrez).
- Produces: `expression_decile_map(exp_df, valid_genes, n_bins=10) -> pd.Series`, `sample_expression_matched(target, decile_map, n_sims, rng) -> np.ndarray` of shape `(len(target_in_map), n_sims)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_disease_validation.py
import sys, os
import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))
from disease_validation import expression_decile_map, sample_expression_matched


@pytest.fixture
def toy_exp():
    # 100 genes with strictly increasing expression -> 10 clean deciles of 10
    return pd.DataFrame({"EXP": np.arange(100, dtype=float)}, index=np.arange(1000, 1100))


def test_decile_map_assigns_ten_equal_bins(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    assert dm.nunique() == 10
    assert dm.value_counts().unique().tolist() == [10]


def test_sampled_genes_preserve_decile_composition(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1001, 1050, 1099]          # deciles 0, 0, 5, 9
    rng = np.random.default_rng(42)
    draws = sample_expression_matched(target, dm, n_sims=50, rng=rng)
    assert draws.shape == (4, 50)
    want = dm.loc[target].value_counts().sort_index()
    for j in range(draws.shape[1]):
        got = dm.loc[draws[:, j]].value_counts().sort_index()
        assert got.equals(want)


def test_sampling_is_without_replacement_within_a_sim(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1001, 1002]                 # three genes, same decile
    rng = np.random.default_rng(42)
    draws = sample_expression_matched(target, dm, n_sims=100, rng=rng)
    for j in range(draws.shape[1]):
        assert len(set(draws[:, j])) == 3


def test_sampling_is_reproducible_under_a_fixed_seed(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1050, 1099]
    a = sample_expression_matched(target, dm, 20, np.random.default_rng(42))
    b = sample_expression_matched(target, dm, 20, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_single_gene_set_is_supported(toy_exp):
    """HD_HTT is one gene; the null must still be drawable."""
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    draws = sample_expression_matched([1050], dm, 30, np.random.default_rng(42))
    assert draws.shape == (1, 30)
    assert (dm.loc[draws[0]] == dm.loc[1050]).all()


def test_genes_absent_from_the_map_are_dropped(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    draws = sample_expression_matched([1000, 999999], dm, 5, np.random.default_rng(42))
    assert draws.shape == (1, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -v`
Expected: FAIL — `ImportError: cannot import name 'expression_decile_map'`

- [ ] **Step 3: Implement the sampler**

Append to `src/disease_validation.py`:

```python
def expression_decile_map(exp_df, valid_genes, n_bins=10, exp_col="EXP"):
    """Bin genes into equal-count expression deciles.

    Ranks before binning so ties cannot collapse a bin. Returns a Series
    entrez -> bin index, restricted to genes present in both inputs.
    """
    idx = pd.Index([int(g) for g in valid_genes])
    exp = exp_df.copy()
    exp.index = exp.index.astype(int)
    shared = exp.index.intersection(idx)
    vals = exp.loc[shared, exp_col].rank(method="first")
    bins = pd.qcut(vals, n_bins, labels=False)
    return pd.Series(bins.values, index=shared, name="decile")


def sample_expression_matched(target, decile_map, n_sims, rng):
    """Draw n_sims gene sets matching target's per-decile composition.

    Sampling is without replacement within a simulation. Genes in target that
    are absent from decile_map are dropped. Returns shape (n_kept, n_sims).
    """
    kept = [int(g) for g in target if int(g) in decile_map.index]
    if not kept:
        raise ValueError("no target genes present in decile_map")
    counts = decile_map.loc[kept].value_counts()
    pools = {d: decile_map.index[decile_map == d].to_numpy() for d in counts.index}
    for d, k in counts.items():
        if len(pools[d]) < k:
            raise ValueError(f"decile {d} has {len(pools[d])} genes, need {k}")
    out = np.empty((len(kept), n_sims), dtype=np.int64)
    for j in range(n_sims):
        drawn = []
        for d, k in counts.items():
            drawn.extend(rng.choice(pools[d], size=k, replace=False))
        out[:, j] = drawn
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/disease_validation.py tests/test_disease_validation.py
git commit -m "Add expression-decile-matched null sampler"
```

---

### Task 3: Wire the expression-matched null into the pipeline

**Files:**
- Modify: `scripts/script_generate_geneweights.py`
- Modify: `Snakefile.bias:21-26,37-57`
- Test: `tests/test_disease_validation.py` (append)

**Interfaces:**
- Consumes: `sample_expression_matched` from Task 2.
- Produces: CLI flag `--null_mode {uniform,mutability,expmatched}` and `--ExpMatch <path>` on `script_generate_geneweights.py`; Snakefile helper `get_null_mode(geneset) -> str`.

- [ ] **Step 1: Write the failing CLI test**

```python
# append to tests/test_disease_validation.py
import subprocess, tempfile, csv


def test_generate_geneweights_expmatched_cli(tmp_path):
    """The script must emit an entrez x n_sims table whose first column is the
    original weight, matching the existing sibling/random output contract."""
    gw = tmp_path / "toy.gw"
    gw.write_text("2629,1\n120892,1\n6622,1\n")     # GBA, LRRK2, SNCA
    out = tmp_path / "null.csv"
    r = subprocess.run(
        ["python", "scripts/script_generate_geneweights.py",
         "--WeightDF", str(gw),
         "--SpecMat", "dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet",
         "--n_sims", "25",
         "--GeneProb", "None",
         "--null_mode", "expmatched",
         "--ExpMatch", "dat/allen-mouse-exp/ExpMatchFeatures.csv",
         "--outfile", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    df = pd.read_csv(out, index_col=0)
    assert df.columns[0] == "GeneWeight"
    assert df.shape == (3, 26)
    assert set(df.index) == {2629, 120892, 6622}
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py::test_generate_geneweights_expmatched_cli -v`
Expected: FAIL — `error: unrecognized arguments: --null_mode`

- [ ] **Step 3: Add the argument and branch to the script**

In `scripts/script_generate_geneweights.py`, add to `GetOptions()`:

```python
    parser.add_argument('--null_mode', type=str, default='uniform',
                        choices=['uniform', 'mutability', 'expmatched'],
                        help='Null sampling scheme for the _random output')
    parser.add_argument('--ExpMatch', type=str, default=None,
                        help='ExpMatchFeatures.csv, required when null_mode=expmatched')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; the existing samplers were unseeded')
```

**All three samplers must be seeded, not just the new one.** `RandomGenes()` and
`SiblingGenes()` currently call the global `np.random.choice` with no seed
(`script_generate_geneweights.py:115,156`), so every rerun produces a different
null and no published p-value is reproducible. Replace the global calls:

```python
# in RandomGenes(...) and SiblingGenes(...), add a seed parameter and use it
def RandomGenes(ExpMat, WeightDF, outfile, GeneProb, n_sims=10000, seed=42):
    ...
    rng = np.random.default_rng(seed)
    ...
    for i in range(n_sims):
        Genes = rng.choice(gene_pool, size=len(Gene_Weights), p=gene_probs, replace=False)
        sim_matrix[:, i] = Genes
```

This changes the null draws for existing gene sets on any future rerun. Do **not**
regenerate existing published nulls as part of this work — only new gene sets go
through the seeded path. Note the behaviour change in the commit message.

Add a new function next to `RandomGenes`:

```python
def ExpMatchedGenes(ExpMat, WeightDF, outfile, ExpMatchFil, n_sims=10000, seed=42):
    """Expression-decile-matched null. Same output contract as RandomGenes."""
    sys.path.insert(1, os.path.join(ProjDIR, 'src'))
    from disease_validation import expression_decile_map, sample_expression_matched

    ExpMat = pd.read_parquet(ExpMat) if '.parquet' in ExpMat else pd.read_csv(ExpMat, index_col=0)
    valid_genes = ExpMat.index.values

    WeightDF = pd.read_csv(WeightDF, header=None)
    ValidWeightDF = WeightDF[WeightDF[0].isin(valid_genes)]
    entrez_ids = ValidWeightDF[0].values
    Gene_Weights = ValidWeightDF[1].values

    exp_df = pd.read_csv(ExpMatchFil, index_col=0)
    decile_map = expression_decile_map(exp_df, valid_genes)
    rng = np.random.default_rng(seed)
    sims = sample_expression_matched(entrez_ids, decile_map, n_sims, rng)

    out_df = pd.DataFrame(sims, index=entrez_ids,
                          columns=[str(i) for i in range(n_sims)])
    out_df.insert(0, "GeneWeight", Gene_Weights)
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(outfile)
    print(f"Saved {n_sims} expression-matched simulations to {outfile}")
```

In `main()`, dispatch on `args.null_mode` — when `expmatched`, call `ExpMatchedGenes(...)` for the `_random` output instead of `RandomGenes(...)`; the `_sibling` output is still produced unchanged so the Snakemake rule's two declared outputs both exist.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py::test_generate_geneweights_expmatched_cli -v`
Expected: PASS

- [ ] **Step 5: Wire it into `Snakefile.bias`**

Replace `get_gene_prob` (line 21) with:

```python
def get_null_mode(geneset):
    return config["gene_sets"][geneset].get("null_model", "uniform")

def get_gene_prob(geneset):
    """GeneProb path for the mutability null, else 'None'."""
    if get_null_mode(geneset) == "mutability":
        return os.path.join(PROJDIR, config["data_files"]["gene_prob"])
    return "None"

def get_expmatch():
    return os.path.join(PROJDIR, config["data_files"]["exp_match_features"])

def get_geneweights(geneset):
    """Join with PROJDIR. os.path.join returns the second arg unchanged when it
    is absolute, so the ~30 existing absolute entries keep working while new
    entries can be written relative."""
    return os.path.join(PROJDIR, config["gene_sets"][geneset]["geneweights"])
```

In `rule generate_geneweights`, change `input.geneweights` to
`lambda wc: get_geneweights(wc.geneset)`, add a **conditional** input
`expmatch=lambda wc: get_expmatch() if get_null_mode(wc.geneset) == "expmatched" else []`
so gene sets on the other nulls keep their existing input list, add
`params: null_mode=..., seed=42, expmatch=lambda wc: get_expmatch()`, and append:
`--null_mode {params.null_mode} --ExpMatch {params.expmatch} --seed {params.seed}`

**Rerun hazard.** Editing this rule changes its `code`, `input` and `params`
signatures, and Snakemake 7's default `--rerun-triggers` includes all three. A
bare invocation could therefore regenerate the *published* ASD/DDD/NT nulls and
silently change every p-value in the manuscript. Mitigations, all required:
the conditional input above; passing explicit targets only; and verifying with
`--rerun-triggers mtime` before any run that touches an existing gene set.

Add to `data_files` in `config/config.STR.yaml` and `config/config.SC.DN.yaml`:
`exp_match_features: "dat/allen-mouse-exp/ExpMatchFeatures.csv"`

**Guard against `rule all`.** `rule all` expands both `_sibling` and `_random`
outputs for every registered gene set (`Snakefile.bias:28`), so a bare
`snakemake -s Snakefile.bias --configfile config/config.STR.yaml` would build
meaningless `*_EM_bias_addP_sibling.csv` files. Always pass explicit targets, as
every command in Task 7 does. Add this comment above `rule all`:

```python
# WARNING: the disease-validation _EM entries exist only to obtain a second null
# under the existing output contract. Their _sibling outputs are meaningless.
# Pass explicit targets rather than invoking this rule with those configs.
```

- [ ] **Step 6: Verify no regression on an existing gene set**

Run:
```bash
conda run -n gencic snakemake -s Snakefile.bias --configfile config/config.STR.yaml \
  -n results/STR_ISH/ASD_All_bias_addP_random.csv
```
Expected: dry-run resolves with no error and reports `nothing to be done` (output already exists and inputs unchanged).

- [ ] **Step 7: Commit**

```bash
git add scripts/script_generate_geneweights.py Snakefile.bias config/config.STR.yaml \
        config/config.SC.DN.yaml tests/test_disease_validation.py
git commit -m "Add expression-matched null model to the bias pipeline"
```

---

### Task 4: Recovery statistics

**Files:**
- Modify: `src/disease_validation.py`
- Test: `tests/test_disease_validation.py` (append)

**Interfaces:**
- Produces: `recovery_stats(bias_df, ground_truth, effect_col='EFFECT') -> dict` with keys `n_ground_truth`, `n_missing`, `u_stat`, `p_mannwhitney`, `auroc`, `precision_at_20`, `median_rank`; and `recovery_permutation_p(bias_df, ground_truth, n_perm=10000, seed=42) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_disease_validation.py
from disease_validation import recovery_stats, recovery_permutation_p


def _bias(order):
    """Build a bias frame whose EFFECT decreases down the given structure order."""
    return pd.DataFrame(
        {"EFFECT": np.linspace(1.0, 0.0, len(order))}, index=list(order)
    )


def test_perfect_recovery_gives_auroc_one():
    # 3 positives over 5 negatives: exact one-sided p = 1/C(8,3) = 0.0179.
    # With 2-over-4 the best achievable p is 1/C(6,2) = 0.0667, which would
    # make a p<0.05 assertion mathematically impossible.
    df = _bias(["A", "B", "C", "D", "E", "F", "G", "H"])
    s = recovery_stats(df, ["A", "B", "C"])
    assert s["auroc"] == 1.0
    assert s["p_mannwhitney"] < 0.05
    assert s["n_ground_truth"] == 3


def test_worst_recovery_gives_auroc_zero():
    df = _bias(["A", "B", "C", "D", "E", "F"])
    s = recovery_stats(df, ["E", "F"])
    assert s["auroc"] == 0.0
    assert s["p_mannwhitney"] > 0.5


def test_missing_ground_truth_structures_are_counted_not_fatal():
    df = _bias(["A", "B", "C"])
    s = recovery_stats(df, ["A", "NOT_IN_ATLAS"])
    assert s["n_ground_truth"] == 1
    assert s["n_missing"] == 1


def test_precision_at_20_counts_hits_in_the_top_20():
    order = [f"S{i:03d}" for i in range(100)]
    df = _bias(order)
    s = recovery_stats(df, ["S000", "S001", "S050"])
    assert s["precision_at_20"] == pytest.approx(2 / 20)


def test_permutation_p_is_small_for_perfect_recovery():
    order = [f"S{i:03d}" for i in range(213)]
    df = _bias(order)
    p = recovery_permutation_p(df, order[:13], n_perm=2000, seed=42)
    assert p < 0.01


def test_permutation_p_is_reproducible():
    order = [f"S{i:03d}" for i in range(213)]
    df = _bias(order)
    a = recovery_permutation_p(df, order[:13], n_perm=500, seed=42)
    b = recovery_permutation_p(df, order[:13], n_perm=500, seed=42)
    assert a == b


def test_all_ground_truth_missing_raises():
    df = _bias(["A", "B"])
    with pytest.raises(ValueError):
        recovery_stats(df, ["X", "Y"])


# --- Gene-set null: the test that actually differs between null models ---
from disease_validation import recovery_null_aurocs, empirical_p


def test_null_aurocs_one_per_simulation():
    order = [f"S{i:03d}" for i in range(20)]
    rng = np.random.default_rng(42)
    null = pd.DataFrame(rng.normal(size=(20, 50)), index=order,
                        columns=[str(i) for i in range(50)])
    aurocs = recovery_null_aurocs(null, order[:5])
    assert aurocs.shape == (50,)
    assert ((aurocs >= 0) & (aurocs <= 1)).all()


def test_null_auroc_distribution_is_centred_on_half():
    order = [f"S{i:03d}" for i in range(50)]
    rng = np.random.default_rng(42)
    null = pd.DataFrame(rng.normal(size=(50, 400)), index=order,
                        columns=[str(i) for i in range(400)])
    aurocs = recovery_null_aurocs(null, order[:10])
    assert 0.42 < np.median(aurocs) < 0.58


def test_null_auroc_handles_ties_like_scipy():
    """Regression: a double-argsort gives 0.5 here; the correct answer is 0.75."""
    null = pd.DataFrame({"0": [1.0, 1.0, 1.0, 0.0]}, index=list("ABCD"))
    got = recovery_null_aurocs(null, ["A", "B"])
    pos, neg = null.loc[["A", "B"], "0"], null.loc[["C", "D"], "0"]
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    assert got[0] == pytest.approx(u / (len(pos) * len(neg))) == pytest.approx(0.75)


def test_empirical_p_is_add_one_smoothed():
    assert empirical_p(1.0, np.zeros(99)) == pytest.approx(1 / 100)
    assert empirical_p(-1.0, np.zeros(99)) == pytest.approx(100 / 100)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -k recovery -v`
Expected: FAIL — `ImportError: cannot import name 'recovery_stats'`

- [ ] **Step 3: Implement**

Append to `src/disease_validation.py`:

```python
from scipy.stats import mannwhitneyu, rankdata


def recovery_stats(bias_df, ground_truth, effect_col="EFFECT"):
    """Do the pre-registered structures rank above the rest on bias?

    One-sided Mann-Whitney U (ground truth greater). AUROC is derived from U,
    so it is the exact rank-based area, not a threshold sweep.
    """
    scores = bias_df[effect_col].dropna()
    present = [s for s in ground_truth if s in scores.index]
    missing = [s for s in ground_truth if s not in scores.index]
    if not present:
        raise ValueError("no ground-truth structures present in bias_df")
    mask = scores.index.isin(present)
    pos, neg = scores[mask], scores[~mask]
    if len(neg) == 0:
        raise ValueError("no background structures to compare against")
    u, p = mannwhitneyu(pos, neg, alternative="greater")
    ranks = bias_df[effect_col].rank(ascending=False)
    top20 = set(scores.nlargest(20).index)
    return {
        "n_ground_truth": len(present),
        "n_missing": len(missing),
        "missing": missing,
        "u_stat": float(u),
        "p_mannwhitney": float(p),
        "auroc": float(u) / (len(pos) * len(neg)),
        "precision_at_20": len(top20 & set(present)) / 20.0,
        "median_rank": float(ranks[present].median()),
    }


def recovery_null_aurocs(null_bias_df, ground_truth):
    """AUROC of the ground-truth set under every null GENE SET simulation.

    null_bias_df is the (n_structures x n_sims) matrix written by the bias
    pipeline to results/{analysis}/null_bias/{geneset}_null_bias_{null}.parquet.
    Each column is the structure bias profile of one null gene set.

    THIS is the statistic that distinguishes null models. The observed EFFECT
    column is computed from the real gene set and is byte-identical regardless
    of which null was configured, so comparing recovery_stats() across nulls
    compares nothing. Confirmed empirically: for ASD_All the uniform and
    sibling bias files differ in P-value but max|dEFFECT| is exactly 0.0.
    """
    present = [s for s in ground_truth if s in null_bias_df.index]
    if not present:
        raise ValueError("no ground-truth structures present in null_bias_df")
    mask = null_bias_df.index.isin(present)
    n_pos, n_neg = int(mask.sum()), int((~mask).sum())
    if n_neg == 0:
        raise ValueError("no background structures to compare against")
    vals = null_bias_df.to_numpy(dtype=float)
    # AUROC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg),
    # the vectorised Mann-Whitney U formulation.
    # MUST use rankdata(method="average"): a double-argsort assigns arbitrary
    # distinct ranks to tied values and silently gives the wrong answer. For
    # scores [1,1,1,0] with positives {A,B}, argsort yields 0.5 where the
    # correct (and scipy) answer is 0.75.
    ranks = rankdata(vals, axis=0, method="average")
    pos_rank_sum = ranks[mask, :].sum(axis=0)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def empirical_p(observed, null_values):
    """Add-one-smoothed one-sided p: P(null >= observed)."""
    null_values = np.asarray(null_values, dtype=float)
    return (int((null_values >= observed).sum()) + 1) / (len(null_values) + 1)


def recovery_permutation_p(bias_df, ground_truth, n_perm=10000,
                           seed=DEFAULT_SEED, effect_col="EFFECT"):
    """Permutation p for the AUROC, drawing random STRUCTURE sets of matched size.

    Distinct from recovery_null_aurocs: this asks whether these particular
    structures sit unusually high in this one ranking (a structure-label
    permutation), while recovery_null_aurocs asks whether this gene set beats
    null gene sets. Both are reported; they are not interchangeable.
    """
    scores = bias_df[effect_col].dropna()
    present = [s for s in ground_truth if s in scores.index]
    if not present:
        raise ValueError("no ground-truth structures present in bias_df")
    observed = recovery_stats(bias_df, ground_truth, effect_col)["auroc"]
    rng = np.random.default_rng(seed)
    all_structs = scores.index.to_numpy()
    hits = 0
    for _ in range(n_perm):
        draw = rng.choice(all_structs, size=len(present), replace=False)
        mask = scores.index.isin(draw)
        u, _ = mannwhitneyu(scores[mask], scores[~mask], alternative="greater")
        if float(u) / (mask.sum() * (~mask).sum()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -k recovery -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/disease_validation.py tests/test_disease_validation.py
git commit -m "Add pre-registered circuit recovery statistics"
```

---

### Task 5: Gene-set size sensitivity

**Files:**
- Modify: `src/disease_validation.py`
- Test: `tests/test_disease_validation.py` (append)

**Interfaces:**
- Consumes: `recovery_stats` (Task 4); `ASD_Circuits.MouseSTR_AvgZ_Weighted(ExpZscoreMat, Gene2Weights)`.
- Produces: `nested_subset_recovery(expr_mat, ordered_entrez, ground_truth, sizes, bias_fn) -> pd.DataFrame` with columns `n_genes, auroc, p_mannwhitney, precision_at_20`; `leave_one_out_recovery(expr_mat, entrez_weights, ground_truth, entrez2symbol, bias_fn) -> pd.DataFrame` with columns `dropped_symbol, auroc, delta_auroc`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_disease_validation.py
from disease_validation import nested_subset_recovery, leave_one_out_recovery


@pytest.fixture
def toy_expr():
    """3 genes x 6 structures. g1 loads on A/B, g2 on C/D, g3 is flat."""
    return pd.DataFrame(
        [[3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 3.0, 3.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        index=[1, 2, 3], columns=list("ABCDEF"),
    )


def _bias_fn(expr, weights):
    from ASD_Circuits import MouseSTR_AvgZ_Weighted
    return MouseSTR_AvgZ_Weighted(expr, weights)


def test_nested_subset_returns_one_row_per_size(toy_expr):
    out = nested_subset_recovery(toy_expr, [1, 2, 3], ["A", "B"], [1, 2, 3], _bias_fn)
    assert list(out["n_genes"]) == [1, 2, 3]
    assert out.loc[out.n_genes == 1, "auroc"].iloc[0] == 1.0


def test_leave_one_out_flags_the_driver_gene(toy_expr):
    weights = {1: 1.0, 2: 1.0, 3: 1.0}
    out = leave_one_out_recovery(toy_expr, weights, ["A", "B"], {1: "G1", 2: "G2", 3: "G3"}, _bias_fn)
    assert set(out["dropped_symbol"]) == {"G1", "G2", "G3"}
    worst = out.sort_values("delta_auroc").iloc[0]
    assert worst["dropped_symbol"] == "G1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -k "nested or leave_one" -v`
Expected: FAIL — `ImportError: cannot import name 'nested_subset_recovery'`

- [ ] **Step 3: Implement**

Append to `src/disease_validation.py`:

```python
def nested_subset_recovery(expr_mat, ordered_entrez, ground_truth, sizes, bias_fn,
                           weight=1.0, effect_col="EFFECT"):
    """Recovery as a function of gene-set size, over nested prefixes.

    ordered_entrez must already be ordered by evidence tier (strongest first).
    """
    rows = []
    for n in sizes:
        subset = list(ordered_entrez)[:n]
        if not subset:
            continue
        weights = {int(g): weight for g in subset}
        bias = bias_fn(expr_mat, weights)
        try:
            s = recovery_stats(bias, ground_truth, effect_col)
        except ValueError:
            continue
        rows.append({"n_genes": n, "auroc": s["auroc"],
                     "p_mannwhitney": s["p_mannwhitney"],
                     "precision_at_20": s["precision_at_20"]})
    return pd.DataFrame(rows)


def leave_one_out_recovery(expr_mat, entrez_weights, ground_truth, entrez2symbol,
                           bias_fn, effect_col="EFFECT"):
    """Drop each gene in turn; report the change in AUROC.

    A large negative delta_auroc means that gene was carrying the result.
    """
    full = recovery_stats(bias_fn(expr_mat, dict(entrez_weights)),
                          ground_truth, effect_col)["auroc"]
    rows = []
    for g in list(entrez_weights):
        reduced = {k: v for k, v in entrez_weights.items() if k != g}
        if not reduced:
            continue
        s = recovery_stats(bias_fn(expr_mat, reduced), ground_truth, effect_col)
        rows.append({"dropped_entrez": g,
                     "dropped_symbol": entrez2symbol.get(g, str(g)),
                     "auroc": s["auroc"],
                     "delta_auroc": s["auroc"] - full})
    return pd.DataFrame(rows).sort_values("delta_auroc").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n gencic python -m pytest tests/test_disease_validation.py -v`
Expected: PASS (all tests, both files)

- [ ] **Step 5: Commit**

```bash
git add src/disease_validation.py tests/test_disease_validation.py
git commit -m "Add gene-set size sensitivity and leave-one-out recovery"
```

---

### Task 6: Notebook 01 — gene curation and weight files

**Files:**
- Create: `notebooks_disease_validation/01.Gene_Curation.py` (+ synced `.ipynb`)
- Creates as output: `dat/Genetics/GeneWeights/{PD_Primary,PD_Sens_DA,PD_Sens_Atypical,PD_GWAS_L2G,HD_HTT,StriatalDegeneration}.gw`, matching `.DN.gw` in `dat/Genetics/GeneWeights_DN/`, `dat/Disease_Validation/pd_l2g_opentargets.json`, `results/tables/disease_validation_gene_report.csv`
- Reads (does not modify): `config/disease_validation_genesets.yaml`

**Interfaces:**
- Consumes: `load_gene_sets`, `gene_set_weights`, `gene_set_report` (Task 1); `ASD_Circuits.Dict2Fil`, `LoadGeneINFO`.
- Produces: six `.gw` files and six `.DN.gw` files consumed by Tasks 7 and 11.

- [ ] **Step 1: Create the notebook skeleton with sections**

Sections: `# 1. Setup` · `# 2. Load pre-registered gene sets` · `# 3. Open Targets L2G pull for the GWAS tier` · `# 4. Resolve to Entrez and report coverage` · `# 5. Write .gw weight files` · `# 6. Write DN weight files (cell-type arm only)` · `# 7. Verify against the pre-registration`

- [ ] **Step 2: Write the Open Targets pull cell**

```python
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
if os.path.exists(CACHE):
    l2g = json.load(open(CACHE))
    print(f"Loaded cached Open Targets pull: {len(l2g)} genes")
else:
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

derived = sorted(
    g for g, v in l2g.items() if v["l2g"] >= L2G_THRESHOLD and g not in L2G_ARTIFACTS)

# AUDIT, do not define. Membership is frozen in the pre-registered YAML; this
# cell re-derives it and asserts agreement. Open Targets is a live API, so a
# silent upstream change must fail loudly rather than quietly alter the gene set.
frozen = sorted(r["symbol"] for r in
                load_gene_sets("../config/disease_validation_genesets.yaml")["PD_GWAS_L2G"])
if derived != frozen:
    print(f"WARNING: Open Targets now yields {len(derived)} genes, "
          f"pre-registration has {len(frozen)}")
    print("  added upstream:", sorted(set(derived) - set(frozen)))
    print("  dropped upstream:", sorted(set(frozen) - set(derived)))
    print("  Proceeding with the FROZEN list. Report this drift in the methods.")
else:
    print(f"PD_GWAS_L2G: {len(frozen)} genes, re-derivation matches pre-registration.")
pd_gwas_symbols = frozen
```

- [ ] **Step 3: Write the `.gw` generation cell**

```python
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
```

- [ ] **Step 4: Write the DN weight cell**

```python
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
```

- [ ] **Step 5: Write the verification cell**

```python
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
```

- [ ] **Step 6: Sync and execute end to end**

```bash
conda run -n gencic jupytext --sync notebooks_disease_validation/01.Gene_Curation.py
conda run -n gencic jupyter nbconvert --to notebook --execute --inplace \
  notebooks_disease_validation/01.Gene_Curation.ipynb
```
Expected: completes with no exception; the Section 7 assertion prints "All gene sets match the pre-registration."

- [ ] **Step 7: Commit**

```bash
git add notebooks_disease_validation/01.Gene_Curation.py \
        notebooks_disease_validation/01.Gene_Curation.ipynb
git commit -m "Add notebook 01: curate PD/striatal gene sets and weight files"
```

---

### Task 7: Register gene sets and run the structure-level bias pipeline

**Files:**
- Modify: `config/config.STR.yaml`
- Modify: `config/config.SC.DN.yaml`

**Interfaces:**
- Consumes: the six `.gw` files from Task 6.
- Produces: `results/STR_ISH/{set}_bias_addP_random.csv` and `results/STR_ISH/{set}_EM_bias_addP_random.csv` for all six sets.

- [ ] **Step 1: Append the 12 STR entries**

Each set is registered twice so both nulls can coexist — output filenames are keyed on the gene-set name, so a suffixed duplicate is how a second null is obtained without changing the Snakefile's output contract.

```yaml
  # --- Disease validation (reviewer 2). Uniform random null. ---
  PD_Primary:
    geneweights: "dat/Genetics/GeneWeights/PD_Primary.gw"
    null_model: "uniform"
  PD_Sens_DA:
    geneweights: "dat/Genetics/GeneWeights/PD_Sens_DA.gw"
    null_model: "uniform"
  PD_Sens_Atypical:
    geneweights: "dat/Genetics/GeneWeights/PD_Sens_Atypical.gw"
    null_model: "uniform"
  PD_GWAS_L2G:
    geneweights: "dat/Genetics/GeneWeights/PD_GWAS_L2G.gw"
    null_model: "uniform"
  HD_HTT:
    geneweights: "dat/Genetics/GeneWeights/HD_HTT.gw"
    null_model: "uniform"
  StriatalDegeneration:
    geneweights: "dat/Genetics/GeneWeights/StriatalDegeneration.gw"
    null_model: "uniform"
  # --- Same sets, expression-decile-matched null. ---
  PD_Primary_EM:
    geneweights: "dat/Genetics/GeneWeights/PD_Primary.gw"
    null_model: "expmatched"
  PD_Sens_DA_EM:
    geneweights: "dat/Genetics/GeneWeights/PD_Sens_DA.gw"
    null_model: "expmatched"
  PD_Sens_Atypical_EM:
    geneweights: "dat/Genetics/GeneWeights/PD_Sens_Atypical.gw"
    null_model: "expmatched"
  PD_GWAS_L2G_EM:
    geneweights: "dat/Genetics/GeneWeights/PD_GWAS_L2G.gw"
    null_model: "expmatched"
  HD_HTT_EM:
    geneweights: "dat/Genetics/GeneWeights/HD_HTT.gw"
    null_model: "expmatched"
  StriatalDegeneration_EM:
    geneweights: "dat/Genetics/GeneWeights/StriatalDegeneration.gw"
    null_model: "expmatched"
```

- [ ] **Step 2: Append the six DN entries to `config/config.SC.DN.yaml`**

Same block, but pointing at `dat/Genetics/GeneWeights_DN/{name}.DN.gw`, `null_model: "uniform"`, no `_EM` duplicates (the cell-type arm uses the uniform null only).

- [ ] **Step 3: Dry-run to confirm the DAG resolves**

```bash
conda run -n gencic snakemake -s Snakefile.bias --configfile config/config.STR.yaml -n \
  $(for s in PD_Primary PD_Sens_DA PD_Sens_Atypical PD_GWAS_L2G HD_HTT StriatalDegeneration \
             PD_Primary_EM PD_Sens_DA_EM PD_Sens_Atypical_EM PD_GWAS_L2G_EM HD_HTT_EM StriatalDegeneration_EM; do
      echo results/STR_ISH/${s}_bias_addP_random.csv; done)
```
Expected: a job list covering 12 gene sets, no `MissingInputException`.

- [ ] **Step 4: Run the pipeline**

Same command without `-n`, adding `--cores 10`.
Expected: 12 `*_bias_addP_random.csv` files in `results/STR_ISH/`.

- [ ] **Step 5: Verify output shape and sanity**

```bash
conda run -n gencic python -c "
import pandas as pd
for s in ['PD_Primary','HD_HTT','StriatalDegeneration']:
    d = pd.read_csv(f'results/STR_ISH/{s}_bias_addP_random.csv', index_col=0)
    assert d.shape[0] == 213, (s, d.shape)
    assert {'EFFECT','Rank','P-value','q-value'} <= set(d.columns), d.columns.tolist()
    print(s, 'top5:', d.sort_values('EFFECT', ascending=False).index[:5].tolist())
"
```
Expected: 213 rows each; `PD_Primary` top structures should include basal-ganglia or midbrain names. **If they do not, stop and report — that is the spec §10 failure condition, not a bug to debug around.**

- [ ] **Step 6: Commit**

```bash
git add config/config.STR.yaml config/config.SC.DN.yaml
git commit -m "Register disease-validation gene sets for both nulls"
```

---

### Task 8: Notebook 02 part A — recovery, sensitivity, negative controls

**Files:**
- Create: `notebooks_disease_validation/02.STR_Bias_and_Circuits.py` (+ `.ipynb`)
- Modify: `src/plot.py` (add `plot_recovery_forest`, `plot_nested_subset_curve`)

**Interfaces:**
- Consumes: `recovery_stats`, `recovery_permutation_p`, `nested_subset_recovery`, `leave_one_out_recovery`, `load_ground_truth`.
- Produces: `results/tables/disease_validation_recovery.csv` with columns
  `gene_set, ground_truth, n_ground_truth, n_missing, u_stat, p_mannwhitney, auroc,
  precision_at_20, median_rank, p_structure_permutation, null, observed_auroc,
  null_auroc_median, null_auroc_p95, p_geneset_null`.
  One row per (gene set x ground-truth group x null); the `auroc`/`p_mannwhitney`
  columns repeat across the two null rows by construction, since EFFECT is
  null-independent. Also `disease_validation_nested_subsets.csv`,
  `disease_validation_leave_one_out.csv`, `disease_validation_negative_controls.csv`.
  Task 10's figures target these exact names.

- [ ] **Step 1: Create sections**

`# 1. Setup and pre-registration hash` · `# 2. Load bias results` · `# 3. Recovery against pre-registered ground truth` · `# 4. Both nulls side by side` · `# 5. Gene-set size sensitivity` · `# 6. Leave-one-out` · `# 7. Negative-control cross-test`

- [ ] **Step 2: Write the recovery table cell**

```python
# %%
# Section 3: observed recovery. EFFECT does not depend on the null model, so
# this table has ONE row per (gene set, ground-truth group) - not one per null.
GT = load_ground_truth("../config/disease_validation_ground_truth.yaml")
GT_MAP = GT["notes"]["gene_sets_to_ground_truth"]
SETS = ["PD_Primary", "PD_Sens_DA", "PD_Sens_Atypical", "PD_GWAS_L2G",
        "HD_HTT", "StriatalDegeneration"]

rows = []
for s in SETS:
    disease = GT_MAP[s]
    bias = pd.read_csv(f"../results/STR_ISH/{s}_bias_addP_random.csv", index_col=0)
    for group, structs in GT["structures"][disease].items():
        st = recovery_stats(bias, structs)
        st.update(gene_set=s, ground_truth=f"{disease}/{group}",
                  p_structure_permutation=recovery_permutation_p(
                      bias, structs, n_perm=10000, seed=42))
        rows.append(st)
observed = pd.DataFrame(rows).drop(columns=["missing"])

# %%
# Section 4: the null-model comparison, done correctly.
# For each null we take the 10,000 null GENE SET bias profiles and ask how often
# a null gene set recovers the ground truth as well as the real one does.
# Sanity check first - proves why Section 3 has no per-null rows.
_u = pd.read_csv("../results/STR_ISH/PD_Primary_bias_addP_random.csv", index_col=0)
_e = pd.read_csv("../results/STR_ISH/PD_Primary_EM_bias_addP_random.csv", index_col=0)
assert np.allclose(_u["EFFECT"], _e.loc[_u.index, "EFFECT"]), \
    "EFFECT must be null-independent; if this fails the pipeline changed"
print("Confirmed: EFFECT is identical across nulls; only the null distribution differs.")

null_rows = []
for s in SETS:
    disease = GT_MAP[s]
    for null, geneset_key in [("uniform", s), ("expression_matched", f"{s}_EM")]:
        null_bias = pd.read_parquet(
            f"../results/STR_ISH/null_bias/{geneset_key}_null_bias_random.parquet")
        for group, structs in GT["structures"][disease].items():
            obs = observed.loc[(observed.gene_set == s)
                               & (observed.ground_truth == f"{disease}/{group}"),
                               "auroc"].iloc[0]
            nulls = recovery_null_aurocs(null_bias, structs)
            null_rows.append({
                "gene_set": s, "null": null, "ground_truth": f"{disease}/{group}",
                "observed_auroc": obs,
                "null_auroc_median": float(np.median(nulls)),
                "null_auroc_p95": float(np.percentile(nulls, 95)),
                "p_geneset_null": empirical_p(obs, nulls),
            })

recovery = observed.merge(pd.DataFrame(null_rows), on=["gene_set", "ground_truth"])
recovery.to_csv("../results/tables/disease_validation_recovery.csv", index=False)
recovery[recovery.ground_truth.str.endswith("core")]
```

- [ ] **Step 3: Write the sensitivity cells**

```python
# %%
# Section 5-6: is one gene carrying the result? Run for EVERY non-singleton set,
# not just PD_Primary - the striatal panel has the same driver-gene risk, and
# HTT-only cannot be leave-one-out'd.
from ASD_Circuits import MouseSTR_AvgZ_Weighted
bias_fn = lambda expr, w: MouseSTR_AvgZ_Weighted(expr, w)

curation = load_gene_sets("../config/disease_validation_genesets.yaml")
nested_all, loo_all = [], []
for s in SETS:
    weights = Fil2Dict(f"../dat/Genetics/GeneWeights/{s}.gw")
    if len(weights) < 2:
        print(f"{s}: {len(weights)} gene - sensitivity analysis not applicable")
        continue
    gt = GT["structures"][GT_MAP[s]]["core"]
    # YAML order is curated strongest-evidence-first.
    ordered = [int(GeneSymbol2Entrez[lookup_symbol(r)]) for r in curation[s]
               if GeneSymbol2Entrez.get(lookup_symbol(r)) is not None
               and int(GeneSymbol2Entrez[lookup_symbol(r)]) in weights]
    sizes = [n for n in (5, 10, 15, 20, 25, 30, 41) if n <= len(ordered)]
    n = nested_subset_recovery(Z2, ordered, gt, sizes, bias_fn); n.insert(0, "gene_set", s)
    l = leave_one_out_recovery(Z2, weights, gt, Entrez2Symbol, bias_fn); l.insert(0, "gene_set", s)
    nested_all.append(n); loo_all.append(l)

nested = pd.concat(nested_all, ignore_index=True)
loo = pd.concat(loo_all, ignore_index=True)
nested.to_csv("../results/tables/disease_validation_nested_subsets.csv", index=False)
loo.to_csv("../results/tables/disease_validation_leave_one_out.csv", index=False)
print("Largest single-gene dependencies (most negative delta_auroc):")
print(loo.sort_values("delta_auroc").groupby("gene_set").head(2))
```

- [ ] **Step 4: Write the negative-control cross-test cell**

```python
# %%
# Section 7: do non-brain trait gene sets recover the PD circuit? They must not.
NEG = ["T2D", "IBD", "HDL_C", "hba1c"]
neg_rows = []
for s in NEG:
    bias = pd.read_csv(f"../results/STR_ISH/{s}_bias_addP_random.csv", index_col=0)
    for disease in ("parkinson", "striatal"):
        st = recovery_stats(bias, GT["structures"][disease]["core"])
        st.update(gene_set=s, ground_truth=f"{disease}/core")
        neg_rows.append(st)
neg = pd.DataFrame(neg_rows).drop(columns=["missing"])
neg.to_csv("../results/tables/disease_validation_negative_controls.csv", index=False)
neg[["gene_set", "ground_truth", "auroc", "p_mannwhitney"]]
```

- [ ] **Step 5: Execute and verify**

```bash
conda run -n gencic jupytext --sync notebooks_disease_validation/02.STR_Bias_and_Circuits.py
conda run -n gencic jupyter nbconvert --to notebook --execute --inplace \
  notebooks_disease_validation/02.STR_Bias_and_Circuits.ipynb
```
Expected: `results/tables/disease_validation_recovery.csv` exists with 6 sets × 2 nulls × 2 ground-truth groups = 24 rows.

- [ ] **Step 6: Commit**

```bash
git add notebooks_disease_validation/02.STR_Bias_and_Circuits.py \
        notebooks_disease_validation/02.STR_Bias_and_Circuits.ipynb src/plot.py
git commit -m "Add recovery, sensitivity and negative-control analyses"
```

---

### Task 9: SA circuit search

**Files:**
- Create: `config/circuit_config_disease.yaml`

**Interfaces:**
- Consumes: `results/STR_ISH/{set}_bias_addP_random.csv` from Task 7.
- Produces: `results/CircuitSearch/{set}/best_circuits/size_{n}_best_circuits.txt`.

- [ ] **Step 1: Derive per-disease circuit sizes**

```python
# run in the notebook; record the chosen sizes in the config
for s in SETS:
    bias = pd.read_csv(f"../results/STR_ISH/{s}_bias_addP_random.csv", index_col=0)
    n10 = int((bias["q-value"] < 0.10).sum())
    ccs = np.array([ScoreCircuit_SI_Joint(
        bias.sort_values("EFFECT", ascending=False).index.values[:n], IpsiInfoMat)
        for n in range(200, 5, -1)])
    peak = int(np.arange(200, 5, -1)[np.argmax(ccs)])
    print(f"{s:22s} FDR<0.10: {n10:3d}   CCS peak: {peak:3d}")
```
Sizes per set = `sorted({n10, peak, peak-5, peak+5})` filtered to `10 <= n <= 120`, capped at 4.

- [ ] **Step 2: Write `config/circuit_config_disease.yaml`**

Copy `config/circuit_config.yaml` verbatim, replacing `Input_str_bias` with one entry per gene set pointing at `results/STR_ISH/{set}_bias_addP_random.csv`, and `circuit_sizes` with the sizes derived in Step 1. Keep `sa_runtimes: 100`, `sa_steps: 100000`, `measure: "SI"`, `min_bias_rank: 50`, `output_dir: "results/CircuitSearch"`.

- [ ] **Step 3: Benchmark ONE size before the full sweep**

```bash
free -h && nproc && uptime      # pre-flight, per the resource policy
time conda run -n gencic snakemake -s Snakefile.circuit \
  --configfile config/circuit_config_disease.yaml --cores 20 \
  results/CircuitSearch/PD_Primary/best_circuits/size_46_best_circuits.txt
```
Multiply the observed wall time by (number of sets × sizes per set) and **report the estimate before proceeding**. If it exceeds ~12 hours, cut to 2 sizes per set and state the reduction explicitly in the notebook — spec §7 forbids silent truncation.

- [ ] **Step 4: Run the full sweep**

```bash
conda run -n gencic snakemake -s Snakefile.circuit \
  --configfile config/circuit_config_disease.yaml --cores 20
```

- [ ] **Step 5: Verify circuits were produced**

```bash
for s in PD_Primary PD_Sens_DA PD_Sens_Atypical PD_GWAS_L2G HD_HTT StriatalDegeneration; do
  echo -n "$s: "; ls results/CircuitSearch/$s/best_circuits/ 2>/dev/null | wc -l
done
```
Expected: a nonzero count for each.

- [ ] **Step 6: Commit**

```bash
git add config/circuit_config_disease.yaml
git commit -m "Add SA circuit search config for disease validation sets"
```

---

### Task 10: Notebook 02 part B — CCS profiles and circuit figures

**Files:**
- Modify: `notebooks_disease_validation/02.STR_Bias_and_Circuits.py`
- Modify: `src/plot.py` (add `plot_circuit_vs_anatomy`)

**Interfaces:**
- Consumes: `results/CircuitSearch/{set}/best_circuits/*.txt`; `ScoreCircuit_SI_Joint`; `dat/allen-mouse-conn/RankScores/RankScore.Ipsi.Cont.npy`.
- Produces: figures in `results/figures/`.

- [ ] **Step 1: Add sections 8–10**

`# 8. CCS profiles vs null` · `# 9. Recovered circuits vs known anatomy` · `# 10. Summary figure`

- [ ] **Step 2: Write the CCS profile cell**

Reuse the existing pattern from `notebooks_mouse_str/10.Positive_Control_Circuits.py:138-161`: plot each set's CCS across `topNs = np.arange(200, 5, -1)` against the `Cont_Distance` null band with `BarLen = 34.1`. Colors from `src/plot.REGION_COLORS` where a region mapping applies, otherwise an explicit dict — never a bare `c=`.

- [ ] **Step 3: Write the circuit-vs-anatomy cell**

For each gene set, load `size_{peak}_best_circuits.txt`, annotate each structure with `STR2Region()`, and report the overlap with the pre-registered ground truth (count, hypergeometric p). Plot as a region-colored bar with ground-truth members marked.

- [ ] **Step 4: Execute and verify figures exist**

```bash
conda run -n gencic jupytext --sync notebooks_disease_validation/02.STR_Bias_and_Circuits.py
conda run -n gencic jupyter nbconvert --to notebook --execute --inplace \
  notebooks_disease_validation/02.STR_Bias_and_Circuits.ipynb
ls results/figures/ | grep -i "disease\|pd_\|striatal"
```

- [ ] **Step 5: Commit**

```bash
git add notebooks_disease_validation/02.STR_Bias_and_Circuits.py \
        notebooks_disease_validation/02.STR_Bias_and_Circuits.ipynb src/plot.py
git commit -m "Add CCS profiles and circuit-vs-anatomy figures"
```

---

### Task 11: Notebook 03 — cell-type arm

**Files:**
- Create: `notebooks_disease_validation/03.CellType_and_MERFISH.py` (+ `.ipynb`)

**Interfaces:**
- Consumes: `dat/Genetics/GeneWeights_DN/{set}.DN.gw` (Task 6); `dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet`; `dat/MouseCT_Cluster_Anno.csv`; `recovery_stats`.
- Produces: `results/tables/disease_validation_celltype_recovery.csv`.

- [ ] **Step 1: Run the CT bias pipeline**

```bash
conda run -n gencic snakemake -s Snakefile.bias --configfile config/config.SC.DN.yaml --cores 10 \
  $(for s in PD_Primary PD_Sens_DA PD_Sens_Atypical PD_GWAS_L2G HD_HTT StriatalDegeneration; do
      echo results/CT_Z2/${s}_bias_addP_random.csv; done)
```
Expected: six files, 5,312 rows each.

- [ ] **Step 2: Write the cluster→subclass mapping cell**

```python
# %%
# Cluster IDs look like '0943 STR D1 Gaba_1'; the subclass is the text between
# the leading number and the trailing _N suffix.
import re
def cluster_to_subclass(cluster_id):
    return re.sub(r'^\d+\s+', '', cluster_id).rsplit('_', 1)[0]

GT = load_ground_truth("../config/disease_validation_ground_truth.yaml")
def clusters_for(subclasses, index):
    want = set(subclasses)
    return [c for c in index if cluster_to_subclass(c) in want]
```

- [ ] **Step 3: Write the cell-type recovery cell**

```python
# %%
rows = []
for s in SETS:
    disease = GT["notes"]["gene_sets_to_ground_truth"][s]
    bias = pd.read_csv(f"../results/CT_Z2/{s}_bias_addP_random.csv", index_col=0)
    targets = clusters_for(GT["cell_type_subclasses"][disease]["core"], bias.index)
    st = recovery_stats(bias, targets)
    st.update(gene_set=s, ground_truth=disease, n_clusters=len(targets),
              p_permutation=recovery_permutation_p(bias, targets, n_perm=10000, seed=42))
    rows.append(st)
ct = pd.DataFrame(rows).drop(columns=["missing"])
ct.to_csv("../results/tables/disease_validation_celltype_recovery.csv", index=False)
ct[["gene_set", "ground_truth", "n_clusters", "auroc", "p_mannwhitney", "p_permutation"]]
```

- [ ] **Step 4: Add the subclass boxplot**

Boxplot of cluster `EFFECT` grouped by subclass, highlighting `SNc-VTA-RAmb Foxa1 Dopa` for PD sets and `STR D1/D2 Gaba` for striatal sets, with the basal-ganglia context subclasses shown in a neutral color.

- [ ] **Step 5: Execute and verify**

```bash
conda run -n gencic jupytext --sync notebooks_disease_validation/03.CellType_and_MERFISH.py
conda run -n gencic jupyter nbconvert --to notebook --execute --inplace \
  notebooks_disease_validation/03.CellType_and_MERFISH.ipynb
```
Expected: `results/tables/disease_validation_celltype_recovery.csv` with 6 rows.

- [ ] **Step 6: Commit**

```bash
git add notebooks_disease_validation/03.CellType_and_MERFISH.py \
        notebooks_disease_validation/03.CellType_and_MERFISH.ipynb
git commit -m "Add cell-type recovery analysis for disease validation"
```

---

### Task 12: MERFISH concordance arm

**Files:**
- Modify: `notebooks_disease_validation/03.CellType_and_MERFISH.py`

**Interfaces:**
- Consumes: the four MERFISH Z2 matrices referenced in `notebooks_mouse_sc/04.MERFISH_Structure_Bias.py`; `MouseSTR_AvgZ_Weighted`.
- Produces: `results/tables/disease_validation_merfish_concordance.csv`.

- [ ] **Step 1: Compute MERFISH structure bias**

For each of the four MERFISH Z2 matrices and each gene set, compute structure bias with the **non-DN** weights (MERFISH is a structure-level analysis). Follow the loading pattern in `notebooks_mouse_sc/04.MERFISH_Structure_Bias.py`.

- [ ] **Step 2: Report ISH↔MERFISH concordance**

Spearman correlation of `EFFECT` across shared structures between the ISH bias and each MERFISH bias, per gene set. Write to `results/tables/disease_validation_merfish_concordance.csv` with columns `gene_set, merfish_matrix, n_shared_structures, spearman_r, p`.

- [ ] **Step 3: Rerun the recovery test on MERFISH bias**

Apply `recovery_stats` to the MERFISH-derived bias with the same pre-registered ground truth, so the reviewer sees an independent-modality replication rather than only a correlation.

- [ ] **Step 4: Execute and verify**

```bash
conda run -n gencic jupytext --sync notebooks_disease_validation/03.CellType_and_MERFISH.py
conda run -n gencic jupyter nbconvert --to notebook --execute --inplace \
  notebooks_disease_validation/03.CellType_and_MERFISH.ipynb
```

- [ ] **Step 5: Commit**

```bash
git add notebooks_disease_validation/03.CellType_and_MERFISH.py \
        notebooks_disease_validation/03.CellType_and_MERFISH.ipynb
git commit -m "Add MERFISH concordance and independent-modality recovery"
```

---

### Task 13: Deprecate legacy files and document

**Files:**
- Modify: `notebooks_mouse_str/10.Positive_Control_Circuits.py:177-184`
- Modify: `DATA_MANIFEST.yaml`
- Create: `docs/DEPRECATED_GENE_WEIGHTS.md`  (docs/ is tracked; `dat/` is gitignored so a note there could never be committed)

- [ ] **Step 1: Remove the conflicting cell**

Delete the cell at `notebooks_mouse_str/10.Positive_Control_Circuits.py:177-184` that rewrites `Parkinson.gw` from the hardcoded 5-gene `PKList`. It silently overwrites a curated file on every run (spec §11 risk 5).

The same notebook also **force-runs Snakemake on the legacy sets** at lines 187-208: `neg_ctrl_sets` includes `"Parkinson"` and `"Alzheimer"`, and both bias files are loaded. Remove those two entries from `neg_ctrl_sets` and drop the corresponding `pd.read_csv` lines — they are already commented out of `disorder_datasets` (lines 215-223), so nothing downstream consumes them. Leave `T2D`, `hba1c`, `IBD`, `HDL_C` untouched; Task 8 depends on those files.

Replace the deleted gene-weight cell with a markdown cell:

```markdown
# %% [markdown]
# Parkinson's analysis moved to `notebooks_disease_validation/`.
# The former 5-gene `Parkinson.gw` and the nearest-gene `Parkinson.top61.gw`
# are deprecated - see `docs/DEPRECATED_GENE_WEIGHTS.md`.
```

- [ ] **Step 2: Write the deprecation note**

```markdown
# Deprecated gene weight files

Do not use these. Superseded by `notebooks_disease_validation/01.Gene_Curation`.

| File | Problem |
|---|---|
| `Parkinson.top61.gw` | Nearest-gene GWAS assignment. Contains COMMD9, GOLGA6L2, KRT76, CYP21A2, IFNL3, ABCB11. Only GBA and MAPT are genuine PD genes. |
| `Parkinson.gw` | Ad-hoc 5-gene list, rewritten on every run of the old notebook 10. |
| `ALZ.top60.gw` | Nearest-gene GWAS assignment. Contains no APP, PSEN1, PSEN2 or APOE. |

Replacements: `PD_Primary.gw`, `PD_Sens_DA.gw`, `PD_Sens_Atypical.gw`,
`PD_GWAS_L2G.gw` (Open Targets locus-to-gene, not nearest-gene).
```

- [ ] **Step 3: Audit for downstream use of the deprecated files**

```bash
grep -rn "Parkinson.top61\|Parkinson\.gw\|ALZ.top60" --include="*.py" --include="*.ipynb" --include="*.yaml" . \
  | grep -v ".ipynb_checkpoints" | grep -v DEPRECATED
```
Any hit in a figure- or table-producing notebook must be reported — a published figure may be built on the bad list.

- [ ] **Step 4: Add `DATA_MANIFEST.yaml` entries**

One entry per new data file (six `.gw`, six `.DN.gw`, two `config/disease_validation_*.yaml`, `pd_l2g_opentargets.json`, the three `results/tables/*.csv`) with `path`, `description`, `format`, `key_fields`, `source`, `size_approx`, `notes`.

- [ ] **Step 5: Run the full test suite**

```bash
conda run -n gencic python -m pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add notebooks_mouse_str/10.Positive_Control_Circuits.py \
        notebooks_mouse_str/10.Positive_Control_Circuits.ipynb \
        docs/DEPRECATED_GENE_WEIGHTS.md DATA_MANIFEST.yaml
git commit -m "Deprecate legacy Parkinson/Alzheimer gene weights; document new data"
```

---

## Self-Review

**Spec coverage.** §2 framing → Tasks 1, 6. §3 gene sets → Tasks 1, 6. §4 pre-registration → Task 1 (committed before Task 8 reads it). §5.1 uniform null → Task 7. §5.2 expression-matched null → Tasks 2, 3, 7. §6 recovery metrics → Task 4, notebook cells in Task 8. §6.1 size sensitivity → Tasks 5, 8. §6.2 negative controls → Task 8. §6.3 CCS → Task 10. §7 circuit search incl. the benchmark-before-sweep requirement → Task 9. §8 cell-type + MERFISH → Tasks 11, 12. §9 deliverables → all. §10 interpretation → Task 7 step 5 halts on the failure condition rather than debugging around it. §11 risks: risk 3 (Open Targets coverage) is documented in Task 6's cell comment; risk 5 (deprecated files) → Task 13.

**Type consistency.** `recovery_stats` returns the same key set everywhere it is called (Tasks 8, 11, 12) and `drop(columns=["missing"])` is applied consistently before writing any CSV. `bias_fn(expr, weights)` has one signature across Tasks 5 and 8. `gene_set_weights(records, symbol2entrez, valid_genes)` matches its call sites in Tasks 1 and 6. `sample_expression_matched(target, decile_map, n_sims, rng)` matches its caller in Task 3.

**Statistical design, stated explicitly.** Three distinct significance tests are reported and must not be conflated:

| Test | What is permuted | What it answers | Differs by null model? |
|---|---|---|---|
| `p_mannwhitney` | nothing (analytic) | Do ground-truth structures rank above the rest in this one profile? | No |
| `p_structure_permutation` | structure labels | Is this *set* of structures unusually high in this ranking? | No |
| `p_geneset_null` | the gene set | Does the real gene set recover the circuit better than null gene sets? | **Yes** — this is the only one that does |

`EFFECT` is computed from the real gene set and is byte-identical regardless of which null was configured (verified: `max|ΔEFFECT| = 0.0` between the ASD uniform and sibling bias files). Any table comparing "recovery under null A vs null B" on `EFFECT` compares nothing. The uniform-vs-expression-matched contrast lives entirely in `p_geneset_null`, computed from the `null_bias/*.parquet` simulation matrices.

**Fixes applied after Codex review (round 2).** Tie-safe `rankdata` in `recovery_null_aurocs` with a regression test (double-argsort was wrong for tied bias values); exact `PD_GWAS_L2G` matrix coverage (40 of 41, `FAM47E` dropped) asserted with the exact dropped set rather than a floor; Task 8's declared output schema aligned with the columns actually written; conditional `expmatch` input plus an explicit rerun-trigger hazard note so published nulls cannot be silently regenerated; added InfoMat and cell-type-subclass data-contract tests.

**Fixes applied after Codex review (round 1).** `GBA1`→`GBA` alias override with a regression test; `PD_GWAS_L2G` frozen in the Task 1 pre-registration and merely audited by notebook 01; all three null samplers seeded; the invalid dual-null recovery table replaced with the gene-set null above; toy Mann-Whitney test corrected to 3-over-5 (2-over-4 has a minimum achievable one-sided p of 0.0667); sensitivity analyses generalized to every non-singleton set; negative-control table now persisted; config paths made relative with a `PROJDIR` join that is a no-op for the existing absolute entries; `rule all` hazard documented; deprecation note moved from gitignored `dat/` to tracked `docs/`; legacy Parkinson/Alzheimer reads removed from notebook 10.
