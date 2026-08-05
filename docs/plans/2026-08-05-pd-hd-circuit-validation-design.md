# Design: Validating GENCIC on Parkinson's Disease and Genetic Striatal Degeneration

**Date:** 2026-08-05
**Status:** Design approved, pending implementation plan
**Motivation:** Reviewer 2, comment 2 (resubmission)

---

## 1. Problem Statement

The reviewer wrote:

> The CCS/GENCIC framework is the primary methodological contribution of this work, so I
> would like to see stronger validation of its ability to recover biologically meaningful
> disease-associated circuits. The only validation presented in the main text is the dopamine
> circuit, with two additional neurotransmitter analyses in the supplement. These examples are
> reassuring sanity checks but do not fully validate the broader claim that disease-associated
> gene sets can recover disease-relevant neural circuits. Neurotransmitter systems are much
> more evolutionarily conserved than complex neurodevelopmental disorders. I would find it
> more convincing to validate the framework using a disease with both well-established causal
> genes and a well-characterized affected circuit, such as Parkinson's disease and/or
> Huntington's disease.

A prior attempt at Parkinson's produced no meaningful result. **The diagnosis is that the gene
set was wrong, not the framework.** `dat/Genetics/GeneWeights/Parkinson.top61.gw` is a
nearest-gene GWAS list containing `COMMD9`, `GOLGA6L2`, `KRT76`, `CYP21A2`, `IFNL3`, `ABCB11`;
only `GBA` and `MAPT` are genuine PD genes. The companion `ALZ.top60.gw` contains no `APP`,
`PSEN1`, `PSEN2`, or `APOE`. Both lists should be considered deprecated.

Individual-gene Z2 profiles confirm the signal is present in the data once the gene set is
correct:

| Gene | Top-ranked structures (Z2) |
|---|---|
| `SLC6A3` | Ventral tegmental area (5.88), SN compacta (5.86), SN reticulata (5.65) |
| `TH` | Ventral tegmental area (3.94), Midbrain reticular n. (3.39), SN compacta (2.68) |
| `LRRK2` | Caudoputamen (4.93), Fundus of striatum (4.52), Olfactory tubercle (4.52) |
| `HTT` | Lateral dorsal thalamus (2.35), Reticular thalamic n. (2.24) — **not striatum** |

## 2. Scope and Framing

Two validations, deliberately kept distinct:

1. **Parkinson's disease** — the primary validation. Polygenic Mendelian architecture with a
   well-characterized nigrostriatal circuit. This is where the framework should succeed.
2. **Genetic striatal degeneration** — a secondary validation on a panel of Mendelian
   syndromes causing striatal degeneration.

**Huntington's disease is tested separately as `HTT` alone, and is expected to fail.** This is
a deliberate scope statement, not an oversight. GENCIC aggregates expression bias across a gene
set; a monogenic repeat-expansion disorder whose causal gene is ubiquitously expressed carries
no aggregate signal to detect. Broadening "Huntington's" into a chorea gene panel and reporting
the result as an HD validation would be a category error — it would test whether a
hand-assembled collection of phenocopy genes maps to the basal ganglia, not whether
HTT-associated genetics maps to the striatum. The rebuttal will state this directly: PD
validates the framework, HD delineates its boundary.

This framing was adopted after an independent domain consultation (Codex, session
`019fd18f-2948-7602-94e2-4a2e099a6511`, 2026-08-05), which raised the objection above.

### 2.1 Inclusion criterion

A gene enters a disease set only if the syndrome it causes involves **anatomical degeneration**
of the target circuit, not merely functional dysfunction of it. This excludes genes causing
chorea without cell loss (`NKX2-1` benign hereditary chorea, `ADCY5`-related dyskinesia,
`GPR88`, `FRRS1L`) and biochemical dopamine-deficiency syndromes without nigral degeneration
(`TH`, `DDC`, `GCH1`, `SPR`, `SLC6A3` — these move to a sensitivity tier). The criterion is
stated because GENCIC predicts anatomical vulnerability; a gene causing dysfunction without
degeneration tests a different claim.

### 2.2 Circularity control

Genes whose mouse expression *defines* the target structure are excluded from every primary
tier: `TH`, `SLC6A3`, `DDC`, `GCH1`, `SPR` (dopaminergic-neuron markers) and `PDE10A`,
`GPR88`, `ADCY5`, `RASD2`, `DRD2` (striatal MSN markers). Including them would make recovery of
SNc or striatum near-mechanical. They are instead added back in a sensitivity tier, so the
direction of the test is "does adding markers strengthen an already-positive result", never
"does removing them preserve it".

## 3. Gene Sets

All sets use **uniform weight 1.0**. No de-novo mutation counts exist for these disorders, and
uniform weighting matches how the published dopamine/serotonin/oxytocin positive controls and
the T2D/IBD/HDL_C negative controls were run, keeping the comparison apples-to-apples.

Six sets. Symbols resolve to Entrez via `LoadGeneINFO()`; genes absent from the Z2 matrix are
reported explicitly in notebook 01 and retained in the source CSV with a `in_matrix` flag.

### 3.1 `PD_Primary` (15 genes)

Mendelian PD with nigrostriatal dopaminergic degeneration. No dopamine-marker genes.

`SNCA` `LRRK2` `VPS35` `GBA1` `CHCHD2` `RAB39B` `PRKN` `PINK1` `PARK7` `ATP13A2` `PLA2G6`
`FBXO7` `DNAJC6` `SYNJ1` `VPS13C`

### 3.2 `PD_Sens_DA` (20 genes)

`PD_Primary` + `TH` `SLC6A3` `DDC` `GCH1` `SPR`. Tests whether adding dopamine-synthesis and
-transport genes strengthens nigrostriatal recovery. Any result here is interpreted only
relative to `PD_Primary`.

### 3.3 `PD_Sens_Atypical` (24 genes)

`PD_Primary` + `DNAJC13` `LRP10` `DCTN1` `MAPT` `POLG` `TWNK` `SPG11` `PTRHD1` `DNAJC12`.
Disputed PD candidates and parkinsonism-plus syndromes. Tests robustness to curation
permissiveness in the direction of clinical heterogeneity.

**Explicitly excluded** as refuted or downgraded: `TMEM230` `UCHL1` `HTRA2` `EIF4G1` `GIGYF2`
`NR4A2` `PODXL` `RIC3` `PSAP`. **Excluded on the degeneration criterion:** `SLC30A10`
(manganese toxicity, pallidal), `ATP1A3` (ion-pump dysfunction), `PRKRA`, `WDR45`, `PANK2`,
`C19orf12` (NBIA, pallidal/SN iron), `ARSA` (leukodystrophy), `VPS16`.

### 3.4 `PD_GWAS_L2G` (41 genes)

Common-variant tier, to demonstrate that the earlier failure was locus-to-gene mapping rather
than GENCIC. Assignment uses **Open Targets locus-to-gene ML scores ≥ 0.5**, pooled across
seven PD GWAS (`GCST009325` Nalls 2019, `GCST009324` Nalls 2019, `GCST004902` Chang 2017,
`GCST002544` Nalls 2014, `GCST003984` Pickrell 2016, `GCST010049` Foo 2020, `GCST009512`
Bandres-Ciga 2019), retrieved from the Open Targets Platform GraphQL API. Nearest-gene
assignment is not used.

Filtered as known artifacts: `FLG` `HRNR` `MUC19` (1q21 epidermal-differentiation cluster and
mucin repeat region).

Resulting set: `LRRK2` `SNCA` `TMEM175` `GPNMB` `BST1` `MCCC1` `ACMSD` `RIT2` `STK39` `SH3GL2`
`CTSB` `ITPKB` `SV2C` `DLG2` `MAPT` `RAB29` `SIPA1L2` `NUCKS1` `INPP5F` `TMEM163` `FGF20`
`CAMK2D` `HIP1R` `BAG3` `GALC` `TOX3` `KLHL7` `IGSF9B` `SLC45A3` `MAP4K4` `GPR65` `FAM47E`
`PRICKLE1` `PLEKHH1` `NDUFAF2` `ITGA8` `TMEM229B` `PM20D1` `ERCC8` `PKP2` `SLC50A1`

The full 119-gene scored pull is cached to `dat/Disease_Validation/pd_l2g_opentargets.json`
with per-gene score and source study, so the threshold is auditable and re-derivable.

### 3.5 `HD_HTT` (1 gene)

`HTT`. The Huntington's test proper. Requires the single-gene expression-matched null
(§5.2). Expected to be negative; reported without rescue.

### 3.6 `StriatalDegeneration` (9 curated, 8 usable)

Mendelian syndromes with documented striatal degeneration. **Labeled as a validation of
"genetic striatal degeneration", never as a Huntington's validation.**

`HTT` `JPH3` (HDL2) `TBP` (SCA17/HDL4) `VPS13A` (chorea-acanthocytosis) `XK` (McLeod)
`FTL` (neuroferritinopathy) `ATN1` (DRPLA) `PRNP` (HDL1) `C9orf72` (commonest HD phenocopy)

**`FTL` is absent from the 17,180-gene Z2 matrix** (verified 2026-08-05), so the effective set
is 8 genes. It is retained in `gene_sets.csv` with `in_matrix=False` and reported in the
methods rather than silently dropped. All other genes in all six sets resolve to Entrez and are
present in the matrix; `GBA1` resolves directly, with `GBA` as a fallback alias.

Excluded on the degeneration criterion: `NKX2-1` `FRRS1L` `ADCY5` `GPR88`. Excluded as
circular striatal markers: `PDE10A` `PDE2A` `PDE8B`. Considered and excluded as biologically
distinct: PFBC calcification genes (`SLC20A2` `PDGFB` `PDGFRB` `XPR1` `MYORG` `JAM2`), metal
metabolism (`ATP7B` `CP`), pediatric striatal necrosis (`NUP62` `SLC19A3` `SLC25A19`),
mitochondrial Leigh syndrome.

Caveat to state in the manuscript: `ATN1` (DRPLA) is dentatorubral-pallidoluysian rather than
striatal-selective, and `C9orf72` pathology is FTLD/ALS-type. Both are retained because they
are established HD phenocopies, but they are the weakest members of the panel.

## 4. Pre-Registration

`config/disease_validation_ground_truth.yaml` is written by notebook 01 and **git-committed
before notebook 02 is executed**. The commit hash is cited in the rebuttal. This is what makes
"pre-registered" a verifiable claim rather than an assertion.

The file lives in `config/`, not `dat/`, for a load-bearing reason: this repository's
`.gitignore` uses a whitelist and excludes `dat/`, `results/`, and `docs/` entirely, so an
artifact placed in `dat/` **cannot be committed and the pre-registration claim would be
unverifiable**. `config/*.yaml` is tracked (verified 2026-08-05). The curated gene lists are
committed alongside it as `config/disease_validation_genesets.yaml` for the same reason.

### 4.1 Structure-level ground truth

**Parkinson's — core** (nigrostriatal + basal ganglia motor loop):
`Substantia_nigra_compact_part`, `Substantia_nigra_reticular_part`, `Ventral_tegmental_area`,
`Caudoputamen`, `Fundus_of_striatum`, `Nucleus_accumbens`,
`Globus_pallidus_external_segment`, `Globus_pallidus_internal_segment`, `Subthalamic_nucleus`,
`Ventral_anterior_lateral_complex_of_the_thalamus`, `Ventral_medial_nucleus_of_the_thalamus`,
`Primary_motor_area`, `Secondary_motor_area`

**Parkinson's — Braak-early extension** (secondary test):
`Dorsal_nucleus_raphe`, `Main_olfactory_bulb`, `Anterior_olfactory_nucleus`,
`Pedunculopontine_nucleus`

Locus coeruleus and dorsal motor nucleus of the vagus are **absent from the 213-structure
atlas**, so Braak stages 1–2 are only partially testable. This limitation is stated in the
manuscript rather than worked around.

**Striatal degeneration / HD — core:**
`Caudoputamen`, `Nucleus_accumbens`, `Fundus_of_striatum`,
`Globus_pallidus_external_segment`, `Globus_pallidus_internal_segment`,
`Substantia_nigra_reticular_part`, `Subthalamic_nucleus`

**Striatal degeneration / HD — late-stage extension:** `Primary_motor_area`,
`Secondary_motor_area`

All names verified present in `STR2Region()`, the Z2 matrix, and `InfoMat.Ipsi.csv`.

### 4.2 Cell-type ground truth

ABC atlas clusters, matched by subclass:

- **Parkinson's:** `SNc-VTA-RAmb Foxa1 Dopa`
- **Striatal degeneration / HD:** `STR D1 Gaba`, `STR D2 Gaba`, `STR D1 Sema5a Gaba`,
  `ACB-BST-FS D1 Gaba`

Basal-ganglia-loop subclasses reported as secondary context: `GPe-SI Sox6 Cyp26b1 Gaba`,
`GPi Tbr1 Cngb3 Gaba-Glut`, `SNr Six3 Gaba`, `SNr-VTA Pax5 Npas1 Gaba`, `STN-PSTN Pitx2 Glut`.

## 5. Null Models

### 5.1 Uniform random (primary)

Sample N genes uniformly from the 17,180 matrix genes, 10,000 permutations.
`Snakefile.bias` already defaults `null_model` to `uniform` (passes `GeneProb=None`), so this
requires no code change. Chosen as primary because it is what the published NT positive
controls and non-brain negative controls used, keeping PD/HD directly comparable to them.

The mutability/sibling null is **not applicable** — it is specific to de-novo mutation gene
sets and encodes a per-gene mutation probability that has no meaning for curated Mendelian
lists.

### 5.2 Expression-decile-matched (robustness)

Bin all 17,180 matrix genes into deciles by `EXP` from
`dat/allen-mouse-exp/ExpMatchFeatures.csv` (verified to cover all 17,180). For a gene set,
count members per decile and sample the same per-decile counts, 10,000 permutations. Answers
the objection that brain-expressed, neuronally-restricted genes are enriched in neuron-dense
structures regardless of disease.

For `HD_HTT`, this is the same procedure with N=1 — the null is 10,000 single genes drawn from
`HTT`'s expression decile. This is the only defensible null for a single-gene test.

**Implementation:** new `null_model: "expmatched"` branch in `get_gene_prob()`
(`Snakefile.bias:21`) plus a matched sampler in `scripts/script_generate_geneweights.py`
alongside `RandomGenes()` and `SiblingGenes()`. Seed `42`.

## 6. Recovery Metrics

For each gene set × each ground-truth set:

- **Mann-Whitney U**, one-sided, testing that ground-truth structures rank higher on `EFFECT`
  than all remaining structures. Primary metric.
- **AUROC** of ground-truth membership against the bias ranking.
- **Precision@20**.
- **Permutation p-value**: recompute the statistic under 10,000 random structure sets of
  matched size.

### 6.1 Gene-set-size sensitivity

Re-run the recovery test on nested subsets (5, 10, 15, 20 … genes ordered by evidence tier) and
on leave-one-out. This answers "is one or two genes carrying the whole result" — the objection
a careful reviewer will raise about `LRRK2` in the PD set.

### 6.2 Negative-control cross-test

Run the PD and striatal ground-truth recovery tests against the **existing** bias files for
`T2D`, `IBD`, `HDL_C`, `hba1c` (`results/STR_ISH/*_bias_addP_*.csv`). These are already
computed, so the test is free, and it demonstrates recovery is disease-specific rather than a
generic property of any gene set.

### 6.3 Circuit connectivity

CCS profile across circuit sizes 200→6 via `ScoreCircuit_SI_Joint` against the null band,
matching the existing NT figure. Tests whether top-biased structures form a *connected*
circuit, not merely a plausible list. Reported with p-values at the FDR-derived and CCS-peak
sizes.

## 7. Circuit Search

All six gene sets through the standard pipeline: bias limits → simulated annealing → Pareto
front → best circuit, via a new `config/circuit_config_disease.yaml`.

Circuit sizes are derived **per gene set** from that set's own FDR<0.10 structure count and CCS
profile peak (≈4 sizes each), not inherited from ASD's 46. SA parameters follow
`circuit_config.yaml`: `sa_runtimes: 100`, `sa_steps: 100000`, `measure: SI`, `min_bias_rank: 50`.

**Runtime is unestimated.** One size for one gene set will be benchmarked first and the
extrapolated wall-clock reported before the full sweep is launched. Current machine state is
idle (48 cores, 161 GB available). If the extrapolation is unreasonable, the fallback is to
reduce to 2 sizes per set, reported explicitly rather than silently.

Resulting circuits are plotted against known anatomy: nigrostriatal / basal ganglia motor loop
for PD, striatopallidal for the striatal panel.

## 8. Cell-Type and MERFISH Arms

**Cell type.** DN gene weights (`weight_DN = weight × max(spearman_r, 0)²`) generated for all
six sets using the V2–V3 Spearman reproducibility procedure from
`notebooks_mouse_sc/02.Cell_Type_Bias`. Cluster-level bias over 5,312 ABC clusters via
`MouseCT_AvgZ_Weighted` with 10,000-permutation nulls, registered in `config/config.SC.DN.yaml`.
Recovery tested by the §6 metrics on the pre-registered clusters. Subclass-level boxplots for
display.

Per project rule, DN weights are used for cell-type analysis **only** — never for the
structure-level arm.

**MERFISH.** Structure-level bias computed from the four MERFISH Z2 matrices, following
`notebooks_mouse_sc/04.MERFISH_Structure_Bias`. Reports ISH↔MERFISH concordance for PD and the
striatal panel as an independent-modality replication.

## 9. Deliverables

New directory `notebooks_disease_validation/`, three jupytext-paired notebooks:

| Notebook | Contents |
|---|---|
| `01.Gene_Curation.ipynb` | Builds all six `.gw` files and DN variants from explicit symbol lists; resolves via `LoadGeneINFO()`; reports genes missing from the expression matrix; fetches and caches the Open Targets L2G pull; writes the two tracked `config/disease_validation_*.yaml` files. **Committed before 02 runs.** |
| `02.STR_Bias_and_Circuits.ipynb` | Bias + both nulls; recovery metrics; size-sensitivity; negative-control cross-test; CCS profiles; SA circuit search; circuit-vs-anatomy figures. |
| `03.CellType_and_MERFISH.ipynb` | Cluster-level bias + recovery tests; subclass boxplots; MERFISH structure bias and ISH concordance. |

**New/changed files:**

- `config/disease_validation_genesets.yaml` — curated lists (symbol, entrez, set, tier,
  syndrome, justification, in_matrix). **Tracked.**
- `config/disease_validation_ground_truth.yaml` — pre-registered structures and cell types.
  **Tracked, committed before notebook 02 runs.**
- `dat/Disease_Validation/pd_l2g_opentargets.json` — cached Open Targets pull (untracked;
  regenerable by notebook 01, which is why it may live under `dat/`)
- `dat/Genetics/GeneWeights/` — six new `.gw`; `dat/Genetics/GeneWeights_DN/` — six new `.DN.gw`
- `config/config.STR.yaml`, `config/config.SC.DN.yaml` — six new gene-set entries
- `config/circuit_config_disease.yaml` — new
- `Snakefile.bias`, `scripts/script_generate_geneweights.py` — `expmatched` null branch
- `DATA_MANIFEST.yaml` — entries for all new data files

**Outputs:** `results/STR_ISH/`, `results/CT_Z2/`, `results/CircuitSearch/`,
`results/figures/`, cache in `results/cache/`.

**Conventions:** seed `42` throughout; all paths relative or config-driven; figures
`transparent=True, dpi=300, bbox_inches='tight'`; region colors imported from
`src/plot.REGION_COLORS`; reusable plotting functions extracted to `src/plot.py`.

## 10. Pre-Specified Interpretation

Stated before results exist, so the conclusion is not fitted to the outcome.

**Success for PD** = `PD_Primary` recovers the core nigrostriatal ground truth at
Mann-Whitney p<0.05 under both nulls, with the result stable under leave-one-out and not
reproduced by the non-brain negative controls.

**Partial success** = recovery significant only when `PD_Sens_DA` markers are included. This
would be reported as evidence that the framework needs marker-level genes, which is a weaker
claim than the reviewer asked for, and will be labeled as such.

**Failure for PD** = no significant recovery in any tier. This would be reported. It would mean
the ISH expression bias does not encode nigrostriatal vulnerability, which is a genuine
limitation of the framework and material to the manuscript's central claim.

**`HD_HTT` is expected to fail** and its failure is not evidence against the framework — it is
the scope boundary described in §2, pre-registered as such here.

**The striatal panel is the informative uncertain case.** If it recovers the striatum, the
framework generalizes beyond neurotransmitter systems to degenerative disease. If it recovers
the striatum only via `PDE10A`-class markers (excluded here by design), that would have been
circular and is why they were excluded. If it fails outright, the honest conclusion is that
GENCIC's demonstrated scope is neurodevelopmental and neurotransmitter-system biology, not
adult-onset neurodegeneration — and the manuscript should say so.

## 11. Open Risks

1. **PD recovery may depend on `LRRK2` alone.** `LRRK2` is the single strongest striatal
   signal in `PD_Primary`. The leave-one-out analysis (§6.1) is the check; if recovery
   collapses without it, that must be reported.
2. ~~`GBA1` symbol resolution.~~ Resolved 2026-08-05: all 30 curated symbols across the six sets
   resolve to Entrez, and only `FTL` is missing from the Z2 matrix (§3.6).
3. **Open Targets coverage.** Only the public (23andMe-excluded) Nalls summary statistics were
   ingested, so `GCST009325` yields 25 credible sets rather than 90 signals. Pooling seven
   studies mitigates this but the tier is not a complete representation of the 90 loci; this is
   stated in the methods.
4. **SA runtime unknown** for six gene sets — see §7.
5. **Deprecated files.** `Parkinson.top61.gw`, `Parkinson.gw`, and `ALZ.top60.gw` should be
   marked deprecated so they are not reused; any existing figure or table drawing on them needs
   an audit. `notebooks_mouse_str/10.Positive_Control_Circuits.py` reads
   `Parkinson_bias_addP_sibling.csv` (currently commented out of the plotted set) and rewrites
   `Parkinson.gw` from a hardcoded 5-gene list at line 177 — that cell must be reconciled with
   the new sets or removed.
6. **This design document is untracked.** `docs/` is gitignored project-wide (`.gitignore:34`,
   zero docs currently tracked). The spec therefore lives on disk only unless force-added. The
   pre-registration artifacts are unaffected — they live in tracked `config/` per §4.
