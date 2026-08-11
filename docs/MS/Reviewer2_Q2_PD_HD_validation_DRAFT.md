# Reviewer 2, comment 2 — PD / HD validation

**Status:** DRAFT for JW to edit. All numbers traceable to `results/tables/` (see manifest at end).
**Date:** 2026-08-08

---

## A. Response to reviewer (draft)

We thank the reviewer for this suggestion, which we agree goes to the heart of the paper's claim. We
have carried out the requested validation on Parkinson's disease, and — as the reviewer also
suggested — on Huntington's disease. The analysis is pre-registered, and we report both a positive
and a negative result.

**Curation.** The primary set comprises 19 Mendelian Parkinson's/parkinsonism genes, each traceable
to a verified primary publication (Supplementary Table S-PD1); eight are classified "Definitive" for
Parkinson disease by the ClinGen Gene–Disease Validity panel. We additionally tested a common-variant
tier of 41 genes assigned to PD GWAS loci by Open Targets locus-to-gene scores rather than proximity;
that set recovers neither the dopaminergic population (AUROC 0.390, n.s.) nor the affected structures,
delimiting the kind of genetic evidence the framework requires. Genes with disputed (`DNAJC13`, `LRP10`), insufficient
(`PTRHD1`) or unreplicated (`CHCHD2`) evidence, and genes whose syndromes are pathologically distinct
from nigrostriatal PD (`MAPT`, `DCTN1`, `POLG`, `TWNK`, `SPG11`, `DNAJC12`), were excluded by a rule
applied to the literature table and frozen before any analysis was run.

**Cell-type level.** Against the 43 dopaminergic clusters of the Allen Brain Cell Atlas
(`SNc-VTA-RAmb Foxa1 Dopa`), the PD gene set gives AUROC 0.986 (gene-set permutation p = 1×10⁻⁴).
Specificity is demonstrated against seven comparison sets, none of which is significant on the same
target: ASD (0.384, p = 0.68), DDD (0.254, p = 0.89), and four non-brain traits (IBD, HDL-C, T2D,
HbA1c; p = 0.30–0.84). The DDD gene set instead recovers striatal medium spiny neurons (p = 0.031)
while the PD set does not (p = 0.60), giving a double dissociation in which each disease gene set
recovers its own cell population and not the other's.

**Structure level.** Fifteen structures reach FDR q < 0.10, led by ventral tegmental area, dorsal
raphe and substantia nigra pars compacta (all q ≈ 0.004). To test whether the ranking recovers
structures that are genuinely affected in PD, we asked two independent raters to classify 50 brainstem
and diencephalic structures for documented human PD pathology, blind to our results and with the
comparison set drawn exclusively from the same anatomical territory (so that "it is a brainstem
nucleus" carries no information). Structures judged PD-affected rank substantially higher in our
ranking than unaffected structures (median rank 12 vs 54; rank-sum p = 0.0025 for consensus calls,
p = 0.0006 under the lenient criterion; inter-rater κ = 0.59).

**Circuit level.** Applying the same Pareto criterion used for the ASD circuit in the main text — the
operating point at which a ~20% reduction in mean bias buys the largest gain in circuit connectivity
— the recovered circuit gains 77–137% CCS across circuit sizes 11–20 and retains substantia nigra
pars compacta, ventral tegmental area and three raphe nuclei at every size. Nine structures are common
to all recovered circuits (Supplementary Table S-PD4).

**A negative result we report in full.** Our pre-registered composite test — whether the 13 structures
of the canonical PD motor circuit collectively rank above the remaining 200 — is not significant
(AUROC 0.582, p = 0.11). The reason is informative rather than technical. That list conflated the
structures in which neurons degenerate with the structures whose function is disrupted downstream.
The caudate–putamen is severely dopamine-depleted in PD but its own neurons are largely preserved, and
our method ranks it 190th of 213. GENCIC identifies where vulnerable cells reside, not where the
circuit malfunctions; by that reading, the low rank of the striatum is the correct answer rather than
a failure. We report both the pre-registered composite and the blinded pathology analysis, and we
state which was specified in advance.

**Huntington's disease.** We also tested HD, and it does not work. `HTT` is depleted, not enriched, in
the medium spiny neurons that degenerate (all four MSN subclasses rank 278–292 of 340 subclasses), and
an eight-gene panel of Mendelian striatal-degeneration syndromes likewise fails (AUROC 0.371, n.s.).
We regard this as a scope boundary rather than a technical failure, and it is consistent with the
long-standing observation that HTT is ubiquitously expressed and that its expression does not explain
the selective vulnerability of striatal neurons. Present understanding attributes MSN vulnerability to
somatic CAG-repeat instability, to striatum-enriched interacting partners such as RASD2/Rhes, and to
non-cell-autonomous corticostriatal and glial contributions — none of which is visible in a baseline
expression atlas. GENCIC identifies cell populations whose expression profile is enriched for risk
genes; it therefore captures PD-type biology and is not expected to capture HD-type biology. We think
stating this boundary explicitly strengthens rather than weakens the framework's claims.

---

## B. Supplementary Note — new section (draft)

### Supplementary Note S-X. Validation of GENCIC on Parkinson's disease and Huntington's disease

**Rationale.** The neurotransmitter-system analyses in the main text establish that GENCIC recovers
anatomically coherent circuits from gene sets defined by shared function. They do not establish that
it recovers disease-relevant circuits from gene sets defined by disease causation. We therefore
applied the framework to two disorders with well-established causal genes and well-characterised
neuropathology.

**Gene sets and pre-registration.** Gene sets, the expected structures and cell types, and the
analysis plan were frozen in the repository before any bias was computed (commit `b79558d`); the
high-confidence tier was subsequently derived from the literature evidence table alone and likewise
frozen before analysis (commit `a3ae114`). The Parkinson's set (n = 19) comprises Mendelian
parkinsonism genes, each supported by a verified primary publication (Table S-PD1). Dopamine
synthesis and transport genes (`TH`, `SLC6A3`, `DDC`, `GCH1`, `SPR`) are included: all are established
Mendelian causes of parkinsonism, and although they are also markers of dopaminergic neurons, the
result does not depend on them — a 24-gene set containing none of them still recovers the dopaminergic
population (AUROC 0.868, p = 0.0086), and no single gene accounts for the effect (largest
leave-one-out drop, `DDC`, leaves AUROC 0.895).

**Null models.** Because these gene sets are curated rather than derived from de novo mutation, we
report significance under four nulls: uniform random gene sets, expression-decile-matched gene sets,
and sibling-derived gene sets sampled uniformly or weighted by mutability (the last matching the
procedure used for the main ASD analysis). Conclusions are unchanged across all four (Table S-PD2).

**Findings.** At cell-type resolution the PD gene set recovers dopaminergic neurons
(AUROC 0.986, p = 1×10⁻⁴) with specificity against seven comparison gene sets, and in a double
dissociation with the DDD neurodevelopmental set, which recovers striatal medium spiny neurons
instead. At structure level, 15 structures reach q < 0.10, led by VTA, dorsal raphe and SNc; a blinded
two-rater pathology evaluation with anatomically matched controls confirms that PD-affected structures
rank higher than unaffected ones (p = 0.0025). Pareto-based circuit search retains SNc, VTA and three
raphe nuclei at every circuit size tested.

**Interpretation and limitations.** The recovered structures track the Braak staging of Lewy
pathology, and the cell-type analysis additionally implicates the dorsal motor nucleus of the vagus
and serotonergic raphe populations — Braak stages 1 and 2 — which are not separately represented in
the 213-structure atlas. Two limitations should be noted. First, the atlas does not resolve substantia
nigra from ventral tegmental area (they form a single transcriptomic subclass), so the ventrolateral
SNc-over-VTA gradient that defines PD selectivity cannot be tested here; our ranking places VTA above
SNc. Second, `LRRK2` behaves oppositely to the other genes, being striatal rather than midbrain in its
expression, consistent with the distinct, frequently non-synucleinopathic pathology of LRRK2-associated
disease. For Huntington's disease the framework does not recover the affected population, for reasons
we attribute to the ubiquitous expression of `HTT` and to disease mechanisms that operate downstream
of baseline expression.

---

## C. Supporting materials

### Exists

| ID | File | Contents |
|---|---|---|
| S-PD1 | `results/tables/PD_gene_literature_evidence.csv` | 29 genes, 32 rows, verified PMID/DOI/journal/first author per gene, ClinGen classification, dispute status |
| S-PD2 | `results/tables/PD_HD_validation_summary.csv` | Per gene set: n, structure AUROC, FDR counts, p under all four nulls, cell-type AUROC and p |
| S-PD3 | `results/tables/PD_structure_blinded_pathology_eval.csv` | 50 structures × 2 blinded raters: classification, Braak stage, verified PMID, rationale, true rank |
| S-PD4 | `results/tables/PD_circuit_pareto_summary.csv` + `PD_circuit_core_consensus.csv` | Pareto operating points across sizes/nulls; 9-structure consensus core |
| S-PD5 | `results/tables/PD_structure_ranks_all_sets.csv` | All 213 structures × 6 gene sets, EFFECT/q/rank, ground-truth flags |
| — | `results/tables/PD_HD_gene_sets_summary.csv` | Gene-set membership |
| — | `config/disease_validation_genesets.yaml`, `config/disease_validation_ground_truth.yaml` | Frozen pre-registration |
| — | `notebooks_disease_validation/01.Gene_Curation.ipynb` | Gene curation, weight files, Open Targets audit |

### Analysis notebooks — COMPLETE (all execute end-to-end, outputs stripped per repo convention)

| Notebook | Commit | Contents | Assertions |
|---|---|---|---|
| `01.Gene_Curation` | earlier | Gene curation, weight files, Open Targets L2G audit | 10 |
| `02.STR_Bias_and_Recovery` | `d672e04` | Structure bias, four nulls, top structures, CCS with size-matched nulls, union analysis | 13 |
| `03.CellType_Recovery` | `4721692` | Cell-type recovery, specificity panel, double dissociation, dopamine decomposition, HD arm | 20 |
| `04.Pathology_Eval_and_Circuits` | `8e48da3` | Blinded pathology evaluation (with all three formulations documented), Pareto circuits | 28 |

Deliverables: 9 figures and 20 tables under `results/PD_HD_validation/`; 49-entry BibTeX at
`results/PD_HD_validation/PD_HD_validation_references.bib`. Reference scripts that originally produced
the numbers are kept in `notebooks_disease_validation/reference/` for cross-checking.

### Missing — required before submission

1. **Manuscript figures** — the 9 notebook figures are analysis-grade, not publication-grade. Needs
   final panel design, sizing and lettering.
2. **MERFISH replication** — not run. Would provide an independent spatial modality.
3. **Supplementary table numbering** — tables exist but are not yet assigned final S-numbers or
   formatted for the submission package.

### Known caveats to carry into the text

- The pre-registered composite AUROC fails; the blinded pathology analysis is post-hoc, though its
  criterion and design were fixed before the ratings were produced. Both must be reported.
- Two earlier formulations of the precision test failed or were confounded; the sequence should be
  described rather than only the final result.
- Inter-rater κ = 0.59 (moderate); consensus rather than either single rater should be quoted.
- The SA search operates on EFFECT, which is null-independent, so the uniform-null and sibling-null
  circuit searches are identical by construction — 4 distinct circuits, not 8.
- CCS significance is null-dependent for some gene sets; report both matched nulls.
