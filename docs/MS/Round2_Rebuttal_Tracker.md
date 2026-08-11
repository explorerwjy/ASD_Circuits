# Round 2 Rebuttal — Status Tracker

**Manuscript:** GENCIC / ASD circuits — `*_resub_final` bundle + `Reviewers_Reponse_Letter_resub_33.docx`
**Comments source:** `docs/MS/ReviewComments_Round2.docx`
**Scope of this file:** Reviewer #2 in detail; Reviewer #1 indexed only (22 comments, to be scoped later)
**Last updated:** 2026-08-11

**Referee expertise (stated in the file):** #1 connectomics + computational · #2 ASD genetics/genomics + computational.
R#2 explicitly defers circuit-biology judgement to R#1 — so R#2 answers should lead with statistics,
controls and genetics, not neuroanatomy.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete — analysis done, artifacts committed, response drafted |
| 🟩 | Near complete — analysis done, packaging/figures outstanding |
| 🟨 | In progress |
| ⬜ | Not started |

---

## Reviewer #2 — overview

| ID | Topic | Status | Effort | Notes |
|---|---|---|---|---|
| R2-M1 | Why mouse? ASD-relevant conservation; mouse models | ⬜ | Medium | Needs literature work, not new computation |
| **R2-M2** | **Validate on a disease with known genes+circuit (PD/HD)** | **🟩** | — | **Essentially done — see below** |
| R2-M3 | What does a single "ASD circuit" mean given heterogeneity? | ⬜ | Low–Med | Conceptual/discussion; possibly reuse gene-clustering work |
| R2-M4a | Why expression *specificity* rather than overall expression? | ⬜ | Low | Justification + possibly a supporting analysis |
| R2-M4b | Is ASD>sibling built into the metric? Alternative nulls | ⬜ | **Medium** | **Reusable: the 4-null framework built for R2-M2** |
| R2-M5 | Circuit implicates much of the brain — specificity? core circuit? | ⬜ | Medium | **PD circuit (13 structures, 9-structure core) is a useful precedent** |
| R2-m1 | Satterstrom/Fu overlap with SPARK — temper "independent" | ⬜ | Low | Compute overlap, soften language |
| R2-m2 | CCS interpretation — what does the maximum mean? | ⬜ | Low–Med | Ties to R2-m3 |
| R2-m3 | Are connectivity and bias really competing objectives? | ⬜ | Low | **We have direct evidence from the PD Pareto analysis** |
| R2-m4 | Same metric used for ranking, sizing, optimizing AND evaluating | ⬜ | **Medium** | Held-out validation requested; PD/HD IS external validation |

**Cross-cutting opportunity.** R2-M2 produced machinery that answers several other comments:
the expression-decile-matched null and the four-null comparison (R2-M4b), a small high-confidence
circuit with a stable core (R2-M5), the Pareto operating-point analysis (R2-m3), and an
externally-validated test set (R2-m4). Draft those four with explicit pointers to the PD work.

---

### R2-M2 — Validate the framework on a disease with established causal genes 🟩

**Comment (verbatim).** "The CCS/GENCIC framework is the primary methodological contribution of this
work, so I would like to see stronger validation of its ability to recover biologically meaningful
disease-associated circuits. The only validation presented in the main text is the dopamine circuit,
with two additional neurotransmitter analyses in the supplement. These examples are reassuring sanity
checks but do not fully validate the broader claim that disease-associated gene sets can recover
disease-relevant neural circuits. Neurotransmitter systems are much more evolutionarily conserved than
complex neurodevelopmental disorders. I would find it more convincing to validate the framework using
a disease with both well-established causal genes and a well-characterized affected circuit, such as
Parkinson's disease and/or Huntington's disease."

**Headline results** — main gene set `PD_HighConf_DA`, 19 literature-backed Mendelian PD genes:

| Level | Result |
|---|---|
| Cell type (dopaminergic neurons) | AUROC 0.986, gene-set-null **p = 1×10⁻⁴** |
| Specificity | 7 comparison sets all n.s. on same target; DDD hits MSNs instead → double dissociation |
| Structure (blinded, anatomy-matched) | PD-affected rank 12 vs 54; **p = 0.0025** consensus, κ = 0.59 |
| Structure (FDR) | 15 structures q<0.10; VTA / dorsal raphe / SNc all q ≈ 0.004 |
| Circuit (Pareto, ASD-matched point) | SNc + VTA + 3 raphe retained at all 4 sizes; 9-structure consensus core |
| **Pre-registered composite AUROC** | **FAILS (p = 0.11)** — reported in full, with explanation |
| Huntington's | Negative — HTT depleted in MSNs (ranks 278–292/340). Reported as scope boundary |

**Deliverables — DONE**

- Notebooks (all execute clean, 71 assertions total):
  `01.Gene_Curation` · `02.STR_Bias_and_Recovery` (`d672e04`) ·
  `03.CellType_Recovery` (`4721692`) · `04.Pathology_Eval_and_Circuits` (`8e48da3`)
- 20 tables + 9 figures in `results/PD_HD_validation/`
- 49-entry BibTeX: `results/PD_HD_validation/PD_HD_validation_references.bib`
- Draft response + Supplementary Note section: `docs/MS/Reviewer2_Q2_PD_HD_validation_DRAFT.md`
- Pre-registration commits: gene sets/ground truth `b79558d`; high-confidence tier `a3ae114`

**Outstanding**

- [ ] Manuscript-grade figures (the 9 are analysis-grade — need panel design, sizing, lettering)
- [ ] MERFISH replication (independent spatial modality; not run)
- [ ] Assign final S-numbers to tables and format for submission package
- [ ] JW review of the draft response and Supplementary Note text

**Caveats that must survive into the final text**

- The pre-registered composite AUROC failed; the blinded pathology analysis is post-hoc, though its
  criterion and design were fixed before ratings were produced. Report both, and say which was
  specified in advance.
- Two earlier formulations of the precision test failed or were confounded. Notebook 04 documents all
  three; the response should describe the sequence rather than only the successful test.
- Inter-rater κ = 0.59 (moderate) — quote consensus, not either single rater.
- SA operates on `EFFECT`, which is null-independent, so the uniform-null and sibling-null circuit
  searches are identical by construction — 4 distinct circuits, not 8.
- The atlas does not resolve SNc from VTA (single transcriptomic subclass), so the ventrolateral-SNc
  gradient that defines PD selectivity cannot be tested; our ranking puts VTA above SNc.
- `LRRK2` behaves oppositely to the other 18 genes (striatal, not midbrain) — consistent with the
  frequently non-synucleinopathic pathology of LRRK2-associated disease.

---

### R2-M1 — Justify the use of mouse data for a human-specific disorder ⬜

Asks: why are ASD-relevant circuits expected to be conserved? How many analysed ASD genes have mouse
models with ASD-relevant phenotypes? Do the implicated mouse circuits overlap circuits disrupted in
existing ASD mouse models?

- [ ] Count ASD genes in the analysed set with published mouse models showing ASD-relevant phenotypes
      (candidate source: SFARI Gene animal-model annotations)
- [ ] Literature survey: circuits disrupted across ASD mouse models vs our implicated structures
- [ ] Draft the conservation argument; concede human-specific phenotypes explicitly
- Note: mostly literature work. The R2-M2 curation pipeline (Europe PMC + verified PMIDs) is reusable.

### R2-M3 — What does a single "ASD circuit" mean given heterogeneity? ⬜

Asks: convergence on a common circuit, or an average population-level vulnerability landscape?

- [ ] Decide and state the claim explicitly
- [ ] Consider reusing the existing gene-clustering analysis to show convergence vs heterogeneity
- [ ] Discussion text on how genetic/phenotypic heterogeneity fits the framework

### R2-M4a — Why expression specificity rather than overall expression? ⬜

- [ ] Biological justification for specificity (Z2) over mean expression
- [ ] Consider a supporting analysis: repeat the main result using overall expression, show it is weaker

### R2-M4b — Is the ASD>sibling result built into the metric? ⬜ **(reuse R2-M2)**

Asks which analyses are independent of the mutation weighting, and whether alternative nulls agree.

- [ ] State plainly which analyses are weighting-independent
- [ ] **Reuse the four-null framework from R2-M2** (uniform random, expression-decile-matched,
      sibling-uniform, sibling-mutability). Code: `scripts/script_generate_geneweights.py`
      (`expmatched` branch) + `Snakefile.bias`
- [ ] Report ASD under all four nulls as PD was
- Note: the expression-decile-matched null also partly answers **R#1 comment 7 and 9**.

### R2-M5 — Is the circuit too large to be specific? Is there a core? ⬜

- [ ] Report the dynamic range of bias values
- [ ] Address whether a 46-structure circuit is typical for complex disorders
- [ ] Identify a smaller high-confidence core for ASD
- Note: PD gives a strong precedent — a 13-structure circuit with a 9-structure core stable across
  sizes 11–20. The same consensus-core method transfers directly.

### R2-m1 — Temper "independent" for Satterstrom / Fu gene sets ⬜

- [ ] Compute exact overlap with SPARK; report it
- [ ] Soften "independent validation" wording throughout

### R2-m2 — CCS interpretation ⬜

- [ ] Explain what maximum CCS means biologically and how to read absolute values
- [ ] Justify why peak CCS sets the preferred circuit size

### R2-m3 — Are connectivity and bias genuinely competing? ⬜ **(evidence exists)**

- [ ] Answer using the PD Pareto analysis: they *are* competing — at the permissive end the optimiser
      drops SNc/VTA for densely-connected reticular formation, while at the ASD-matched operating
      point (−20% bias, +77–137% CCS) the dopaminergic core is retained. That trade-off is exactly
      what a Pareto formulation is for. Figure `04b_pareto_fronts_operating_points.png`.

### R2-m4 — Same metric used for ranking, sizing, optimizing and evaluating ⬜

- [ ] **Point at R2-M2 as external validation**: the PD analysis evaluates against an independent
      criterion (published human neuropathology, blinded two-rater) rather than the bias metric itself
- [ ] Consider a held-out validation for ASD

---

## Reviewer #1 — 22 comments, not yet scoped ⬜

Connectomics/computational referee. Index only; scope after R#2.

1 terminology ("mutation bias" overclaims) · 2 weighting sensitivity (equal weights, leave-one-out,
remove recurrent genes, risk-based weights) · 3 LGD-only vs Dmis-only · 4 orthology mapping ·
5 developmental-stage mismatch · 6 Fig 2 model/controls · 7 matched null models (expression, length,
mutability, LOEUF, network degree) · 8 compositional confounding · 9 how expression-matched sampling
was done · 10 FDR 0.1 too permissive; spatial permutation · 11–22 to be transcribed.

**Overlap with R#2 work already done:** R#1-2 (leave-one-out) and R#1-7/9 (matched nulls) are directly
served by the R2-M2 machinery. R#1-1 (terminology) affects wording throughout and should be resolved
before final text is written.

---

## Cross-cutting

- [x] Locate the round-2 bundle — arrived 2026-08-11 in `docs/MS/`
- [ ] Transcribe R#1 comments 11–22 into this tracker
- [ ] Decide where PD/HD lands: new Supplementary Note section vs main-text figure
- [ ] Check round-1 response text for consistency with the R2-M2 answer
- [ ] Deprecate legacy `Parkinson.top61.gw` / `ALZ.top60.gw`; audit any figure/table built on them
- [ ] Confirm `Supplementary_Tables_resub_final.xlsx` S-numbering before assigning S-PD numbers
