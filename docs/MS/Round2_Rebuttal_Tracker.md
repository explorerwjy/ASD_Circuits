# Round 2 Rebuttal — Status Tracker

**Manuscript:** GENCIC / ASD circuits (resubmission 2)
**Scope of this file:** Reviewer #2 (Reviewer #1 to be added)
**Last updated:** 2026-08-11

> ⚠️ **Reviewer comments not yet ingested.** The round-2 bundle has not reached this machine
> (Archon) as of 2026-08-11 — nothing manuscript-related modified since March in `docs/MS/`,
> and no recent `.docx`/`.zip` anywhere under `/home/jw3514` or `/mnt/data0`. Only R2-Q2 (PD/HD)
> is recorded below, transcribed from chat. **Remaining comments are placeholders and must be
> filled in from the bundle before this tracker is usable for planning.**

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete — analysis done, artifacts committed, response drafted |
| 🟩 | Near complete — analysis done, packaging/figures outstanding |
| 🟨 | In progress |
| ⬜ | Not started |
| ❓ | Comment text not yet available |

---

## Reviewer #2

| ID | Topic | Status | Owner | Blocking |
|---|---|---|---|---|
| R2-Q1 | ❓ TBD — read from bundle | ❓ | — | bundle |
| **R2-Q2** | **PD / HD validation of CCS-GENCIC** | **🟩** | Claude | manuscript figures |
| R2-Q3 | ❓ TBD — read from bundle | ❓ | — | bundle |
| R2-Q4+ | ❓ TBD — read from bundle | ❓ | — | bundle |

---

### R2-Q2 — Validate the framework on a disease with established causal genes 🟩

**Comment (verbatim).** "The CCS/GENCIC framework is the primary methodological contribution of this
work, so I would like to see stronger validation of its ability to recover biologically meaningful
disease-associated circuits. The only validation presented in the main text is the dopamine circuit,
with two additional neurotransmitter analyses in the supplement. These examples are reassuring sanity
checks but do not fully validate the broader claim that disease-associated gene sets can recover
disease-relevant neural circuits. Neurotransmitter systems are much more evolutionarily conserved than
complex neurodevelopmental disorders. I would find it more convincing to validate the framework using
a disease with both well-established causal genes and a well-characterized affected circuit, such as
Parkinson's disease and/or Huntington's disease."

**Approach.** Re-curated PD gene sets from scratch (the legacy `Parkinson.top61.gw` was nearest-gene
GWAS output containing almost no real PD genes). Pre-registered gene sets and expected
structures/cell types before computing anything. Tested at structure, cell-type and circuit level
under four null models, with a blinded two-rater neuropathology evaluation and specificity controls.

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

### R2-Q1 — ❓ awaiting bundle

- [ ] Transcribe comment
- [ ] Scope the analysis
- [ ] Draft response

### R2-Q3 — ❓ awaiting bundle

- [ ] Transcribe comment
- [ ] Scope the analysis
- [ ] Draft response

### R2-Q4+ — ❓ awaiting bundle

- [ ] Determine how many comments Reviewer #2 raised
- [ ] Create an entry per comment

---

## Reviewer #1 — ❓ awaiting bundle

Not yet scoped. Add a matching section once the comments are available.

---

## Cross-cutting

- [ ] Locate the round-2 bundle (**not on Archon as of 2026-08-11**)
- [ ] Confirm bundle contents: main text, methods, supplementary note, response letter, both reviewers' comments
- [ ] Decide where the PD/HD material lands: new Supplementary Note section vs main-text figure
- [ ] Check whether any round-1 response text needs updating for consistency with R2-Q2
- [ ] Deprecate legacy `Parkinson.top61.gw` / `ALZ.top60.gw` and audit any figure or table built on them
