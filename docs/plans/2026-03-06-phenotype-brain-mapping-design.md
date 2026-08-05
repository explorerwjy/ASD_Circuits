# Phenotype-Specific Brain Mapping in ASD

**Date**: 2026-03-06
**Status**: Design

## Motivation

Notebook 06 shows that IQ-stratified ASD mutations target different brain regions (LIQ > HIQ in striatal/thalamic structures). Can we extend this to other phenotype dimensions — social deficits, motor impairment, repetitive behaviors, adaptive function, language — to build a multi-dimensional phenotype-to-brain map?

## Data Assets

### Mutations
- **SPARK**: 455 subjects with de novo LGD/Dmis mutations in 61 exome-wide significant ASD genes
- **SSC**: ~114 subjects (same gene set, same mutation filter)
- **ASC**: ~190 subjects (IQ data available, limited other phenotypes)
- **Combined**: ~570 SPARK+SSC subjects (expandable to 159-gene set if needed)

### Phenotype Instruments (estimated overlap with HC mutation carriers)

| Instrument | Domain | SPARK N | SSC N | Combined | Quality |
|-----------|--------|---------|-------|----------|---------|
| RBS-R | Repetitive behavior (6 subscales) | 331 | ~110 | ~440 | Parent-reported (SPARK) + parent-reported (SSC) |
| DCDQ | Motor coordination (3 subscales) | 270 | ~95 | ~365 | Parent-reported |
| Vineland-3/II | Adaptive (Communication, DLS, Social, Motor) | 186 | ~110 | ~300 | Clinician-administered (SSC) / parent-reported (SPARK) |
| SRS-2/SRS | Social (5 subscales) | 102 | ~110 | ~210 | Parent (SPARK+SSC) + teacher (SSC) |
| SCQ | ASD screening (40 items) | large | ~110 | ~400+ | Parent-reported |
| CBCL | Behavioral/emotional | ~90 | ~110 | ~200 | Parent-reported |
| IQ | Cognitive | 37 | ~110 | ~150 | Clinician (SSC) / sparse (SPARK) |
| Milestones | Developmental (walked, words) | ~200 | ~110 | ~310 | Parent-reported |

### Brain Expression
- ISH Z2 matrix: 17,208 genes x 213 structures (same as all existing analyses)
- Cluster Z2 matrix: for cell-type level extension (future)

## Architecture: `notebook_phenotype/`

### Notebook 01: Phenotype Data Cleaning & Integration

**Purpose**: Build a unified mutation-phenotype table linking SPARK + SSC subjects.

**Steps**:
1. Load mutation files (ASD_Discov_DNVs.txt, ASD_Rep_DNVs.txt)
2. Filter to 61 exome-significant genes + LGD/Dmis (same as notebook 06)
3. Link SPARK subjects to SPARK phenotype files via `subject_sp_id`
4. Link SSC subjects to SSC phenotype files via `individual` (IID)
5. Harmonize instruments across datasets:
   - RBS-R: identical 43-item instrument in both → combine directly
   - Vineland: SPARK uses V3, SSC uses V-II → use domain standard scores (comparable across versions)
   - SRS: SPARK uses SRS-2, SSC uses SRS → use T-scores (normed, comparable)
   - DCDQ: identical instrument → combine directly
   - CBCL: identical → combine directly
6. QC checks:
   - Flag subjects with `asd_validity_flag` (SPARK)
   - Remove family SF0006897 (invalid per release notes)
   - Check for `888` (not applicable) values in SPARK
   - Report missingness per instrument per cohort
7. Output: `results/phenotype/mutation_phenotype_master.parquet` — one row per subject, all phenotype scores + mutation info

**Key decisions**:
- Use **standard/T-scores** (not raw) for cross-dataset harmonization
- Keep SPARK and SSC flags so we can analyze jointly or separately
- For subjects with multiple mutations, sum gene weights (same as notebook 06)

### Notebook 02: Phenotype Stratification (Approach A)

**Purpose**: Binary phenotype splits → separate gene weights → brain bias comparison. Direct extension of notebook 06 methodology.

**Phenotype dimensions to stratify**:
1. **RBS-R total** (high vs low repetitive behavior) — best powered (~440 subjects)
2. **RBS-R subscales**: stereotypy vs sameness-dominant (within high-RBS group)
3. **DCDQ total** (motor-impaired vs motor-typical) — DCD cutoff available
4. **Vineland ABC** (high vs low adaptive function) — complement to IQ
5. **Vineland subscales**: Communication vs Socialization vs Motor (which domain drives the bias?)
6. **SRS total** (social-severe vs social-mild) — lower powered but directly tests social circuit hypothesis
7. **Age of first words** (language-delayed vs typical) — from milestones

**For each dimension**:
- Split at median (primary) and extreme quartiles (sensitivity)
- Compute gene weights per group: `Mut2GeneDF()` → `MouseSTR_AvgZ_Weighted()`
- Bootstrap CI (1000 resamples, cached)
- Permutation test for group difference (10,000 permutations, cached)
- Visualize: regional bar plots with significance bars (reuse notebook 06 plotting functions)

**Confound control**:
- Check that split groups don't differ systematically in total mutation count, sex ratio, or cohort composition
- If they do, include matched-sample sensitivity analysis

### Notebook 03: Continuous Phenotype-Brain Mapping (Approach B — backbone)

**Purpose**: For each brain structure, quantify how its expression-weighted mutation load relates to phenotype severity. No binary splits — use continuous scores.

**Method**:

For each subject *i* with phenotype score *p_i* and mutation set *M_i*:
- Compute per-subject structure bias vector: `b_i(s) = Σ_{g ∈ M_i} w_g × Z2(g,s)` where *w_g* is the gene weight
- This gives a (N_subjects × 213 structures) matrix

For each structure *s*:
- Compute Spearman correlation: `ρ(s) = corr(b_i(s), p_i)` across subjects
- Permutation p-value: shuffle phenotype labels 10,000 times
- FDR correction across 213 structures

**Output**: A "phenotype gradient map" — which brain structures show the steepest phenotype-bias relationship.

**Multi-dimensional version**:
- Run for each phenotype dimension independently
- Visualize as a (phenotype × structure) heatmap
- Cluster structures by their phenotype correlation profiles → reveals which structures co-vary with which phenotypes

**Confound regression**:
- Partial correlation controlling for: total mutation count, sex, age, cohort
- Compare SPARK-only vs combined results as sensitivity check

### Notebook 04: Data-Driven Phenotype Subtypes (Approach C — exploratory)

**Purpose**: Can we find phenotype subtypes that map onto distinct brain circuits?

**Steps**:
1. Build phenotype matrix: subjects (rows) × instrument scores (columns)
   - Only subjects with ≥3 instruments completed
   - Impute missing values with iterative imputation or restrict to complete cases
2. PCA/NMF on phenotype matrix → extract 3-5 components
3. Characterize components (e.g., "social-communication", "motor-RRB", "cognitive-adaptive")
4. For each component, compute component-weighted brain bias (like Approach B but using component scores)
5. Visualize: component-specific brain maps side by side

**Risk**: Parent-reported data may not yield clean components. Validate by:
- Comparing PCA structure in SPARK vs SSC
- Checking whether components are robust to leaving out one instrument at a time

### Notebook 05: SSC Cross-Validation

**Purpose**: SSC has gold-standard clinician data. Use it to validate SPARK-derived findings.

**Steps**:
1. For top findings from notebooks 02-04, replicate in SSC-only sample
2. Compare parent-reported (SRS) vs teacher-reported (SRS-teacher) vs clinician (ADOS item-level) findings in SSC
3. Use SSC's ADI-R data (not available in SPARK) as additional phenotype dimension
4. Quantify: does the phenotype-brain correlation survive when using clinician measures?

**This notebook is the credibility check** — if SPARK findings replicate in the smaller but cleaner SSC sample, they're real.

## Data Flow

```
dat/Genetics/SPARK/ASD_{Discov,Rep}_DNVs.txt   (mutations)
dat/Phenotype/ → /home/jw3514/Work/ASD_Phenotype/dat/  (symlink)
       ↓
Notebook 01: Clean & integrate
       ↓
results/phenotype/mutation_phenotype_master.parquet
       ↓
  ┌────────┼────────┬────────┐
  NB02     NB03     NB04     NB05
  Strat.   Contin.  Subtypes SSC val.
  ↓        ↓        ↓        ↓
results/phenotype/{stratification,continuous,subtypes,ssc_validation}/
```

## Caching Strategy

All expensive computations (bootstrap, permutation) cached to `results/phenotype/cache/`:
- `bootstrap_{instrument}_{group}.parquet` (1000 replicates × 213 structures)
- `permutation_{instrument}_{test_type}.csv` (p-values per structure)
- `phenotype_correlation_{instrument}.parquet` (correlation + p-value per structure)

## Success Criteria

1. **Minimum**: At least 2 phenotype dimensions show structure-specific bias differences (p < 0.05, FDR-corrected permutation test)
2. **Good**: Phenotype-specific brain maps are distinct from each other (different structures highlighted for social vs motor vs RRB)
3. **Great**: SPARK findings replicate in SSC; data-driven subtypes map onto interpretable brain circuits
4. **Publishable**: Multi-panel figure showing phenotype-specific brain circuit targeting in ASD

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Too few subjects after filtering | Medium | Expand to 159-gene set; use continuous approach (no split) |
| Parent-reported data too noisy | Medium | Cross-validate with SSC clinician data; use validated composite scores |
| All phenotype splits show same brain pattern | Low-Medium | Would still be interesting (ASD mutations hit same regions regardless of phenotype severity) |
| SPARK/SSC instrument versions not comparable | Low | Use standardized scores; analyze separately as sensitivity |
| Confounds (sex, IQ) drive apparent phenotype effects | Medium | Partial correlations; stratify by sex; check IQ correlation with each phenotype |

## Not In Scope (for now)

- Cell-type level phenotype mapping (future extension using cluster Z2 matrix)
- Circuit search with phenotype-specific bias (would need phenotype-stratified gene weight files)
- Longitudinal phenotype trajectories
- Genetic architecture differences (e.g., LGD-only vs Dmis-only per phenotype)
