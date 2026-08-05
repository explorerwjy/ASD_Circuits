# Preprocessing Pipeline: Z2 Matrix Comparison (Feb 2026)

## Overview

During the notebook 02 rework, the Z2 expression-matched z-score matrix was
regenerated from scratch. This document records the differences between the
**old reference** (produced ~May 2022) and the **new pipeline** (Feb 2026),
and traces the root cause through each processing layer.

## File Inventory

### Old reference (from `/home/jw3514/Work/ASD_Circuits/`)

| File | Shape | Date | Notes |
|------|-------|------|-------|
| `dat/allen-mouse-exp/ExpLevel.csv` | 17196 x 213 | May 2022 | Expression level matrix |
| `dat/allen-mouse-exp/JW_Z1-Mat.ArithmeticMean.0422.csv` | 17115 x 213 | May 2022 | Z1 matrix (fewer genes than ExpLevel — some dropped) |
| `dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv.gz` | 17180 x 213 | Apr 2025 | Z2 matrix (parquet copy saved as `.parquet.bak` in new project) |
| `dat/allen-mouse-exp/mouse2sectionID.0420.json` | 20600 genes | Apr 2020 | Mouse gene → ISH section ID mapping |
| `dat/genes/ExpMatch_RootExp_uniform_kernal/` | ~17189 files | May 2022 | Expression match files (10K samples each) |

### New pipeline (current project)

| File | Shape | Date | Notes |
|------|-------|------|-------|
| `dat/BiasMatrices/AllenMouseBrain_ExpLevel.parquet` | 17198 x 213 | Feb 2026 | Also saved as `dat/allen-mouse-exp/ExpLevel.csv.gz` |
| `dat/BiasMatrices/AllenMouseBrain_Z1.parquet` | 17198 x 213 | Feb 2026 | Also saved as `dat/allen-mouse-exp/ExpLevel.log2.Zscore.csv.gz` |
| `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` | 17198 x 213 | Feb 2026 | Also saved as `dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv.gz` |
| `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet.bak` | 17180 x 213 | — | Backup of old Z2 reference before overwrite |
| `dat/allen-mouse-exp/mouse2sectionID.0420.json` | 20600 genes | Feb 2026 | **Merged** from old + new Allen API (union of section IDs) |
| `dat/allen-mouse-exp/ExpMatch/` | 17191 files | Feb 2026 | Expression match files (seed=42, 10K samples, ±5% quantile) |
| `dat/allen-mouse-exp/ExpMatchFeatures.csv` | 17191 genes | Feb 2026 | Root expression + quantile per gene |

## Gene Count Differences

| Dataset | Old | New | Notes |
|---------|-----|-----|-------|
| `mouse2sectionID` mapping | 20600 | 20600 | Same gene count, but 68 mouse genes lost section IDs in new API |
| ExpLevel | 17196 | 17198 | +2 genes (new API added 3 section IDs, merged mapping recovers old) |
| Z1 | 17115 | 17198 | Old Z1 dropped ~81 genes (reason unclear) |
| Z2 | 17180 | 17198 | New is superset of old; all 17180 old genes present |

### Mapping file merge

The new Allen API (Feb 2026) returned empty `allen_section_data_set_id` for
68 mouse genes that had data in the original download (Apr 2020). These 279
section IDs were likely retired/deprecated by Allen Institute. Since the ISH
CSV files still exist locally, we **merged** old and new mappings (union of
section IDs) to preserve coverage.

Merge stats:
- 238 mouse genes gained section IDs from old mapping
- 3 new section IDs from new API
- All 279 "lost" section IDs have ISH CSV files available

## Layer-by-Layer Comparison (new vs old, on common genes)

### Layer 1: Expression Level (ExpMat)

```
Common genes:      17196
Pearson r:          0.976
Mean abs error:     0.283
Max abs error:      3.15
Exact matches:      0.05%
Per-gene r:         mean=0.991, median=0.992, min=0.964
```

**This is the primary source of difference.** Nearly all values differ because
the old and new pipelines average different sets of ISH experiments per gene
(279 section IDs differ between old and new mappings). Even for genes with
identical section IDs, floating-point order of operations may differ.

### Layer 2: Z1 (Per-Gene Z-Score)

```
Common genes:      17115
Pearson r:          0.998
Mean abs error:     0.004
Max abs error:      7.89
Exact matches:      28.8%
```

Z-scoring within each gene absorbs most ExpMat differences. The 29% exact
matches are genes where both pipelines had identical ISH inputs.

### Layer 3: Z2 (Expression-Matched Z-Score)

```
New Z2 vs old reference:
  Common genes:     17180
  Pearson r:        0.949
  Mean abs error:   0.214

Z2 sensitivity to random seed (new ExpMat, seeds 42/123/999):
  Pairwise r:       0.9986
  Pairwise MAE:     0.030
```

**Z2 is barely sensitive to random seed** (r=0.999 between different seeds).
The r=0.949 gap vs old reference is almost entirely from the ExpMat layer
(different ISH experiments averaged), not from expression matching randomness.

### Downstream: ASD Bias (Nucleus accumbens EFFECT)

| Source | EFFECT |
|--------|--------|
| Old reference | 0.5046 |
| New (match-file, seed=42) | 0.4750 |
| New (on-the-fly, seed=42) | 0.4694 |
| New (on-the-fly, seed=123) | 0.4644 |
| New (on-the-fly, seed=999) | 0.4787 |

Seed-driven variation: ~±0.015. Old-vs-new gap: ~0.030.
The gap is ~2x the seed noise, confirming it's driven by different ISH data,
not by expression matching randomness.

### ASD Bias Correlation (full 213 structures)

```
New vs old:  Pearson r = 0.971
Top 10 overlap: 7-8 / 10
```

## Root Cause Summary

```
ExpMat (r=0.976)  →  Z1 (r=0.998)  →  Z2 (r=0.949)
                                         ↑
                              Expression matching amplifies
                              small Z1 differences because
                              matched gene sets also differ
```

1. **Primary cause**: 279 ISH section IDs differ between old and new Allen API
   downloads, causing different experiments to be averaged per gene.
2. **Secondary cause**: Expression matching (Z2 step) uses the same genes as
   both targets and controls. Small Z1 changes propagate through both the
   numerator and the matched-gene denominator, amplifying differences.
3. **Not a cause**: Random seed in expression matching (r=0.999 between seeds).

## Reproducibility Notes

- New pipeline uses **seed=42** for expression matching, saved in match files
- Match files are deterministic given the same `ExpMatchFeatures.csv`
- Z2 script supports both on-the-fly matching (`--seed`) and pre-computed
  match files (`--match-dir`) for reproducibility
- The merged `mouse2sectionID.0420.json` ensures future runs produce the same
  17198-gene ExpMat (assuming ISH CSV files remain available)
