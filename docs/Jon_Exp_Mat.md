# Jon's Expression Energy Matrix: Provenance and Aggregation Method

## Summary

Jon's raw expression energy matrix (`expression_energy-human-connectome_struc.csv`,
17,208 genes x 213 structures) was built from Allen Brain Atlas ISH data using
**arithmetic mean of raw expression energy** across ISH experiments, then log2(1+x).

This differs from JW's `ExpLevel.csv` which applies log2(1+x) per experiment first,
then averages (mean of log vs log of mean).

## Jon's Pipeline

```
Allen ISH raw CSV files (one per section_data_set_id)
  │  /mnt/data0/AllenBrainAtlas/ISH/{section_data_set_id}.csv
  │  Column: expression_energy (per structure)
  │
  ├── Step 0: Human→mouse ortholog mapping (HGNC)
  │   Maps human Entrez IDs → mouse Entrez IDs → Allen section_data_set_ids
  │   17,208 human genes with ISH data
  │
  ├── Step 1: For each gene, read all ISH section CSVs
  │   Extract expression_energy for each of 213 selected structures
  │
  ├── Step 2: Arithmetic mean of RAW expression energy across sections
  │   expression_energy-human-connectome_struc.csv (17,208 × 213)
  │   Formula: mean(expression_energy)  [NOT log-transformed]
  │
  ├── Step 3: log2(1+x) transform
  │   expression_energy-human-connectome_struc-log2.csv
  │   Verified: log2(1+raw) matches file exactly (max diff ~6e-15)
  │
  └── Step 4: Quantile normalization (R preprocessCore::normalize.quantiles)
      expression_energy-human-connectome_struc-log2-qn.csv
      → This is THE input for the legacy Z2 pipeline
```

## JW's Pipeline (for comparison)

```
Same Allen ISH raw CSV files
  │
  ├── Step 0: Human→mouse ortholog mapping (HGNC, different version)
  │   17,196 human genes
  │
  ├── Step 1: For each gene, read ISH CSVs
  │   Apply log2(1+expression_energy) PER SECTION
  │
  └── Step 2: Arithmetic mean of LOG-TRANSFORMED values
      ExpLevel.csv (17,196 × 213)
      Formula: mean(log2(1 + expression_energy))
```

## Key Difference: Aggregation Order

| Method | Formula | Used by |
|--------|---------|---------|
| Jon | `log2(1 + mean(raw))` | Legacy Z2 pipeline |
| JW | `mean(log2(1 + raw))` | Notebook 01 `ExpLevel.csv` |

By Jensen's inequality (log is concave): `mean(log(1+x)) ≤ log(1+mean(x))`,
so Jon's log2 values are systematically **higher** than JW's.

### Quantitative comparison (17,180 common genes × 213 structures)

| Metric | Value |
|--------|-------|
| Pearson r (log2 values) | 0.975 |
| Mean bias (Jon − JW) | +0.253 |
| MAE | 0.306 |
| Max abs diff | 4.06 |
| 90th percentile abs diff | 1.09 |

## Gene Set Differences

| Source | Genes | Notes |
|--------|------:|-------|
| Jon | 17,208 | Older HGNC ortholog mapping |
| JW | 17,196 | `human2mouse.0420.json` mapping |
| Jon only | 28 | Different ortholog version |
| JW only | 16 | Different ortholog version |
| Intersection | 17,180 | Used for legacy Z2 |

### Section Dataset IDs

For all 17,121 common mouse genes, section dataset IDs are **identical** between
Jon and JW. The gene set difference comes entirely from different human→mouse
ortholog mapping versions, not from different ISH experiment selections.

## Recovery Test (v1 — JW's fallback mapping, Feb 2026)

Attempted to reproduce Jon's raw expression matrix from ISH files using JW's
original ortholog mapping (Entrez-first fallback) + Jon's aggregation method:

| Metric | Value |
|--------|-------|
| Genes tested | 17,115 |
| Exact match (<1e-10) | 16,656 (97.3%) |
| Different (≥0.001) | 459 (2.7%) |

The 459 discrepant genes were caused by two bugs in the original notebook 01:
1. **Discontinued ID bug**: `GeneID` column in `gene_history.human.mouse.tsv`
   is string dtype (has `'-'` entries). The `isinstance(result, (int, float))`
   check silently rejected all lookups, losing 414 Allen sections across 253
   mouse genes.
2. **Fallback vs union**: Jon unions section IDs from both Entrez-based AND
   symbol-based routes. The original notebook only used symbol matching as a
   fallback for completely unlinked sections.

## Recovery Test (v2 — Jon's union strategy, Feb 2026)

After fixing both issues (notebook 01 rewrite), the recovery is near-perfect:

| Metric | Value |
|--------|-------|
| Our genes | 17,210 |
| Jon's genes | 17,208 |
| Common genes | 17,203 |
| **Exact match (<1e-10)** | **17,203/17,203 (100%)** |
| Jon-only genes | 5 (from older HGNC version) |
| Our-only genes | 7 |

The 5 irrecoverable genes (`7795, 100124696, 100187828, 100505381, 100652885`)
require Jon's specific `org.Hs.eg.db` R package version for human Entrez → symbol
mapping. Our HGNC `HOM_MouseHumanSequence.rpt` does not map them.

### QN and Z1 validation (same gene set)

| Step | Max diff | Notes |
|------|----------|-------|
| Raw expression | 4.97e-14 | Machine epsilon |
| Python QN vs R QN | 1.16e-13 | Bit-exact (same input) |
| Z1 | 3.46e-03 | Propagated from QN rounding |

Python `quantileNormalize_withNA` matches R `preprocessCore::normalize.quantiles`
perfectly when given identical input matrices.

## File Locations

### Current project (`ASD_Circuits_CellType`)
| File | Description |
|------|-------------|
| `dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv` | Jon's log2+QN (copied from legacy, THE Z2 input) |
| `dat/allen-mouse-exp/AllenMouse_z1_mat.0511.csv` | Legacy Z1 reference |
| `dat/allen-mouse-exp/ExpMatch_Legacy/` | Legacy match files (symlink, 17,189 files) |

### Legacy project (`/home/jw3514/Work/ASD_Circuits/`)
| File | Description |
|------|-------------|
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc.csv` | Jon's raw (arithmetic mean) |
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc-log2.csv` | Jon's log2(1+raw) |
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc-log2-qn.csv` | Jon's log2+QN (R) |
| `dat/allen-mouse-exp/ExpLevel.csv` | JW's log2 expression (mean of log) |
| `dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv` | Allen gene→section mapping |

### Raw ISH data
| Path | Description |
|------|-------------|
| `/mnt/data0/AllenBrainAtlas/ISH/` | ~26,000 ISH CSV files (one per section_data_set_id) |

## Created

2026-02-23.
