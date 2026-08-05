# Legacy Z2 Pipeline: ISH Expression → Z2 Bias Matrix

## Summary

The legacy Z2 matrix (`AllenMouseBrain_Z2bias.csv`, 17,180 genes x 213 structures) was
produced by a pipeline that used **Jon's R-generated log2+QN expression matrix** as input,
not the JW `ExpLevel.csv`. This document traces the exact pipeline and identifies key
dependencies for reproducibility.

## Verified Pipeline (produces exact Z2 match)

```
Jon's raw ISH expression energy (17,208 genes x 213 structures)
    │  expression_energy-human-connectome_struc.csv
    │
    ├── Step 1: log2(1+x) transform
    │   expression_energy-human-connectome_struc-log2.csv
    │   (verified: log2(1+raw) matches log2 file, r=1.0, 100% within 1e-10)
    │
    ├── Step 2: Quantile normalization (R, NOT Python)
    │   expression_energy-human-connectome_struc-log2-qn.csv
    │   (R's normalize.quantiles from preprocessCore package)
    │
    ├── Step 3: Filter to genes in JW's ExpLevel.csv (17,196 genes)
    │   Intersection: 17,180 genes
    │
    ├── Step 4: Z1 (per-gene z-score via ZscoreConverting)
    │   AllenMouse_z1_mat.0511.csv (17,180 genes x 213 structures)
    │
    ├── Step 5: Expression matching (±5% quantile, 10K samples w/ replacement)
    │   ExpMatch_RootExp_uniform_kernal/ (17,189 match files)
    │
    └── Step 6: Z2 (expression-matched z-score)
        AllenMouseBrain_Z2bias.csv (17,180 genes x 213 structures)
```

## Reproduction Results

Using `AllenMouse_z1_mat.0511.csv` + legacy match files → Z2:
- **r = 0.9999991**, MAE = 0.0000515
- 99.66% of values within 1e-6 of legacy Z2
- 55.2% bit-exact

The small residual (~0.34%) comes from genes whose match files reference genes
present in the match file set (17,189) but absent from the Z1 (17,180).

## Key Files (Legacy Project)

All paths relative to `/home/jw3514/Work/ASD_Circuits/`:

| File | Genes | Description |
|------|------:|-------------|
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc.csv` | 17,208 | Jon's raw ISH expression energy (indexed by `ROW`) |
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc-log2.csv` | 17,208 | log2(1+x) of above |
| `scripts/dat/Jon_data/expression_energy-human-connectome_struc-log2-qn.csv` | 17,208 | **THE INPUT**: R quantile-normalized log2 expression |
| `dat/allen-mouse-exp/ExpLevel.csv` | 17,196 | JW's log2(1+x) expression (used only for gene filtering) |
| `dat/allen-mouse-exp/AllenMouse_z1_mat.0511.csv` | 17,180 | Z1 from Jon's QN data ∩ JW genes |
| `dat/genes/Match_Use_Root_Exp_Energy.csv` | 17,180 | Expression matching feature table |
| `dat/genes/ExpMatch_RootExp_uniform_kernal/` | 17,189 files | Per-gene expression match lists (10K samples each) |
| `dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv` | 17,180 | **THE OUTPUT**: Legacy Z2 matrix |

## Jon vs JW Raw Expression Data

Jon and JW independently processed Allen ISH data. The two key differences are:

### 1. Aggregation Order (major difference)

Both read per-section ISH expression energy CSVs and average across multiple
ISH experiments per gene, but in **different order**:

- **Jon**: `log2(1 + mean(raw_expression_energy))` — arithmetic mean first, then log2
- **JW**: `mean(log2(1 + raw_expression_energy))` — log2 first, then mean

By Jensen's inequality (log is concave): `mean(log(x)) ≤ log(mean(x))`,
so Jon's values are systematically higher.

Comparison on 17,180 common genes × 213 structures:
- r = 0.975 between log2 expression values
- Mean bias (Jon − JW): +0.253
- MAE: 0.306, max diff: 4.06
- 90th percentile abs diff: 1.09

### 2. Human→Mouse Ortholog Mapping (minor difference)

Both used HGNC-based human→mouse ortholog mappings, but from different versions:
- Jon: 17,208 human genes
- JW: 17,196 human genes (from `human2mouse.0420.json`)
- 28 genes in Jon only, 16 genes in JW only

For common mouse genes (17,121), **section dataset IDs are identical**. The gene
set difference comes entirely from different ortholog mapping versions.

### Recovery Test

Using JW's ortholog mapping + Jon's aggregation method (arithmetic mean of raw)
on all common genes:
- **97.3% exact match** (16,656/17,115 genes) with Jon's raw expression
- 2.7% differ (459 genes) due to different mouse ortholog assignments
- 0% differ in section dataset IDs for shared orthologs

## Quantile Normalization Difference

The legacy Z2 used **R's quantile normalization** (likely `preprocessCore::normalize.quantiles`).
Neither Python implementation matches:

| Implementation | vs Jon's log2-qn file |
|---------------|----------------------|
| R (Jon's original) | exact (by definition) |
| Python `quantileNormalize` (utils.py, no NaN handling) | r = 0.993, 0 exact |
| Python `quantileNormalize_withNA` (ASD_Circuits.py) | r = 0.993, 0 exact |

The difference is in tie-breaking: R's `normalize.quantiles` uses a different
algorithm for handling tied values than Python's rank-based approaches.

## Other Z1 Files (Not Used for Z2)

| File | Genes | Notes |
|------|------:|-------|
| `JW_Z1-Mat.ArithmeticMean.0422.csv` | 17,115 | Older Z1, different pipeline (not used for Z2) |
| `energy-zscore-conn-model.0524.csv` | 15,681 | Connectivity-model Z1 (smaller gene set) |

## Gene Count Summary

| Step | Genes | Source |
|------|------:|--------|
| Jon's expression matrix | 17,208 | Jon's ISH processing (R) |
| JW's ExpLevel.csv | 17,196 | JW's ISH processing (Python, notebook 01) |
| Intersection | 17,180 | `set(Jon) & set(JW)` |
| Expression match files | 17,189 | Built from ExpLevel features (pre-filtering) |
| Legacy Z1 (`0511`) | 17,180 | Jon QN data filtered to intersection |
| Legacy Z2 | 17,180 | Z2 from Z1 + match files |
| New Z2 (notebook 02, current) | 17,198 | JW ExpLevel → Z1 → new matches |

## Notebook 02 (Current Pipeline)

Notebook 02 now uses Jon's `log2-qn` file as input with legacy match files,
reproducing the legacy Z2 exactly:
- Z1: r = 1.0, 100% bit-exact vs legacy
- Z2: r = 0.9999991, 99.7% within 1e-6, ASD bias bit-exact
- Match files use all 10K entries (with duplicates) for frequency-weighted mean/std

The script `scripts/script_compute_Z2.py` preserves duplicate match entries when
using pre-computed match files, matching the legacy `script_Z2_calculation.py`
behavior.

## Created

2026-02-23, traced via systematic comparison of pipeline permutations.
Updated 2026-02-23: added aggregation method analysis and recovery test results.
