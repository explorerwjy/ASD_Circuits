# Permutation Bias Comparison Script - Examples

This script (`script_permutation_bias_comparison.py`) performs permutation-based statistical testing to compare bias patterns between two gene sets.

## Overview

The analysis pipeline:
1. Loads two gene weight dictionaries (e.g., ASD vs DDD genes)
2. Calculates bias for each gene set using expression data
3. Fits a linear regression model to control for shared signal
4. Runs permutations by shuffling gene labels while preserving weights
5. Calculates p-values for bias differences, rank differences, and residuals

## Quick Start

### Example 1: Compare ASD vs DDD on all cell types

```bash
python scripts/script_permutation_bias_comparison.py \
    --gw1 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \
    --gw2 dat/Genetics/GeneWeights_DN/DDD.top293.ExcludeASD.DN.gw \
    --expr_mat dat/BiasMatrices/MouseCT.TPM.1.Filt.Spec.clip.lowexp.100000.qn.parquet \
    --cluster_ann dat/MouseCT_Cluster_Anno.csv \
    --n_perms 10000 \
    --output results/permutation_ASD_vs_DDD_all_celltypes.csv \
    --label1 ASD \
    --label2 DDD \
    --seed 42
```

### Example 2: Compare ASD vs DDD on specific cell type class (CNU-LGE GABA neurons)

```bash
python scripts/script_permutation_bias_comparison.py \
    --gw1 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \
    --gw2 dat/Genetics/GeneWeights_DN/DDD.top293.ExcludeASD.DN.gw \
    --expr_mat dat/BiasMatrices/MouseCT.TPM.1.Filt.Spec.clip.lowexp.100000.qn.parquet \
    --cluster_ann dat/MouseCT_Cluster_Anno.csv \
    --filter_class "09 CNU-LGE GABA" \
    --n_perms 10000 \
    --output results/permutation_ASD_vs_DDD_CNU_LGE.csv \
    --label1 ASD \
    --label2 DDD \
    --seed 42
```

### Example 3: Compare constrained genes vs ASD genes

```bash
python scripts/script_permutation_bias_comparison.py \
    --gw1 dat/Genetics/GeneWeights_DN/constraint_top_decile_LOEUF_excludeASD.DN.gw \
    --gw2 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \
    --expr_mat dat/BiasMatrices/MouseCT.TPM.1.Filt.Spec.clip.lowexp.100000.qn.parquet \
    --cluster_ann dat/MouseCT_Cluster_Anno.csv \
    --n_perms 10000 \
    --output results/permutation_Constraint_vs_ASD.csv \
    --label1 Constraint \
    --label2 ASD \
    --seed 42
```

### Example 4: Quick test with fewer permutations

```bash
python scripts/script_permutation_bias_comparison.py \
    --gw1 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \
    --gw2 dat/Genetics/GeneWeights_DN/DDD.top293.ExcludeASD.DN.gw \
    --expr_mat dat/BiasMatrices/MouseCT.TPM.1.Filt.Spec.clip.lowexp.100000.qn.parquet \
    --cluster_ann dat/MouseCT_Cluster_Anno.csv \
    --n_perms 1000 \
    --output results/permutation_test_quick.csv \
    --label1 ASD \
    --label2 DDD \
    --seed 42
```

### Example 5: Use structural ISH data instead of single-cell data

```bash
python scripts/script_permutation_bias_comparison.py \
    --gw1 dat/Genetics/GeneWeights_DN/Spark_Meta_EWS.GeneWeight.DN.gw \
    --gw2 dat/Genetics/GeneWeights_DN/DDD.top293.ExcludeASD.DN.gw \
    --expr_mat dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet \
    --n_perms 10000 \
    --output results/permutation_ASD_vs_DDD_structures.csv \
    --label1 ASD \
    --label2 DDD \
    --seed 42
```

## Command-line Arguments

### Required Arguments
- `--gw1`: Path to first gene weight file (.gw format)
- `--gw2`: Path to second gene weight file (.gw format)
- `--expr_mat`: Path to expression matrix (CSV, CSV.GZ, or Parquet)
- `--output`: Output file path for results (CSV format)

### Optional Arguments
- `--cluster_ann`: Path to cluster annotation file (adds class labels to results)
- `--filter_class`: Filter analysis to specific cell type class (e.g., "09 CNU-LGE GABA")
- `--n_perms`: Number of permutations (default: 10000)
- `--seed`: Random seed for reproducibility
- `--label1`: Label for first gene set (default: "GeneSet1")
- `--label2`: Label for second gene set (default: "GeneSet2")
- `--n_processes`: Number of parallel processes (default: 1, currently not implemented)

## Output Format

The output CSV file contains the following columns:

### Observed Statistics
- `EFFECT_[label1]`: Bias score for gene set 1
- `Rank_[label1]`: Rank for gene set 1
- `EFFECT_[label2]`: Bias score for gene set 2
- `Rank_[label2]`: Rank for gene set 2
- `DIFF_EFFECT`: Difference in bias scores
- `ABS_DIFF_EFFECT`: Absolute difference in bias scores
- `DIFF_Rank`: Difference in ranks
- `ABS_DIFF_Rank`: Absolute difference in ranks
- `predicted`: Predicted bias for gene set 1 based on linear model
- `residual`: Residual (observed - predicted) for gene set 1

### P-values (from permutation testing)
- `pval_residual`: P-value for residual (main metric of interest)
- `pval_DIFF_EFFECT`: P-value for difference in bias
- `pval_DIFF_Rank`: P-value for difference in ranks

### Additional columns (if cluster_ann provided)
- `class_id_label`: Cell type class label
- `subclass_id_label`: Cell type subclass label
- Other annotation columns

## Interpretation

The **residual** represents the bias in gene set 1 after accounting for shared signal with gene set 2 via linear regression. Significant positive residuals indicate cell types/structures that are preferentially affected by gene set 1 compared to gene set 2.

- **pval_residual < 0.05**: Cell type shows significantly different bias pattern between the two gene sets
- **Positive residual**: Gene set 1 shows higher bias than expected based on gene set 2
- **Negative residual**: Gene set 1 shows lower bias than expected based on gene set 2

## Tips

1. **Number of permutations**: Use 10,000 permutations for publication-quality results. Use 1,000 for quick testing.
2. **Random seed**: Always set a seed for reproducibility
3. **Runtime**: With 10,000 permutations on ~3,000 cell types, expect ~10-30 minutes runtime
4. **Memory**: Large expression matrices may require significant RAM. Consider filtering to specific classes if needed.
5. **Multiple testing correction**: Apply FDR correction (e.g., Benjamini-Hochberg) to p-values when testing many cell types

## Example Analysis Workflow

```python
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Load results
results = pd.read_csv('results/permutation_ASD_vs_DDD_all_celltypes.csv', index_col=0)

# Apply FDR correction
results['pval_residual_fdr'] = multipletests(results['pval_residual'], method='fdr_bh')[1]

# Filter to significant results
sig_results = results[results['pval_residual_fdr'] < 0.05]

# Sort by residual to find top cell types
top_positive = sig_results.nlargest(10, 'residual')
top_negative = sig_results.nsmallest(10, 'residual')

print("Cell types with highest ASD-specific bias:")
print(top_positive[['EFFECT_ASD', 'EFFECT_DDD', 'residual', 'pval_residual_fdr']])
```
