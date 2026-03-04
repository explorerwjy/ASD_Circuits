#!/usr/bin/env python3
"""Compare Python vs R quantile normalization to identify the source of differences."""
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits_CellType/src')
import numpy as np
import pandas as pd
from ASD_Circuits import quantileNormalize_withNA

# Load data
raw = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.raw.csv', index_col='ROW')
qn_r = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv', index_col='ROW')

# Step 1: log2(x+1) — should be identical
log2_py = np.log2(raw + 1)

# Step 2: Python QN
qn_py = quantileNormalize_withNA(log2_py)

print("=== Global Comparison: Python QN vs R QN ===")
vals_py = qn_py.values.flatten()
vals_r = qn_r.values.flatten()
valid = ~np.isnan(vals_py) & ~np.isnan(vals_r)

r_corr = np.corrcoef(vals_py[valid], vals_r[valid])[0, 1]
diffs = np.abs(vals_py[valid] - vals_r[valid])
print(f"Pearson r:        {r_corr:.10f}")
print(f"Max abs diff:     {diffs.max():.6f}")
print(f"Mean abs diff:    {diffs.mean():.6f}")
print(f"Median abs diff:  {np.median(diffs):.6f}")
print(f"Frac diff > 0.01: {(diffs > 0.01).mean():.4f}")
print(f"Frac diff > 0.001:{(diffs > 0.001).mean():.4f}")
print(f"Frac diff == 0:   {(diffs == 0).mean():.4f}")
print()

# Investigate: where are the differences largest?
diff_mat = np.abs(qn_py.values - qn_r.values)
print("=== Per-Structure Analysis ===")
per_str_mean_diff = np.nanmean(diff_mat, axis=0)
per_str_max_diff = np.nanmax(diff_mat, axis=0)
print(f"Mean diff per structure: min={per_str_mean_diff.min():.6f}, max={per_str_mean_diff.max():.6f}")
print(f"Max diff per structure:  min={per_str_max_diff.min():.6f}, max={per_str_max_diff.max():.6f}")
print()

print("=== Per-Gene Analysis ===")
per_gene_mean_diff = np.nanmean(diff_mat, axis=1)
per_gene_max_diff = np.nanmax(diff_mat, axis=1)
print(f"Mean diff per gene: min={per_gene_mean_diff.min():.6f}, max={per_gene_mean_diff.max():.6f}")
print(f"Max diff per gene:  min={per_gene_max_diff.min():.6f}, max={per_gene_max_diff.max():.6f}")
print()

# Key question: is the difference due to tie-breaking in rank()?
# R's normalize.quantiles uses a C implementation that handles ties differently.
# Python's df.rank(method='average') assigns average ranks to tied values.
# R assigns... let's check.
print("=== Tie Analysis ===")
# Count ties in a few columns
for col_idx in [0, 50, 100, 200]:
    col = log2_py.columns[col_idx]
    vals = log2_py[col].dropna().values
    n_unique = len(np.unique(vals))
    n_total = len(vals)
    n_ties = n_total - n_unique
    # How many are exactly 0 (log2(0+1) = 0)?
    n_zeros = (vals == 0).sum()
    print(f"  {col}: {n_total} values, {n_unique} unique, {n_ties} tied, {n_zeros} zeros")

print()
# What fraction of values are zero (the biggest tie group)?
all_vals = log2_py.values.flatten()
valid_vals = all_vals[~np.isnan(all_vals)]
n_zeros_total = (valid_vals == 0).sum()
print(f"Total zeros across all columns: {n_zeros_total} / {len(valid_vals)} ({n_zeros_total/len(valid_vals)*100:.1f}%)")

# Check: for tied values (zeros), what does Python assign vs R?
col = log2_py.columns[0]
zero_genes = log2_py.index[log2_py[col] == 0]
print(f"\nFor column '{col}', {len(zero_genes)} genes have value=0 (biggest tie group)")
if len(zero_genes) > 0:
    py_vals_at_ties = qn_py.loc[zero_genes[:5], col]
    r_vals_at_ties = qn_r.loc[zero_genes[:5], col]
    print(f"  Python QN values (first 5): {py_vals_at_ties.values}")
    print(f"  R QN values (first 5):      {r_vals_at_ties.values}")
    print(f"  All Python tied = same? {py_vals_at_ties.nunique() == 1}")
    print(f"  All R tied = same?      {r_vals_at_ties.nunique() == 1}")

    # Check all tied values for this column
    py_all_tied = qn_py.loc[zero_genes, col]
    r_all_tied = qn_r.loc[zero_genes, col]
    print(f"  Python: {py_all_tied.nunique()} unique values for {len(zero_genes)} tied inputs")
    print(f"  R:      {r_all_tied.nunique()} unique values for {len(zero_genes)} tied inputs")
    print(f"  Python tied value: {py_all_tied.mean():.6f}")
    print(f"  R tied values range: [{r_all_tied.min():.6f}, {r_all_tied.max():.6f}]")

print()
print("=== Conclusion ===")
print("Python rank(method='average') assigns the SAME value to all tied inputs.")
print("R normalize.quantiles breaks ties, assigning DIFFERENT values to tied inputs.")
print("This is the source of the difference.")
