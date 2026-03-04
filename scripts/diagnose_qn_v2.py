#!/usr/bin/env python3
"""Deeper diagnosis: compare reference distributions and rank mappings."""
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits_CellType/src')
import numpy as np
import pandas as pd

raw = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.raw.csv', index_col='ROW')
qn_r = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv', index_col='ROW')
log2_py = np.log2(raw + 1)

# === Step 1: Are the reference distributions the same? ===
# After QN, sorting any column of the output gives the reference distribution.
# R's reference = sort any column of R's QN output
# Python's reference = row means of sorted input matrix

print("=== Reference Distribution Comparison ===")

# R's reference (from first column with no NAs)
col0 = qn_r.columns[0]
r_ref = np.sort(qn_r[col0].dropna().values)
print(f"R reference (from sorted col '{col0}'): {len(r_ref)} values, range [{r_ref[0]:.6f}, {r_ref[-1]:.6f}]")

# Python's reference
sorted_all = {}
col_lengths = {}
for col in log2_py.columns:
    v = log2_py[col].dropna().values
    v.sort()
    sorted_all[col] = v
    col_lengths[col] = len(v)

max_len = max(col_lengths.values())
min_len = min(col_lengths.values())
print(f"Column lengths: min={min_len}, max={max_len}, unique={len(set(col_lengths.values()))}")

sorted_matrix = np.full((max_len, len(log2_py.columns)), np.nan)
for i, col in enumerate(log2_py.columns):
    v = sorted_all[col]
    sorted_matrix[:len(v), i] = v

py_ref = np.nanmean(sorted_matrix, axis=1)
print(f"Python reference: {len(py_ref)} values, range [{py_ref[0]:.6f}, {py_ref[-1]:.6f}]")

# Compare first N values (where both have same length)
n_compare = min(len(r_ref), len(py_ref))
ref_diff = np.abs(r_ref[:n_compare] - py_ref[:n_compare])
print(f"\nReference distribution comparison (first {n_compare} values):")
print(f"  Max diff:  {ref_diff.max():.10f}")
print(f"  Mean diff: {ref_diff.mean():.10f}")
print(f"  Exact matches: {(ref_diff == 0).sum()} / {n_compare}")
print(f"  Diff > 1e-6: {(ref_diff > 1e-6).sum()}")

# Check R's reference consistency: are all columns the same after QN?
print("\n=== R QN output: column distribution consistency ===")
r_sorted_cols = []
for col in qn_r.columns[:5]:
    s = np.sort(qn_r[col].dropna().values)
    r_sorted_cols.append(s)

for i in range(1, len(r_sorted_cols)):
    n = min(len(r_sorted_cols[0]), len(r_sorted_cols[i]))
    d = np.abs(r_sorted_cols[0][:n] - r_sorted_cols[i][:n])
    print(f"  Col 0 vs col {i}: max diff = {d.max():.10f}")

# === Step 2: Check if column lengths vary ===
print(f"\n=== NA pattern ===")
na_counts = log2_py.isna().sum()
print(f"NAs per column: min={na_counts.min()}, max={na_counts.max()}, mean={na_counts.mean():.1f}")
print(f"Columns with 0 NAs: {(na_counts == 0).sum()} / {len(na_counts)}")

# If columns have different lengths, the row mean at position i uses
# different numbers of columns. This means the reference distribution
# differs depending on which column you check.
# Let's verify:
if na_counts.max() > 0:
    # Columns with all values
    full_cols = na_counts[na_counts == 0].index
    # Columns with some NAs
    na_cols = na_counts[na_counts > 0].index
    print(f"Full columns: {len(full_cols)}, columns with NAs: {len(na_cols)}")

    # R's sorted output for a full column vs an NA column
    if len(na_cols) > 0:
        col_full = full_cols[0]
        col_na = na_cols[0]
        r_full = np.sort(qn_r[col_full].dropna().values)
        r_na = np.sort(qn_r[col_na].dropna().values)
        print(f"  Full col '{col_full}': {len(r_full)} sorted values")
        print(f"  NA col '{col_na}': {len(r_na)} sorted values (has {na_counts[col_na]} NAs)")

        n = min(len(r_full), len(r_na))
        d = np.abs(r_full[:n] - r_na[:n])
        print(f"  Overlap comparison: max diff = {d.max():.6f}")
        print(f"  This means columns with different NA counts get DIFFERENT reference distributions!")

# === Step 3: Detailed single-column comparison ===
print(f"\n=== Single Column Deep Dive ===")
col = log2_py.columns[0]
n_valid = col_lengths[col]
print(f"Column '{col}': {n_valid} valid values")

# Python: rank -> map to reference
py_sorted = sorted_all[col]
py_ranks_first = log2_py[col].dropna().rank(method='first').astype(int)

# What Python assigns at each position
py_assigned = py_ref[py_ranks_first.values - 1]

# What R assigns (from the QN output)
r_assigned = qn_r.loc[py_ranks_first.index, col].values

# Compare position by position
pos_diff = np.abs(py_assigned - r_assigned)
print(f"  Position-wise comparison:")
print(f"    Max diff:  {pos_diff.max():.6f}")
print(f"    Mean diff: {pos_diff.mean():.6f}")
print(f"    Exact:     {(pos_diff == 0).sum()} / {len(pos_diff)}")

# What if we use R's reference instead?
# R's reference for this column
r_ref_col = np.sort(qn_r[col].dropna().values)
py_with_r_ref = r_ref_col[py_ranks_first.values - 1]
ref_swap_diff = np.abs(py_with_r_ref - r_assigned)
print(f"\n  Using R's reference distribution instead of Python's:")
print(f"    Max diff:  {ref_swap_diff.max():.6f}")
print(f"    Mean diff: {ref_swap_diff.mean():.6f}")
print(f"    Exact:     {(ref_swap_diff == 0).sum()} / {len(ref_swap_diff)}")

# So how different are the two reference distributions for this column?
ref_compare = np.abs(py_ref[:n_valid] - r_ref_col)
print(f"\n  Reference distribution difference for this column:")
print(f"    Max:  {ref_compare.max():.6f}")
print(f"    Mean: {ref_compare.mean():.6f}")

# === Step 4: The crux — does R normalize.quantiles use row means? ===
# Or does it do something else with the sorted values?
print(f"\n=== Does R use simple row means? ===")
# If R uses row means of the sorted matrix, then sorting any column
# of R's output should give those row means. Let's check.
# Compute Python's row means for the full sorted matrix
print(f"Python row mean at position 0: {py_ref[0]:.10f}")
print(f"R sorted output at position 0: {r_ref_col[0]:.10f}")
print(f"Difference: {abs(py_ref[0] - r_ref_col[0]):.10f}")
print()
print(f"Python row mean at position 1000: {py_ref[1000]:.10f}")
print(f"R sorted output at position 1000: {r_ref_col[1000]:.10f}")
print(f"Difference: {abs(py_ref[1000] - r_ref_col[1000]):.10f}")
print()
print(f"Python row mean at position -1: {py_ref[n_valid-1]:.10f}")
print(f"R sorted output at position -1: {r_ref_col[-1]:.10f}")
print(f"Difference: {abs(py_ref[n_valid-1] - r_ref_col[-1]):.10f}")
