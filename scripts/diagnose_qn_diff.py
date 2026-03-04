#!/usr/bin/env python3
"""Diagnose the exact source of Python vs R QN difference.

The bug: Python's quantileNormalize_withNA uses df.rank(method='average')
which gives fractional ranks for ties (e.g., 1.5), then does:
    rank_indices = (col_ranks - 1).astype(int).values
This TRUNCATES fractional ranks (1.5 → 1), so tied values get a single
reference quantile rather than interpolating. R's normalize.quantiles
uses a C implementation that handles this correctly.
"""
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits_CellType/src')
import numpy as np
import pandas as pd

# === Small example ===
print("=== Small Example: 5 genes x 3 structures ===")
data = pd.DataFrame({
    'A': [1.0, 1.0, 2.0, 3.0, 4.0],  # Ties: rows 0,1 both have 1.0
    'B': [2.0, 4.0, 1.0, 3.0, 5.0],
    'C': [3.0, 5.0, 4.0, 1.0, 2.0],
}, index=['g1', 'g2', 'g3', 'g4', 'g5'])

print("Input:")
print(data)
print()

# Step 1: Rank each column (average method)
ranks = data.rank(method='average')
print("Ranks (method='average'):")
print(ranks)
print()

# Step 2: Sort each column to get reference quantiles
sorted_vals = {}
for col in data.columns:
    sorted_vals[col] = np.sort(data[col].values)
sorted_matrix = np.column_stack([sorted_vals[c] for c in data.columns])
print("Sorted values (reference quantiles):")
print(pd.DataFrame(sorted_matrix, columns=data.columns,
                   index=[f'rank_{i+1}' for i in range(5)]))
print()

# Step 3: Row means of sorted matrix = target quantiles
mean_vals = sorted_matrix.mean(axis=1)
print("Target quantiles (row means of sorted):")
for i, v in enumerate(mean_vals):
    print(f"  rank {i+1}: {v:.4f}")
print()

# Step 4: Python's approach — truncate fractional ranks
print("Python's approach (truncate fractional ranks):")
for col in data.columns:
    col_ranks = ranks[col]
    rank_indices = (col_ranks - 1).astype(int).values  # THE BUG
    normalized = mean_vals[rank_indices]
    print(f"  {col}: ranks={col_ranks.values} → indices={rank_indices} → values={normalized}")
print()

# Step 5: Correct approach — interpolate for fractional ranks
print("Correct approach (interpolate for fractional ranks):")
for col in data.columns:
    col_ranks = ranks[col]
    normalized = []
    for r in col_ranks:
        if r == int(r):
            # Integer rank — just look up
            normalized.append(mean_vals[int(r) - 1])
        else:
            # Fractional rank — interpolate
            lo = int(np.floor(r)) - 1
            hi = int(np.ceil(r)) - 1
            frac = r - np.floor(r)
            val = mean_vals[lo] * (1 - frac) + mean_vals[hi] * frac
            normalized.append(val)
    print(f"  {col}: ranks={col_ranks.values} → values={normalized}")
print()

# Step 6: R's actual approach — average the target quantile values for tied inputs
print("R's approach (ties get AVERAGE of their target quantiles):")
print("In R, normalize.quantiles ranks with ties broken by row order (method='first'),")
print("then assigns each input the mean of the quantiles for its tie group.")
for col in data.columns:
    col_ranks_first = data[col].rank(method='first')
    rank_indices = (col_ranks_first - 1).astype(int).values
    # Each input gets its unique quantile based on 'first' ranking
    raw_quantiles = mean_vals[rank_indices]
    # Then ties get averaged
    final = np.zeros(len(data))
    for val in data[col].unique():
        mask = data[col] == val
        final[mask] = raw_quantiles[mask].mean()
    print(f"  {col}: first_ranks={col_ranks_first.values} → raw_q={raw_quantiles} → avg_tied={final}")
print()

print("=" * 60)
print("KEY INSIGHT:")
print("  Python rank(method='average') rank=1.5 for tied values.")
print("  Then (1.5 - 1).astype(int) = 0, mapping to quantile[0].")
print("  So tied values get quantile at the LOWER rank, not the average.")
print()
print("  R's normalize.quantiles breaks ties by row order (method='first'),")
print("  assigns each to its distinct quantile position, then averages")
print("  the quantile values across the tie group.")
print()
print("  With many ties (e.g., 100+ genes with value=0 in a column),")
print("  Python assigns them ALL to quantile[49] (if avg rank=50),")
print("  while R would assign the mean of quantiles[0..99].")
print("  This can create large differences.")
print()

# === Real data ===
print("=== Real Data Impact ===")
raw = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.raw.csv', index_col='ROW')
log2_py = np.log2(raw + 1)
qn_r = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv', index_col='ROW')

# Check a column with many ties
col = log2_py.columns[0]
vals = log2_py[col].dropna()
ties_tab = vals.value_counts()
biggest_tie = ties_tab.max()
biggest_tie_val = ties_tab.idxmax()
n_ties = (ties_tab > 1).sum()

print(f"\nColumn '{col}': {len(vals)} genes")
print(f"  {n_ties} tied groups, largest: {biggest_tie} genes at value={biggest_tie_val:.4f}")

# Show how ranks differ
tied_mask = vals == biggest_tie_val
tied_genes = vals[tied_mask].index
print(f"\n  For {biggest_tie} genes tied at {biggest_tie_val:.4f}:")

avg_ranks = vals.rank(method='average')
first_ranks = vals.rank(method='first')
print(f"    Average ranks: {avg_ranks[tied_genes].unique()} (all same)")
print(f"    First ranks: {first_ranks[tied_genes].min():.0f} to {first_ranks[tied_genes].max():.0f}")

# Compute what Python assigns vs what R assigns
# Sort all non-NA values in each column, compute row means
sorted_all = {}
for c in log2_py.columns:
    v = log2_py[c].dropna().values
    v.sort()
    sorted_all[c] = v

# Handle different column lengths (some columns may have NAs)
max_len = max(len(v) for v in sorted_all.values())
sorted_matrix = np.full((max_len, len(log2_py.columns)), np.nan)
for i, c in enumerate(log2_py.columns):
    v = sorted_all[c]
    sorted_matrix[:len(v), i] = v
mean_quantiles = np.nanmean(sorted_matrix, axis=1)

# Python: truncated index
avg_rank = avg_ranks[tied_genes].iloc[0]
py_index = int(avg_rank - 1)
py_value = mean_quantiles[py_index]

# R: average of the quantile values for ranks [first_min..first_max]
first_min = int(first_ranks[tied_genes].min()) - 1
first_max = int(first_ranks[tied_genes].max()) - 1
r_value = mean_quantiles[first_min:first_max+1].mean()

print(f"\n    Python: avg_rank={avg_rank:.1f} → index={py_index} → quantile={py_value:.6f}")
print(f"    R: first_ranks [{first_min+1}..{first_max+1}] → mean(quantiles[{first_min}..{first_max}]) = {r_value:.6f}")
print(f"    R actual (from file): {qn_r.loc[tied_genes, col].mean():.6f}")
print(f"    Difference: {abs(py_value - r_value):.6f}")

# Check how close our R prediction is to R actual
r_actual_vals = qn_r.loc[tied_genes, col]
print(f"\n    R actual: all same? {r_actual_vals.nunique() == 1}")
print(f"    R actual value: {r_actual_vals.iloc[0]:.6f}")
print(f"    Our R prediction: {r_value:.6f}")
print(f"    Prediction error: {abs(r_actual_vals.iloc[0] - r_value):.8f}")

# Overall: count tied values and estimate total impact
print("\n=== Summary Statistics ===")
total_tied = 0
total_values = 0
for c in log2_py.columns:
    v = log2_py[c].dropna()
    tab = v.value_counts()
    total_tied += (tab[tab > 1]).sum()
    total_values += len(v)
print(f"Total values: {total_values}")
print(f"Values in tie groups: {total_tied} ({total_tied/total_values*100:.1f}%)")
print(f"Values NOT tied: {total_values - total_tied} ({(total_values - total_tied)/total_values*100:.1f}%)")
print(f"\nFor non-tied values, Python and R should agree exactly.")
print(f"For tied values, Python truncates the fractional rank to an integer,")
print(f"while R averages the quantile values across the tie group.")
