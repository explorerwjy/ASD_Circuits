#!/usr/bin/env python3
"""Validate the fixed Python QN against R's output."""
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits_CellType/src')
import numpy as np
import pandas as pd
from ASD_Circuits import quantileNormalize_withNA

# Load data
raw = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.raw.csv', index_col='ROW')
qn_r = pd.read_csv('dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv', index_col='ROW')

# log2(x+1) then QN with fixed Python
log2_py = np.log2(raw + 1)
print(f"Input: {log2_py.shape[0]} genes x {log2_py.shape[1]} structures")
print("Running fixed Python QN...")
qn_py = quantileNormalize_withNA(log2_py)

# Compare
vals_py = qn_py.values.flatten()
vals_r = qn_r.values.flatten()
valid = ~np.isnan(vals_py) & ~np.isnan(vals_r)

diffs = np.abs(vals_py[valid] - vals_r[valid])
r_corr = np.corrcoef(vals_py[valid], vals_r[valid])[0, 1]

print(f"\n=== Fixed Python QN vs R QN ===")
print(f"Pearson r:         {r_corr:.10f}")
print(f"Max abs diff:      {diffs.max():.8f}")
print(f"Mean abs diff:     {diffs.mean():.8f}")
print(f"Median abs diff:   {np.median(diffs):.8f}")
print(f"Frac diff > 0.01:  {(diffs > 0.01).mean():.6f}")
print(f"Frac diff > 0.001: {(diffs > 0.001).mean():.6f}")
print(f"Frac diff > 1e-4:  {(diffs > 1e-4).mean():.6f}")
print(f"Frac diff == 0:    {(diffs == 0).mean():.6f}")

# Check tied values specifically
col = log2_py.columns[0]
vals = log2_py[col].dropna()
ties_tab = vals.value_counts()
biggest_tie_val = ties_tab.idxmax()
tied_genes = vals[vals == biggest_tie_val].index

print(f"\n=== Tie group check (column '{col}', {len(tied_genes)} genes tied at {biggest_tie_val:.4f}) ===")
py_tied = qn_py.loc[tied_genes, col]
r_tied = qn_r.loc[tied_genes, col]
print(f"Python tied values: all same? {py_tied.nunique() == 1}, value={py_tied.iloc[0]:.8f}")
print(f"R tied values:      all same? {r_tied.nunique() == 1}, value={r_tied.iloc[0]:.8f}")
print(f"Difference:         {abs(py_tied.iloc[0] - r_tied.iloc[0]):.8f}")
