"""
Aggregate sibling null circuit search results.

Processes pareto fronts from all sibling iterations and computes:
1. 95% confidence intervals for circuit scores at each bias limit
2. Real sibling bias (recalculated from parquet, replacing transferred bias)
3. NPZ-compatible arrays for notebook plotting:
   - meanbias: mean REAL sibling bias at each Pareto point
   - meanSI: mean SI score at each Pareto point
   - topbias_sub: subsampled profiles for plotting

Note on "bias transfer": The SA search uses transferred bias values (aligned
to the reference dataset by rank) for constraint evaluation.  The pareto front
CSVs contain these transferred values in mean_bias.  This script recalculates
the REAL sibling bias for each circuit using the original parquet data and the
structure lists from the pareto fronts.

Snakemake inputs:
    input.sibling_paretos: List of sibling pareto front CSV files
    input.main_pareto: Main dataset pareto front (if included)

Snakemake outputs:
    output.summary_csv: Summary statistics
    output.ci_by_bias: Confidence intervals by bias limit
    output.profiles_npz: NPZ with profile arrays for notebooks
    output.complete: Completion marker

Snakemake params:
    params.size: Circuit size
    params.ci_level: Confidence interval level
    params.n_sibling: Number of sibling iterations
    params.main_included: Whether main dataset is included
    params.bias_parquet: Path to sibling parquet (for real bias recalculation)
    params.sibling_id_map: Dict mapping dataset_name -> sibling_id
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts/workflow')
from sibling_utils import get_real_sibling_bias

# Get Snakemake variables
size = int(snakemake.params.size)
ci_level = float(snakemake.params.ci_level)
n_sibling = int(snakemake.params.n_sibling)
main_included = snakemake.params.main_included
bias_parquet = snakemake.params.bias_parquet
sibling_id_map = snakemake.params.sibling_id_map

print(f"Aggregating sibling results for circuit size {size}")
print(f"Number of sibling iterations: {n_sibling}")
print(f"Confidence level: {ci_level*100:.0f}%")
print(f"Parquet for real bias: {bias_parquet}")
print()

# ============================================================================
# Load Sibling Results & Recalculate Real Bias
# ============================================================================

print("Loading sibling pareto fronts...")
sibling_data = []

for i, fpath in enumerate(snakemake.input.sibling_paretos):
    try:
        df = pd.read_csv(fpath)
        df['sibling_id'] = i

        # Extract dataset name from filename to get NPZ sim_index
        # Filename pattern: {dataset_name}_size_{size}_pareto_front.csv
        basename = os.path.basename(fpath)
        dataset_name = basename.rsplit('_size_', 1)[0]
        sim_index = sibling_id_map.get(dataset_name, i)

        # Recalculate real sibling bias for each circuit
        real_biases = []
        for _, row in df.iterrows():
            if pd.notna(row.get('structures', np.nan)):
                structs = row['structures'].split(',')
                real_bias = get_real_sibling_bias(bias_parquet, sim_index, structs)
            else:
                real_bias = np.nan
            real_biases.append(real_bias)

        df['transferred_bias'] = df['mean_bias']  # keep transferred for reference
        df['mean_bias'] = real_biases  # replace with real sibling bias

        sibling_data.append(df)
    except Exception as e:
        print(f"Warning: Failed to load {fpath}: {e}")

df_sibling = pd.concat(sibling_data, ignore_index=True)
print(f"Loaded {len(sibling_data)} sibling pareto fronts")
print(f"Total circuits: {len(df_sibling)}")

# ============================================================================
# Load Main Dataset (if included)
# ============================================================================

df_main = None
if main_included and snakemake.input.main_pareto:
    print("\nLoading main dataset pareto front...")
    main_path = (snakemake.input.main_pareto[0]
                 if isinstance(snakemake.input.main_pareto, list)
                 else snakemake.input.main_pareto)
    df_main = pd.read_csv(main_path)
    print(f"Loaded main pareto front with {len(df_main)} circuits")

# ============================================================================
# Compute Statistics by Bias Limit
# ============================================================================

print("\nComputing confidence intervals by bias limit...")

optimized = df_sibling[df_sibling['circuit_type'] == 'optimized']
bias_limits = sorted(optimized['bias_limit'].unique())

print(f"Found {len(bias_limits)} unique bias limits")

alpha = 1 - ci_level
ci_data = []

for bias_lim in bias_limits:
    circuits = optimized[optimized['bias_limit'] == bias_lim]

    scores = circuits['circuit_score'].values
    real_biases = circuits['mean_bias'].values  # real sibling bias

    score_mean = np.mean(scores)
    score_std = np.std(scores, ddof=1)
    score_ci_lower = np.percentile(scores, alpha / 2 * 100)
    score_ci_upper = np.percentile(scores, (1 - alpha / 2) * 100)

    bias_mean = np.nanmean(real_biases)
    bias_std = np.nanstd(real_biases, ddof=1)
    bias_ci_lower = np.nanpercentile(real_biases, alpha / 2 * 100)
    bias_ci_upper = np.nanpercentile(real_biases, (1 - alpha / 2) * 100)

    main_score = None
    main_bias = None
    if df_main is not None:
        main_at_bias = df_main[(df_main['circuit_type'] == 'optimized') &
                               (df_main['bias_limit'] == bias_lim)]
        if len(main_at_bias) > 0:
            main_score = main_at_bias['circuit_score'].iloc[0]
            main_bias = main_at_bias['mean_bias'].iloc[0]

    ci_data.append({
        'bias_limit': bias_lim,
        'n_sibling': len(scores),
        'score_mean': score_mean,
        'score_std': score_std,
        'score_ci_lower': score_ci_lower,
        'score_ci_upper': score_ci_upper,
        'score_main': main_score,
        'real_bias_mean': bias_mean,
        'real_bias_std': bias_std,
        'real_bias_ci_lower': bias_ci_lower,
        'real_bias_ci_upper': bias_ci_upper,
        'bias_main': main_bias,
    })

df_ci = pd.DataFrame(ci_data)

# ============================================================================
# Build NPZ Profile Arrays (backward-compatible with notebook plotting)
# ============================================================================

print("\nBuilding NPZ profile arrays...")

n_bias_limits = len(bias_limits)
n_loaded = len(sibling_data)

# Per-sibling profiles: (real_mean_bias, circuit_score) at each bias limit
all_profiles = np.full((n_loaded, 2, n_bias_limits), np.nan)

for sid in range(n_loaded):
    sib_opt = optimized[optimized['sibling_id'] == sid]
    for j, bl in enumerate(bias_limits):
        row = sib_opt[sib_opt['bias_limit'] == bl]
        if len(row) > 0:
            all_profiles[sid, 0, j] = row['mean_bias'].iloc[0]  # real sibling bias
            all_profiles[sid, 1, j] = row['circuit_score'].iloc[0]

# Mean profiles
meanbias = np.nanmean(all_profiles[:, 0, :], axis=0)
meanSI = np.nanmean(all_profiles[:, 1, :], axis=0)

# Subsample for plotting (cap at 1000)
n_sub = min(1000, n_loaded)
topbias_sub = all_profiles[:n_sub]

print(f"  meanbias shape: {meanbias.shape}")
print(f"  meanSI shape:   {meanSI.shape}")
print(f"  topbias_sub shape: {topbias_sub.shape}")

# ============================================================================
# Compute Summary Statistics
# ============================================================================

print("\nComputing summary statistics...")

baseline = df_sibling[df_sibling['circuit_type'] == 'baseline']
baseline_scores = baseline['circuit_score'].values

summary_stats = {
    'circuit_size': size,
    'n_sibling': n_sibling,
    'ci_level': ci_level,
    'n_bias_limits': n_bias_limits,
    'baseline_score_mean': np.mean(baseline_scores),
    'baseline_score_std': np.std(baseline_scores, ddof=1),
    'baseline_score_ci_lower': np.percentile(baseline_scores, alpha / 2 * 100),
    'baseline_score_ci_upper': np.percentile(baseline_scores, (1 - alpha / 2) * 100),
    'best_score_mean': df_ci['score_mean'].max(),
    'best_score_ci_lower': df_ci.loc[df_ci['score_mean'].idxmax(), 'score_ci_lower'],
    'best_score_ci_upper': df_ci.loc[df_ci['score_mean'].idxmax(), 'score_ci_upper'],
    'best_bias_limit': df_ci.loc[df_ci['score_mean'].idxmax(), 'bias_limit'],
}

if df_main is not None:
    main_baseline = df_main[df_main['circuit_type'] == 'baseline']
    if len(main_baseline) > 0:
        summary_stats['main_baseline_score'] = main_baseline['circuit_score'].iloc[0]
    main_best = df_main[df_main['circuit_type'] == 'optimized']['circuit_score'].max()
    summary_stats['main_best_score'] = main_best

df_summary = pd.DataFrame([summary_stats])

# ============================================================================
# Save Results
# ============================================================================

print("\nSaving results...")
os.makedirs(os.path.dirname(snakemake.output.summary_csv), exist_ok=True)

df_ci.to_csv(snakemake.output.ci_by_bias, index=False)
print(f"Saved CI by bias limit: {snakemake.output.ci_by_bias}")

df_summary.to_csv(snakemake.output.summary_csv, index=False)
print(f"Saved summary: {snakemake.output.summary_csv}")

np.savez_compressed(
    snakemake.output.profiles_npz,
    meanbias=meanbias,
    meanSI=meanSI,
    topbias_sub=topbias_sub,
    bias_limits=np.array(bias_limits),
)
print(f"Saved NPZ profiles: {snakemake.output.profiles_npz}")

with open(snakemake.output.complete, 'w') as f:
    f.write(f"Sibling aggregation completed for size {size}\n")
    f.write(f"Number of sibling iterations: {n_sibling}\n")
    f.write(f"Confidence level: {ci_level*100:.0f}%\n")
    f.write(f"Number of bias limits: {n_bias_limits}\n")
    f.write(f"\nSummary:\n")
    f.write(f"  Baseline score: {summary_stats['baseline_score_mean']:.6f} "
            f"[{summary_stats['baseline_score_ci_lower']:.6f}, "
            f"{summary_stats['baseline_score_ci_upper']:.6f}]\n")
    f.write(f"  Best score: {summary_stats['best_score_mean']:.6f} "
            f"[{summary_stats['best_score_ci_lower']:.6f}, "
            f"{summary_stats['best_score_ci_upper']:.6f}]\n")
    if df_main is not None:
        f.write(f"\nMain dataset:\n")
        f.write(f"  Baseline score: {summary_stats.get('main_baseline_score', 'N/A')}\n")
        f.write(f"  Best score: {summary_stats.get('main_best_score', 'N/A')}\n")

print(f"Saved completion marker: {snakemake.output.complete}")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "=" * 80)
print("SIBLING NULL AGGREGATION SUMMARY")
print("=" * 80)
print(f"Circuit size: {size}")
print(f"Sibling iterations: {n_sibling}")
print(f"Confidence level: {ci_level*100:.0f}%")
print(f"Bias limits analyzed: {n_bias_limits}")
print(f"Note: mean_bias values are REAL sibling bias (not transferred)")
print()
print(f"Baseline (naive) circuit score:")
print(f"  Mean: {summary_stats['baseline_score_mean']:.6f}")
print(f"  95% CI: [{summary_stats['baseline_score_ci_lower']:.6f}, "
      f"{summary_stats['baseline_score_ci_upper']:.6f}]")
print()
print(f"Best optimized circuit score "
      f"(at bias_limit={summary_stats['best_bias_limit']:.3f}):")
print(f"  Mean: {summary_stats['best_score_mean']:.6f}")
print(f"  95% CI: [{summary_stats['best_score_ci_lower']:.6f}, "
      f"{summary_stats['best_score_ci_upper']:.6f}]")

if df_main is not None:
    print()
    print("Main dataset comparison:")
    print(f"  Baseline score: {summary_stats.get('main_baseline_score', 'N/A')}")
    print(f"  Best score: {summary_stats.get('main_best_score', 'N/A')}")
    if 'main_best_score' in summary_stats:
        in_ci = (summary_stats['best_score_ci_lower']
                 <= summary_stats['main_best_score']
                 <= summary_stats['best_score_ci_upper'])
        print(f"  Main score within sibling 95% CI: {'Yes' if in_ci else 'No'}")

print("=" * 80)
