"""
Aggregate bootstrap circuit search results to compute confidence intervals.

This script processes pareto fronts from all bootstrap iterations and computes:
1. 95% confidence intervals for circuit scores at each bias limit
2. Summary statistics across all bootstrap iterations
3. Comparison with main dataset result

Snakemake inputs:
    input.bootstrap_paretos: List of bootstrap pareto front CSV files
    input.main_pareto: Main dataset pareto front (if included)

Snakemake outputs:
    output.summary_csv: Summary statistics for all bootstrap iterations
    output.ci_by_bias: Confidence intervals by bias limit
    output.complete: Completion marker

Snakemake params:
    params.size: Circuit size
    params.ci_level: Confidence interval level (e.g., 0.95 for 95% CI)
    params.n_bootstrap: Number of bootstrap iterations
    params.main_included: Whether main dataset is included
"""

import os
import numpy as np
import pandas as pd

# Get Snakemake variables
size = int(snakemake.params.size)
ci_level = float(snakemake.params.ci_level)
n_bootstrap = int(snakemake.params.n_bootstrap)
main_included = snakemake.params.main_included

print(f"Aggregating bootstrap results for circuit size {size}")
print(f"Number of bootstrap iterations: {n_bootstrap}")
print(f"Confidence level: {ci_level*100:.0f}%")
print()

# ============================================================================
# Load Bootstrap Results
# ============================================================================

print("Loading bootstrap pareto fronts...")
bootstrap_data = []

for i, fpath in enumerate(snakemake.input.bootstrap_paretos):
    try:
        df = pd.read_csv(fpath)
        # Add bootstrap iteration ID
        df['bootstrap_id'] = i
        bootstrap_data.append(df)
    except Exception as e:
        print(f"Warning: Failed to load {fpath}: {e}")

df_bootstrap = pd.concat(bootstrap_data, ignore_index=True)
print(f"Loaded {len(bootstrap_data)} bootstrap pareto fronts")
print(f"Total circuits: {len(df_bootstrap)}")

# ============================================================================
# Load Main Dataset (if included)
# ============================================================================

df_main = None
if main_included and snakemake.input.main_pareto:
    print("\nLoading main dataset pareto front...")
    main_path = snakemake.input.main_pareto[0] if isinstance(snakemake.input.main_pareto, list) else snakemake.input.main_pareto
    df_main = pd.read_csv(main_path)
    print(f"Loaded main pareto front with {len(df_main)} circuits")

# ============================================================================
# Compute Statistics by Bias Limit
# ============================================================================

print("\nComputing confidence intervals by bias limit...")

# Get unique bias limits (excluding baseline circuits)
optimized_bootstrap = df_bootstrap[df_bootstrap['circuit_type'] == 'optimized']
bias_limits = sorted(optimized_bootstrap['bias_limit'].unique())

print(f"Found {len(bias_limits)} unique bias limits")

# Compute statistics for each bias limit
ci_data = []

for bias_lim in bias_limits:
    # Get all circuits at this bias limit across bootstrap iterations
    circuits_at_bias = optimized_bootstrap[optimized_bootstrap['bias_limit'] == bias_lim]

    scores = circuits_at_bias['circuit_score'].values
    mean_biases = circuits_at_bias['mean_bias'].values

    # Compute confidence intervals
    alpha = 1 - ci_level
    score_mean = np.mean(scores)
    score_std = np.std(scores, ddof=1)
    score_ci_lower = np.percentile(scores, alpha/2 * 100)
    score_ci_upper = np.percentile(scores, (1 - alpha/2) * 100)

    bias_mean = np.mean(mean_biases)
    bias_std = np.std(mean_biases, ddof=1)
    bias_ci_lower = np.percentile(mean_biases, alpha/2 * 100)
    bias_ci_upper = np.percentile(mean_biases, (1 - alpha/2) * 100)

    # Main dataset value at this bias limit (if available)
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
        'n_bootstrap': len(scores),
        'score_mean': score_mean,
        'score_std': score_std,
        'score_ci_lower': score_ci_lower,
        'score_ci_upper': score_ci_upper,
        'score_main': main_score,
        'bias_mean': bias_mean,
        'bias_std': bias_std,
        'bias_ci_lower': bias_ci_lower,
        'bias_ci_upper': bias_ci_upper,
        'bias_main': main_bias
    })

df_ci = pd.DataFrame(ci_data)

# ============================================================================
# Compute Summary Statistics
# ============================================================================

print("\nComputing summary statistics...")

# Statistics for baseline circuits (top N by bias)
baseline_bootstrap = df_bootstrap[df_bootstrap['circuit_type'] == 'baseline']
baseline_scores = baseline_bootstrap['circuit_score'].values

summary_stats = {
    'circuit_size': size,
    'n_bootstrap': n_bootstrap,
    'ci_level': ci_level,
    'n_bias_limits': len(bias_limits),

    # Baseline (naive) circuit statistics
    'baseline_score_mean': np.mean(baseline_scores),
    'baseline_score_std': np.std(baseline_scores, ddof=1),
    'baseline_score_ci_lower': np.percentile(baseline_scores, alpha/2 * 100),
    'baseline_score_ci_upper': np.percentile(baseline_scores, (1 - alpha/2) * 100),

    # Best optimized circuit statistics (across all bias limits)
    'best_score_mean': df_ci['score_mean'].max(),
    'best_score_ci_lower': df_ci.loc[df_ci['score_mean'].idxmax(), 'score_ci_lower'],
    'best_score_ci_upper': df_ci.loc[df_ci['score_mean'].idxmax(), 'score_ci_upper'],
    'best_bias_limit': df_ci.loc[df_ci['score_mean'].idxmax(), 'bias_limit'],
}

# Add main dataset comparison if available
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

# Save confidence intervals by bias limit
df_ci.to_csv(snakemake.output.ci_by_bias, index=False)
print(f"Saved CI by bias limit: {snakemake.output.ci_by_bias}")

# Save summary statistics
df_summary.to_csv(snakemake.output.summary_csv, index=False)
print(f"Saved summary statistics: {snakemake.output.summary_csv}")

# Create completion marker
with open(snakemake.output.complete, 'w') as f:
    f.write(f"Bootstrap aggregation completed for size {size}\n")
    f.write(f"Number of bootstrap iterations: {n_bootstrap}\n")
    f.write(f"Confidence level: {ci_level*100:.0f}%\n")
    f.write(f"Number of bias limits: {len(bias_limits)}\n")
    f.write(f"\nSummary:\n")
    f.write(f"  Baseline score: {summary_stats['baseline_score_mean']:.6f} ")
    f.write(f"[{summary_stats['baseline_score_ci_lower']:.6f}, {summary_stats['baseline_score_ci_upper']:.6f}]\n")
    f.write(f"  Best score: {summary_stats['best_score_mean']:.6f} ")
    f.write(f"[{summary_stats['best_score_ci_lower']:.6f}, {summary_stats['best_score_ci_upper']:.6f}]\n")
    if df_main is not None:
        f.write(f"\nMain dataset:\n")
        f.write(f"  Baseline score: {summary_stats.get('main_baseline_score', 'N/A')}\n")
        f.write(f"  Best score: {summary_stats.get('main_best_score', 'N/A')}\n")

print(f"Saved completion marker: {snakemake.output.complete}")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*80)
print("BOOTSTRAP AGGREGATION SUMMARY")
print("="*80)
print(f"Circuit size: {size}")
print(f"Bootstrap iterations: {n_bootstrap}")
print(f"Confidence level: {ci_level*100:.0f}%")
print(f"Bias limits analyzed: {len(bias_limits)}")
print()
print(f"Baseline (naive) circuit score:")
print(f"  Mean: {summary_stats['baseline_score_mean']:.6f}")
print(f"  95% CI: [{summary_stats['baseline_score_ci_lower']:.6f}, {summary_stats['baseline_score_ci_upper']:.6f}]")
print()
print(f"Best optimized circuit score (at bias_limit={summary_stats['best_bias_limit']:.3f}):")
print(f"  Mean: {summary_stats['best_score_mean']:.6f}")
print(f"  95% CI: [{summary_stats['best_score_ci_lower']:.6f}, {summary_stats['best_score_ci_upper']:.6f}]")

if df_main is not None:
    print()
    print("Main dataset comparison:")
    print(f"  Baseline score: {summary_stats.get('main_baseline_score', 'N/A')}")
    print(f"  Best score: {summary_stats.get('main_best_score', 'N/A')}")
    if 'main_best_score' in summary_stats:
        in_ci = (summary_stats['best_score_ci_lower'] <= summary_stats['main_best_score'] <= summary_stats['best_score_ci_upper'])
        print(f"  Main score within bootstrap 95% CI: {'Yes' if in_ci else 'No'}")

print("="*80)
