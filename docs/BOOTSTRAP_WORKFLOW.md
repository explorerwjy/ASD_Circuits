##

 Bootstrap Circuit Search Workflow

**Purpose**: Run circuit search on 100 bootstrap mutation bias replicates to compute 95% confidence intervals for the main result.

## Overview

This workflow extends the main circuit search pipeline to support bootstrap analysis:

1. **Auto-discovers** bootstrap bias files from a directory
2. **Runs circuit search** on all 100 bootstrap iterations in parallel
3. **Aggregates results** to compute 95% confidence intervals
4. **Keeps config clean** - no need to list 100 datasets manually

## Quick Start

### 1. List discovered bootstrap datasets

```bash
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  list_bootstrap_datasets
```

### 2. Run circuit search on all bootstrap iterations

```bash
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  --cores 20 \
  -n  # Dry run to check

snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  --cores 20
```

### 3. Aggregate bootstrap results to compute 95% CI

```bash
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  aggregate_all
```

## Directory Structure

### Input Files

Bootstrap bias files:
```
results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling/
├── Spark_ExomeWide.GeneWeight.boot0.csv
├── Spark_ExomeWide.GeneWeight.boot1.csv
├── ...
└── Spark_ExomeWide.GeneWeight.boot99.csv
```

### Output Files

Individual bootstrap results:
```
results/CircuitSearch_Bootstrap/
├── ASD_Boot0/
│   ├── pareto_fronts/
│   │   └── ASD_Boot0_size_46_pareto_front.csv
│   └── SA_results/
├── ASD_Boot1/
│   └── ...
└── ...
```

Aggregated results (95% CI):
```
results/CircuitSearch_Bootstrap_Summary/
└── size_46/
    ├── bootstrap_summary.csv                    # Overall statistics
    ├── confidence_intervals_by_biaslimit.csv    # CI for each bias limit
    └── bootstrap_aggregation_complete.txt       # Summary text
```

## Configuration

### Bootstrap Config (`config/circuit_config_bootstrap.yaml`)

```yaml
# Enable bootstrap mode
bootstrap_mode: true

# Bootstrap file discovery (auto-detects all matching files)
bootstrap_bias_dir: "results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling"
bootstrap_bias_pattern: "Spark_ExomeWide.GeneWeight.boot*.csv"
bootstrap_dataset_prefix: "ASD_Boot"

# Include main dataset for comparison
include_main_dataset: true

Input_str_bias:
  ASD_Main:
    name: "ASD_SPARK_Main"
    bias_df: "dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv"

# Use reduced SA runs for bootstrap (faster)
sa_runtimes: 5  # vs 20 for main dataset

# Other parameters same as main analysis
circuit_sizes: [46]
top_n: 213
measure: "SI"
```

## Advanced Usage

### Run only first 10 bootstrap iterations (for testing)

```bash
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  --config n_bootstrap=10 \
  --cores 10
```

### Run specific bootstrap iteration

```bash
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  results/CircuitSearch_Bootstrap/ASD_Boot5/pareto_fronts/ASD_Boot5_size_46_pareto_front.csv
```

### Skip main dataset (only bootstrap)

Edit config:
```yaml
include_main_dataset: false
```

## Output Format

### 1. Individual Pareto Fronts

Each bootstrap iteration produces a pareto front CSV identical to the main analysis:

```csv
bias_limit,circuit_score,mean_bias,n_structures,structures,circuit_type
,0.432567,0.4312,46,"Nucleus_accumbens,...",baseline
0.300,0.715234,0.3005,46,"Nucleus_accumbens,...",optimized
0.350,0.798123,0.3502,46,"Prelimbic_area,...",optimized
...
```

### 2. Confidence Intervals by Bias Limit

`confidence_intervals_by_biaslimit.csv`:

| Column | Description |
|--------|-------------|
| `bias_limit` | Minimum bias constraint |
| `n_bootstrap` | Number of bootstrap iterations at this limit |
| `score_mean` | Mean circuit score across bootstraps |
| `score_std` | Standard deviation of circuit score |
| `score_ci_lower` | Lower bound of 95% CI |
| `score_ci_upper` | Upper bound of 95% CI |
| `score_main` | Main dataset score (for comparison) |
| `bias_mean` | Mean bias across bootstraps |
| `bias_std` | Standard deviation of bias |
| `bias_ci_lower` | Lower bound of 95% CI for bias |
| `bias_ci_upper` | Upper bound of 95% CI for bias |
| `bias_main` | Main dataset bias (for comparison) |

Example:
```csv
bias_limit,n_bootstrap,score_mean,score_std,score_ci_lower,score_ci_upper,score_main,...
0.300,100,0.715234,0.012345,0.692145,0.738234,0.721453,...
0.350,100,0.798123,0.010234,0.778123,0.818123,0.802069,...
```

### 3. Summary Statistics

`bootstrap_summary.csv`:

Single row with overall statistics:
- `baseline_score_mean`: Mean score of naive (top N by bias) circuit
- `baseline_score_ci_lower/upper`: 95% CI for baseline score
- `best_score_mean`: Mean score of best optimized circuit
- `best_score_ci_lower/upper`: 95% CI for best score
- `best_bias_limit`: Bias limit where best score was achieved
- `main_baseline_score`: Main dataset baseline score
- `main_best_score`: Main dataset best score

## Visualization Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load aggregated results
df_ci = pd.read_csv("results/CircuitSearch_Bootstrap_Summary/size_46/confidence_intervals_by_biaslimit.csv")

# Plot pareto front with 95% CI
fig, ax = plt.subplots(figsize=(10, 6))

# Bootstrap mean and CI
ax.plot(df_ci['bias_mean'], df_ci['score_mean'],
        'o-', label='Bootstrap Mean', linewidth=2)
ax.fill_between(df_ci['bias_mean'],
                df_ci['score_ci_lower'],
                df_ci['score_ci_upper'],
                alpha=0.3, label='95% CI')

# Main dataset (if available)
if 'score_main' in df_ci.columns:
    ax.plot(df_ci['bias_main'], df_ci['score_main'],
            's--', label='Main Dataset', linewidth=2)

ax.set_xlabel('Mean Bias')
ax.set_ylabel('Circuit Score (Shannon Information)')
ax.set_title('Circuit Optimization Pareto Front with Bootstrap 95% CI')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bootstrap_pareto_front_with_ci.png', dpi=150)
```

## Performance Considerations

### Computational Cost

- **100 bootstrap iterations** × **~20 bias limits** × **5 SA runs** = **10,000 SA optimizations**
- Each SA run: ~10-30 seconds (depending on steps)
- Total time: ~30-100 hours on single core

### Recommended Settings

**For quick testing** (first 10 bootstraps):
```yaml
n_bootstrap: 10
sa_runtimes: 3
sa_steps: 10000
```
Time: ~1-2 hours on 10 cores

**For production** (all 100 bootstraps):
```yaml
n_bootstrap: null  # Use all 100
sa_runtimes: 5     # Reduced from 20 (main dataset)
sa_steps: 50000    # Keep quality
```
Time: ~30 hours on 20 cores

### Parallelization Strategy

The pipeline is embarrassingly parallel:
- Each bootstrap iteration is independent
- Snakemake automatically distributes jobs across cores
- Recommended: Use 20-40 cores for efficient parallel processing

```bash
# Use SLURM cluster
snakemake -s Snakefile.circuit.bootstrap \
  --configfile config/circuit_config_bootstrap.yaml \
  --cluster "sbatch -p short -c 1 --mem=4G" \
  -j 100  # Submit up to 100 jobs in parallel
```

## Key Benefits

1. **Clean Config**: No need to list 100 datasets - auto-discovery from directory
2. **Reusable**: Same Snakefile works for any number of bootstrap files
3. **Parallel**: All bootstrap iterations run independently
4. **Confidence Intervals**: Automatic 95% CI calculation for publication
5. **Comparison**: Main dataset result shown alongside bootstrap CI

## Troubleshooting

### Issue: Not finding bootstrap files

```bash
# Test the discovery function
python scripts/workflow/bootstrap_utils.py \
  results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling \
  "Spark_ExomeWide.GeneWeight.boot*.csv"
```

### Issue: Column name mismatch

Bootstrap files might use different column names. Check format:
```bash
head -3 results/Bootstrap_bias/Spark_ExomeWide/Weighted_Resampling/Spark_ExomeWide.GeneWeight.boot0.csv
```

Expected columns: `Structure`, `EFFECT`, `Rank` (or `STR`, `EFFECT`, `Rank`)

### Issue: Out of memory

Reduce parallel jobs:
```bash
snakemake ... --cores 10  # Instead of 20
```

Or increase per-job memory in Snakefile:
```python
resources:
    mem_mb = 4000  # Instead of 2000
```

## Files Created

- `Snakefile.circuit.bootstrap`: Bootstrap-specific Snakemake pipeline
- `config/circuit_config_bootstrap.yaml`: Bootstrap configuration
- `scripts/workflow/bootstrap_utils.py`: Bootstrap dataset discovery utilities
- `scripts/workflow/aggregate_bootstrap_results.py`: Aggregation and CI calculation
- `BOOTSTRAP_WORKFLOW.md`: This documentation
