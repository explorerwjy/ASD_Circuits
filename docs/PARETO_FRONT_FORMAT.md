# Pareto Front CSV Format

**Date**: 2025-11-01
**Feature**: Baseline circuit included in pareto front output

## Summary

The pareto front CSV now includes the **baseline circuit** (top N structures by bias without optimization) as the first row, allowing easy comparison with SA-optimized circuits.

## CSV Format

The pareto front CSV file (`{dataset_name}_size_{size}_pareto_front.csv`) contains:

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `bias_limit` | float or None | Minimum bias constraint for SA optimization. `None` for baseline circuit |
| `circuit_score` | float | Circuit score (Shannon Information or Connectivity depending on measure) |
| `mean_bias` | float | Average mutation bias across circuit structures |
| `n_structures` | int | Number of structures in circuit (should equal size parameter) |
| `structures` | string | Comma-separated list of structure names |
| `circuit_type` | string | Either `'baseline'` (top N by bias) or `'optimized'` (SA-optimized) |

### Row Order

1. **Row 1**: Baseline circuit (naive approach)
   - `circuit_type = 'baseline'`
   - `bias_limit = None`
   - Top N structures by bias, no optimization

2. **Row 2+**: SA-optimized circuits (pareto front)
   - `circuit_type = 'optimized'`
   - `bias_limit = specific value` (sorted ascending)
   - Optimized for maximum score while satisfying bias constraint

## Example

```csv
bias_limit,circuit_score,mean_bias,n_structures,structures,circuit_type
,0.443142,0.4338,46,"Nucleus_accumbens,Orbital_area_lateral_part,...",baseline
0.300,0.721453,0.3001,46,"Nucleus_accumbens,Prelimbic_area,...",optimized
0.320,0.765289,0.3203,46,"Nucleus_accumbens,Orbital_area_lateral_part,...",optimized
0.340,0.789012,0.3405,46,"Prelimbic_area,Primary_somatosensory_area,...",optimized
0.350,0.802069,0.3500,46,"Nucleus_accumbens,Prelimbic_area,...",optimized
...
```

## Usage

### Loading in Python

```python
import pandas as pd

# Load pareto front
df = pd.read_csv("results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv")

# Get baseline circuit
baseline = df[df['circuit_type'] == 'baseline'].iloc[0]
print(f"Baseline score: {baseline['circuit_score']:.6f}")
print(f"Baseline bias: {baseline['mean_bias']:.4f}")

# Get optimized circuits
optimized = df[df['circuit_type'] == 'optimized']
print(f"\nOptimized circuits: {len(optimized)}")
print(f"Best optimized score: {optimized['circuit_score'].max():.6f}")

# Compare improvement
best_opt = optimized.loc[optimized['circuit_score'].idxmax()]
improvement = (best_opt['circuit_score'] - baseline['circuit_score']) / baseline['circuit_score'] * 100
print(f"\nImprovement: {improvement:.1f}%")
```

### Loading in R

```r
library(tidyverse)

# Load pareto front
df <- read_csv("results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv")

# Get baseline circuit
baseline <- df %>% filter(circuit_type == "baseline")
cat(sprintf("Baseline score: %.6f\n", baseline$circuit_score))

# Get optimized circuits
optimized <- df %>% filter(circuit_type == "optimized")
cat(sprintf("Optimized circuits: %d\n", nrow(optimized)))
cat(sprintf("Best optimized score: %.6f\n", max(optimized$circuit_score)))

# Plot pareto front
ggplot(df, aes(x = mean_bias, y = circuit_score, color = circuit_type)) +
  geom_point(size = 3) +
  geom_line(data = optimized) +
  theme_minimal() +
  labs(title = "Circuit Optimization Pareto Front",
       x = "Mean Bias",
       y = "Circuit Score")
```

## Benefits

1. **Easy Comparison**: Baseline circuit provides reference point to measure optimization improvement
2. **Validation**: Confirms that SA optimization actually improves over naive approach
3. **Interpretability**: Shows the score you'd get by simply taking top N biased structures
4. **Plotting**: First point on plots shows naive approach, rest shows optimization trade-offs

## Implementation Details

The baseline circuit is calculated in `scripts/workflow/create_pareto_front.py`:

1. Load bias data from dataset configuration
2. Select top N structures by bias (where N = circuit size)
3. Calculate mean bias of baseline circuit
4. Calculate circuit score using same measure as optimization (SI or Connectivity)
5. Add as first row in pareto front CSV with `circuit_type = 'baseline'`
6. Append all SA-optimized circuits with `circuit_type = 'optimized'`

## File Locations

Pareto front CSVs are saved at:
```
{output_dir}/{dataset_name}/pareto_fronts/{dataset_name}_size_{size}_pareto_front.csv
```

Default location:
```
results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv
```

## Notes

- The baseline circuit has the **highest mean bias** (top N structures) but typically a **lower score** than optimized circuits
- SA optimization explores the trade-off: slightly lower bias in exchange for much higher circuit score
- The pareto front shows this trade-off curve, with baseline as the extreme high-bias point
