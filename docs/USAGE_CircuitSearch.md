# Quick Usage Guide - Circuit Search Pipeline

## Your New Config Structure

Your config now supports multiple datasets in one file:

```yaml
Input_str_bias:
  ASD_All:                     # Dataset key (used internally)
    name: "ASD_SPARK_61"       # Dataset name (used for folder names)
    bias_df: "dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv"
    description: "ASD SPARK cohort with meta-analysis gene weights"

  DDD_293:                     # You can add more datasets
    name: "DDD_293"
    bias_df: "dat/Unionize_bias/DDD_top293.Z2.bias.csv"
    description: "DDD cohort top 293 genes"

  SCZ_61:
    name: "SCZ_61"
    bias_df: "dat/Unionize_bias/SCZ_top61.Z2.bias.csv"
    description: "SCZ cohort top 61 genes"
```

## Running the Pipeline

### Option 1: Run a Specific Dataset

```bash
# Run only ASD_All dataset
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=ASD_All --cores 10

# Run only DDD_293 dataset
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=DDD_293 --cores 10
```

### Option 2: Run All Datasets

```bash
# Run all datasets defined in config
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10
```

## Output Structure

Results will be organized by dataset name:

```
results/CircuitSearch/
├── ASD_SPARK_61/                    # Dataset name from config
│   ├── biaslims/
│   │   ├── biaslim.size.46.txt
│   │   └── biaslim.size.46.filtered.txt
│   ├── SA_results/
│   │   └── size_46/
│   │       ├── SA..topN_213-keepN_46-minbias_0.300.txt
│   │       └── ...
│   ├── best_circuits/
│   │   └── size_46_best_circuits.txt
│   ├── pareto_fronts/
│   │   └── ASD_SPARK_61_size_46_pareto_front.csv  ⭐
│   └── analysis_metadata.yaml
│
├── DDD_293/                         # Another dataset
│   ├── pareto_fronts/
│   │   └── DDD_293_size_46_pareto_front.csv
│   └── ...
│
└── SCZ_61/                          # Yet another dataset
    ├── pareto_fronts/
    │   └── SCZ_61_size_46_pareto_front.csv
    └── ...
```

## Adding a New Dataset

1. Edit `config/circuit_config.yaml`:

```yaml
Input_str_bias:
  # ... existing datasets ...

  MyNewDataset:                      # New dataset key
    name: "MyNewDataset_v1"          # Folder name
    bias_df: "dat/Unionize_bias/my_new_bias.csv"
    description: "My new analysis"
```

2. Run it:

```bash
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=MyNewDataset --cores 10
```

## Loading Results

### Python

```python
import pandas as pd

# Load ASD results
asd_df = pd.read_csv("results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv")

# Load DDD results
ddd_df = pd.read_csv("results/CircuitSearch/DDD_293/pareto_fronts/DDD_293_size_46_pareto_front.csv")

# Compare
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(asd_df['circuit_score'], asd_df['mean_bias'], 'o-', label='ASD', alpha=0.7)
plt.plot(ddd_df['circuit_score'], ddd_df['mean_bias'], 's-', label='DDD', alpha=0.7)
plt.xlabel('Circuit Score')
plt.ylabel('Mean Bias')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### R

```r
library(tidyverse)

# Load results
asd <- read_csv("results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_46_pareto_front.csv")
ddd <- read_csv("results/CircuitSearch/DDD_293/pareto_fronts/DDD_293_size_46_pareto_front.csv")

# Add dataset labels
asd <- asd %>% mutate(dataset = "ASD")
ddd <- ddd %>% mutate(dataset = "DDD")

# Combine and plot
combined <- bind_rows(asd, ddd)

ggplot(combined, aes(x=circuit_score, y=mean_bias, color=dataset)) +
  geom_line() +
  geom_point() +
  labs(title="Pareto Fronts Comparison",
       x="Circuit Score",
       y="Mean Bias") +
  theme_minimal()
```

## Common Tasks

### Test with Dry Run

```bash
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --config dataset=ASD_All -n
```

### Run Faster (for testing)

Edit your config:
```yaml
sa_runtimes: 3
sa_steps: 10000
min_bias_rank: 30
```

### Run Multiple Datasets in Parallel

```bash
# Run ASD in background
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=ASD_All --cores 10 &

# Run DDD in background
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=DDD_293 --cores 10 &

# Wait for all to finish
wait
```

### Check Status

```bash
# See what files would be created
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --config dataset=ASD_All -n

# See the DAG
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --config dataset=ASD_All --dag | dot -Tpdf > dag.pdf
```

## Troubleshooting

### Error: Dataset not found

```
ValueError: Dataset 'XXX' not found in Input_str_bias
```

**Solution**: Check that the dataset key in `--config dataset=XXX` matches a key in `Input_str_bias` section of your config.

### Different dataset name vs key

- **Dataset KEY** (e.g., `ASD_All`): Used in `--config dataset=ASD_All`
- **Dataset NAME** (e.g., `ASD_SPARK_61`): Used for folder names in output

Make sure your command uses the KEY, not the NAME:
```bash
# ✅ Correct
snakemake ... --config dataset=ASD_All

# ❌ Wrong
snakemake ... --config dataset=ASD_SPARK_61
```

## Best Practices

1. **Use descriptive dataset names**: `ASD_SPARK_61_v2` instead of `test1`
2. **Keep dataset keys short**: `ASD_All`, `DDD_293`, `SCZ_61`
3. **Document in descriptions**: Useful for future reference
4. **Test with one dataset first**: Before running all
5. **Use version control**: Track your config file changes

## Examples

### Example 1: Compare ASD vs Siblings

```yaml
Input_str_bias:
  ASD:
    name: "ASD_SPARK_61"
    bias_df: "dat/Unionize_bias/ASD_denovo.bias.csv"
    description: "ASD de novo mutations"

  Sibling:
    name: "Sibling_Controls"
    bias_df: "dat/Unionize_bias/Sibling_denovo.bias.csv"
    description: "Sibling controls"
```

```bash
# Run both
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 20
```

### Example 2: Different Gene Sets

```yaml
Input_str_bias:
  ASD_LGD:
    name: "ASD_LGD"
    bias_df: "dat/Unionize_bias/ASD_LGD.bias.csv"
    description: "ASD likely gene-disrupting variants"

  ASD_Missense:
    name: "ASD_Missense"
    bias_df: "dat/Unionize_bias/ASD_Missense.bias.csv"
    description: "ASD missense variants"
```

### Example 3: IQ Stratification

```yaml
Input_str_bias:
  ASD_High_IQ:
    name: "ASD_HIQ"
    bias_df: "dat/Unionize_bias/ASD_HIQ.bias.csv"
    description: "ASD high IQ (>100)"

  ASD_Low_IQ:
    name: "ASD_LIQ"
    bias_df: "dat/Unionize_bias/ASD_LIQ.bias.csv"
    description: "ASD low IQ (<70)"
```
