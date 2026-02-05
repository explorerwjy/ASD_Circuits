# GENCIC - Project Instructions for Claude Code

## Project Overview

GENCIC (Genetics and Expression iNtegration with Connectome to Infer affected Circuits) identifies brain circuits targeted by autism spectrum disorder (ASD) mutations. It integrates genomic data with spatial transcriptomics and brain-wide connectome data.

## Environment

- **Conda environment**: `gencic`
- **Always activate before running anything**: `conda activate gencic`
- **Python**: 3.7+
- **R**: 4.2.1+ (used in some analyses)
- **Snakemake**: installed separately (use `conda activate snakemake` or `conda activate smk` if snakemake is not in `gencic` env)

## Repository Structure

```
├── src/                        # Core Python modules
│   ├── ASD_Circuits.py         # Main analysis: bias calculation, gene loading, circuit search
│   ├── CellType_PSY.py         # Cell type-specific & psychiatric disorder analysis
│   ├── SA.py                   # Simulated annealing base implementation
│   └── SA_optimized.py         # Numba-optimized SA (6-15x speedup)
│
├── scripts/                    # Standalone analysis scripts
│   ├── script_bias_cal.py      # Calculate bias with p-values
│   ├── script_generate_geneweights.py  # Generate random gene weights
│   ├── script_run_ctrl_sim.py  # Null distribution simulations
│   ├── script_circuit_search.SI.py     # Circuit search via SA
│   └── workflow/               # Snakemake workflow helper scripts
│       ├── generate_bias_limits.py
│       ├── run_sa_search.py
│       ├── create_pareto_front.py
│       ├── extract_best_circuits.py
│       ├── create_metadata.py
│       └── aggregate_bootstrap_results.py
│
├── config/                     # YAML configuration files
│   ├── config.yaml             # Main bias calculation config
│   ├── config.SC.DN.yaml       # Single-cell de novo config
│   ├── circuit_config.yaml     # Circuit search parameters
│   └── circuit_config_bootstrap.yaml
│
├── Snakefile.bias              # Bias calculation pipeline
├── Snakefile.circuit           # Circuit search pipeline
├── Snakefile.circuit.bootstrap # Bootstrap analysis pipeline
├── Snakefile.circuit.refactored # Refactored circuit search
│
├── notebooks_mouse_str/        # 28 structure-level analysis notebooks (01-07 = main pipeline)
├── notebooks_mouse_sc/         # 17 cell type-level analysis notebooks
├── dat/                        # Data directory (not committed to git)
└── requirements.txt            # Python dependencies
```

## Key Dependencies

numpy, pandas, scipy, matplotlib, seaborn, statsmodels, python-igraph, anndata, tqdm, tabulate, Requests

## Snakemake Workflows

Three main pipelines, run from the project root:

```bash
# 1. Bias calculation
snakemake -s Snakefile.bias --configfile config/config.yaml --cores 10

# 2. Circuit search (simulated annealing + Pareto fronts)
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10

# 3. Bootstrap robustness analysis
snakemake -s Snakefile.circuit.bootstrap --configfile config/circuit_config_bootstrap.yaml --cores 10
```

## Coding Conventions

- Core modules in `src/` are imported by scripts and notebooks via `sys.path` manipulation (e.g., `sys.path.insert(0, 'src/')`)
- Config files use YAML with paths relative to `ProjDIR` or absolute paths
- Analysis outputs go to `results/` directory
- Scripts use argparse for CLI arguments
- Notebooks are tracked with Git LFS (see `.gitattributes`)

## Analysis Modes

- **STR_ISH**: Structure-level analysis using Allen Mouse Brain ISH data (213 brain structures)
- **CT_Z2**: Cell type-level analysis using cluster specificity scores

## Data & File Conventions

- Gene weight files: `.gw` or `.gw.csv` format
- Expression matrices: `.parquet` format
- Bias results: `.csv` format
- Circuit search results: text files in `results/CircuitSearch/`
- Config references data under `dat/` and external paths under `/home/jw3514/Work/`

## Important Notes

- Never commit data files (*.csv, *.parquet, *.h5, *.gw, etc.) to git
- Jupyter notebooks are managed via Git LFS — do not change `.gitattributes`
- The `dat/` directory contains large data files and should not be committed
- Config files contain absolute paths specific to this machine — be aware when modifying
- SA optimization uses numba JIT compilation; first run may be slow due to compilation
