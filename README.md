# GENCIC: Genetics and Expression iNtegration with Connectome to Infer affected Circuits

## Overview

GENCIC is a computational framework that identifies brain circuits preferentially targeted by autism spectrum disorder (ASD) mutations. It integrates genomic data from ASD patients with spatial transcriptomics and brain-wide connectome data to discover functionally cohesive neural circuits affected in autism.

## Citation

Wang J, Chang J, Chiang AH, Vitkup D. Autism Mutations Preferentially Target Distributed Brain Circuits Associated with Sensory Integration and Decision Making. 2025.

## Quick Start

```bash
# 1. Set up environment
conda create -n gencic python=3.10
conda activate gencic
pip install numpy pandas scipy matplotlib seaborn statsmodels igraph anndata snakemake jupytext nbstripout joblib numba

# 2. Download data
#    Get the intermediate data bundle from:
#    https://vitkuplab.c2b2.columbia.edu/Gencic/index.html
#    Unpack into dat/

# 3. Run structure-level analysis
cd notebooks_mouse_str
jupyter nbconvert --to notebook --execute --inplace 01.Download_ISH_data.ipynb
jupyter nbconvert --to notebook --execute --inplace 02.Preprocessing_ISH_data.ipynb
# ... etc (see notebooks_mouse_str/README.md)

# 4. Run bias significance pipeline
snakemake -s Snakefile.bias --configfile config/config.yaml --cores 10

# 5. Run circuit search
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10
```

## Repository Structure

```
ASD_Circuits_CellType/
│
├── config/                              # Configuration files
│   ├── config.yaml                      # Main config (gene sets, data paths, pipeline params)
│   ├── config.SC.DN.yaml                # Cell-type bias pipeline config (DN weights)
│   ├── circuit_config.yaml              # Circuit search parameters
│   ├── circuit_config_bootstrap.yaml    # Bootstrap circuit search config
│   └── circuit_config_sibling_*.yaml    # Sibling null circuit search configs
│
├── src/                                 # Core Python modules
│   ├── ASD_Circuits.py                  # Main library (bias, gene weights, circuit scoring)
│   ├── CellType_PSY.py                  # Cell-type analysis functions (Z1, Z2)
│   ├── SA.py                            # Simulated annealing for circuit optimization
│   ├── SA_optimized.py                  # Numba-accelerated SA variant
│   └── plot.py                          # Shared plotting functions (22 functions)
│
├── scripts/                             # Standalone scripts
│   ├── script_bias_cal.py               # Calculate bias with p-values from null distributions
│   ├── script_generate_geneweights.py   # Generate random gene weights for null
│   ├── script_run_ctrl_sim.py           # Run sibling/random null simulations
│   ├── script_compute_Z2.py             # Parallelized Z2 computation
│   ├── script_circuit_search.SI.py      # Circuit search via simulated annealing
│   ├── build_cluster_mean_log_umi.py    # Build cluster expression from Allen h5ad files
│   ├── build_celltype_z2_matrix.py      # Build cell-type Z1/Z2 matrices
│   ├── build_subclass_expmat_cpm.py     # Build V2/V3 subclass CPM expression matrices
│   ├── build_merfish_umi_matrices.py    # Build MERFISH structure-level UMI matrices
│   ├── build_merfish_annotation.py      # Annotate MERFISH cells with ISH structure names
│   ├── generate_sibling_bias_files.py   # Generate sibling null bias parquets
│   └── workflow/                        # Snakemake helper scripts
│       ├── generate_bias_limits.py      # Pareto front bias limits
│       ├── run_sa_search.py             # Run SA search
│       ├── create_pareto_front.py       # Create Pareto fronts
│       ├── extract_best_circuits.py     # Extract best circuits
│       ├── sibling_utils.py             # Sibling null utilities
│       └── aggregate_*.py              # Aggregation scripts
│
├── notebooks_mouse_str/                 # Structure-level analysis (Figures 1–4)
├── notebooks_mouse_sc/                  # Cell-type analysis (Figure 5)
├── notebooks_figures/                   # Manuscript figures & tables
├── notebook_rebuttal/                   # Reviewer-requested analyses
│
├── Snakefile.bias                       # Bias significance pipeline
├── Snakefile.circuit                    # Circuit search pipeline
├── Snakefile.circuit.bootstrap          # Bootstrap analysis pipeline
└── Snakefile.circuit.sibling            # Sibling null circuit search
```

## Data

### Raw Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| Allen Mouse Brain ISH | In-situ hybridization expression energy for ~17K genes × 213 structures | [Allen Brain Atlas API](https://mouse.brain-map.org/) |
| Allen Mouse Connectivity | Structural connectivity atlas (Oh et al. 2014 Nature) | [Allen Connectivity Atlas](https://connectivity.brain-map.org/) |
| Allen Brain Cell Atlas | Single-cell RNA-seq (10x V2/V3), 5,312 cell-type clusters | [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas) |
| Allen MERFISH | Whole-brain spatial transcriptomics (~3.7M cells) | [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas) |
| SPARK/ASC | De novo mutations from ASD exome sequencing | [SFARI Base](https://base.sfari.org/) |
| DDD/NDD | Developmental disorder de novo mutations (Kaplanis et al. 2020) | [Nature paper](https://doi.org/10.1038/s41586-020-2832-5) |
| gnomAD v4.0 | Gene constraint metrics (LOEUF) | [gnomAD](https://gnomad.broadinstitute.org/) |

### Intermediate Data (Provided)

Key intermediate files are available at [https://vitkuplab.c2b2.columbia.edu/Gencic/](https://vitkuplab.c2b2.columbia.edu/Gencic/index.html). Once these are in `dat/`, all notebooks can run without the raw data above.

| Directory | Contents |
|-----------|----------|
| `dat/allen-mouse-exp/` | ISH expression matrices (raw, log2+QN, Z1, Z2), expression match files |
| `dat/allen-mouse-conn/` | Connectivity scoring matrices (InfoMat, WeightMat), null rank scores |
| `dat/BiasMatrices/` | Z2 expression specificity matrices (structure-level and cell-type) |
| `dat/Genetics/` | Gene weights, mutation tables, constraint metrics, sibling weights |
| `dat/MERFISH/` | MERFISH cell annotations, structure-level UMI matrices, Z1/Z2 matrices |
| `dat/ExpressionMats/` | Subclass-level V2/V3 CPM expression matrices |
| `dat/SC_UMI_Mats/` | Cluster-level mean log UMI expression |
| `dat/Unionize_bias/` | Pre-computed structure-level bias with FDR |

## Analysis Pipelines

### 1. Structure-Level Bias (`Snakefile.bias`)

Computes bias significance via permutation testing against sibling and random null distributions.

```bash
# Structure-level (ISH)
snakemake -s Snakefile.bias --cores 10

# Cell-type level (requires DN weights from notebook_mouse_sc/02)
snakemake -s Snakefile.bias --configfile config/config.SC.DN.yaml --cores 10
```

**Steps**: generate random gene weights → compute null distributions (10K permutations) → calculate bias with p-values and FDR.

### 2. Circuit Search (`Snakefile.circuit`)

Finds brain circuits that maximize both ASD bias and connectivity using simulated annealing along Pareto fronts.

```bash
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10
```

**Steps**: generate bias limits → run SA search → create Pareto fronts → extract best circuits.

### 3. Bootstrap Robustness (`Snakefile.circuit.bootstrap`)

Bootstrap resampling of mutations to assess circuit robustness.

```bash
snakemake -s Snakefile.circuit.bootstrap --configfile config/circuit_config_bootstrap.yaml --cores 10
```

## Notebook Organization

### Structure-Level (`notebooks_mouse_str/`)

Core pipeline (01–07) plus supplementary analyses (08–13). See `notebooks_mouse_str/README.md` for details.

| Notebook | Description | Manuscript |
|----------|-------------|------------|
| 01–02 | ISH data download and preprocessing (→ Z2 matrix) | Methods |
| 03 | Connectivity data preprocessing (→ InfoMat) | Methods |
| 04 | ASD mutation bias computation (5 gene sets) | Fig 1 |
| 05 | Circuit search and Pareto front analysis | Fig 2 |
| 06 | IQ stratification (HIQ vs LIQ) | Fig 3 |
| 07 | Distance-stratified connectivity analysis | Fig 3 |
| 08 | DDD/NDD comparison + circuit network visualization | Fig 4 |
| 09 | Mutation bootstrap analysis | Fig S |
| 10 | Positive control circuits (neurotransmitter systems) | Fig S |
| 11 | Gene clustering and mutation rate analysis | Fig S |
| 12 | Cross-species validation vs Buch et al. human fMRI | Fig S |
| 13 | Mouse model fMRI validation | Fig S |

### Cell-Type Level (`notebooks_mouse_sc/`)

Pipeline for Figure 5. See `notebooks_mouse_sc/README.md` for details.

| Notebook | Description | Manuscript |
|----------|-------------|------------|
| 01 | Cluster Z1 → Z2 expression specificity | Methods |
| 02 | DN gene weights, ASD/DDD bias, residual analysis | Fig 5 |
| 03 | MERFISH preprocessing (ISH mapping, Z1, Z2) | Methods |
| 04 | MERFISH structure-level bias (4 methods) | Fig 5A |
| 05 | Sibling null QQ plots | Fig 5B |
| 06 | Spatial bias visualization on brain sections | Fig 5 |
| 07 | Figure 5 panels (scatter, QQ, dotplot, connectivity) | Fig 5 |

### Figures & Tables (`notebooks_figures/`)

| Notebook | Description |
|----------|-------------|
| Figures_Tables | Main manuscript figures (Figs 1–5) |
| SupplementaryTables | Supplementary tables for publication |

## Notebook Conventions

- All notebooks are paired with `.py` files via [jupytext](https://jupytext.readthedocs.io/) — edit `.py` files, then `jupytext --sync`
- Outputs are stripped from `.ipynb` on commit via [nbstripout](https://github.com/kynan/nbstripout)
- Data paths are loaded from `config/config.yaml` (not hardcoded)
- Transparent figure backgrounds for compositing
- Conda environment: `gencic` (Python 3.10)

## Configuration

All data file paths, gene sets, and pipeline parameters are centralized in `config/config.yaml`. Gene sets are defined under `gene_sets:` with paths to their weight files and null model type (random or mutability-based sibling).

See `DATA_MANIFEST.yaml` (project root) and `notebooks_mouse_sc/DATA_MANIFEST.yaml` for detailed documentation of every data file dependency.
