# GENCIC: Genetics and Expression iNtegration with Connectome to Infer affected Circuits

## Overview
This repository contains the computational framework GENCIC (Genetics and Expression iNtegration with Connectome to Infer affected Circuits), which identifies brain circuits preferentially targeted by autism spectrum disorder (ASD) mutations. GENCIC integrates genomic data from ASD patients with spatial transcriptomics and brain-wide connectome data to discover functionally cohesive neural circuits affected in autism.

## Citation
If you use this code or methodology in your research, please cite:
Wang J, Chang J, Chiang AH, Vitkup D. Autism Mutations Preferentially Target Distributed Brain Circuits Associated with Sensory Integration and Decision Making. [Journal Name]. 2025.

## Requirements
- Python 3.7.0+
- R 4.2.1+
- Required Python packages:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - statsmodels
  - networkx
- Required R packages:
  - dplyr
  - ggplot2
  - data.table
  - tidyr

## Data Sources
GENCIC integrates data from the following sources:

1. **Genetic Data**:
   - SPARK project genetic data (de novo mutations in ASD probands and siblings)
   - Available through SFARI Base (https://base.sfari.org/)

2. **Mouse Brain Atlas Data**:
   - Allen Mouse Brain Atlas gene expression data (https://mouse.brain-map.org/)
   - Allen Mouse Brain Connectivity Atlas (Oh et al., 2014)
   - Allen Brain Cell Atlas 10X RNA-sequencing and MERFISH spatial transcriptomics data

Key intermediate data for running GENCIC is available at:
https://vitkuplab.c2b2.columbia.edu/Gencic/index.html

## Repository Structure

```
ASD_Circuits_CellType/
├── config/                          # Configuration files for Snakemake workflows
│   ├── config.yaml                  # Main bias calculation config
│   ├── config.SC.DN.yaml           # Single-cell de novo analysis config
│   ├── circuit_config.yaml          # Circuit search config
│   └── circuit_config_bootstrap.yaml # Bootstrap analysis config
│
├── src/                             # Core Python modules
│   ├── ASD_Circuits.py             # Main analysis functions (bias calculation, gene loading)
│   ├── CellType_PSY.py             # Cell type-specific analysis functions
│   ├── SA.py                       # Simulated annealing for circuit optimization
│   └── SA_optimized.py             # Optimized SA implementation
│
├── scripts/                         # Analysis scripts
│   ├── script_bias_cal.py          # Calculate bias with p-values
│   ├── script_generate_geneweights.py  # Generate random gene weights for null
│   ├── script_run_ctrl_sim.py      # Run null distribution simulations
│   ├── script_circuit_search.SI.py # Circuit search using simulated annealing
│   └── workflow/                   # Workflow helper scripts
│       ├── generate_bias_limits.py # Generate Pareto front bias limits
│       ├── run_sa_search.py        # Run SA search for circuits
│       ├── create_pareto_front.py  # Create Pareto front profiles
│       └── aggregate_bootstrap_results.py  # Bootstrap aggregation
│
├── notebooks_mouse_str/             # Brain structure-level analysis notebooks
│   ├── 01.Download_ISH_data.ipynb  # Download Allen ISH data
│   ├── 02.Preprocessing_ISH_data.ipynb  # Preprocess expression data
│   ├── 03.Preprocessing_Connectivity_data.ipynb  # Preprocess connectivity
│   ├── 04.Weighted_ASD_bias.ipynb  # Calculate ASD mutation bias
│   ├── 05.circuit_search.ipynb     # GENCIC circuit search
│   ├── 06.Phenotype_Analysis_seperating_HIQ_LIQ.ipynb  # IQ stratification
│   └── 07.Stratified_distance_analysis.ipynb  # Spatial analysis
│
├── notebooks_mouse_sc/              # Cell type-level analysis notebooks
│   ├── ABC_Bias_Cal.ipynb          # Allen Brain Cell Atlas bias calculation
│   ├── ASD_SC_MERFISH_Bias.ipynb   # MERFISH spatial transcriptomics bias
│   ├── BiasDisplay.ipynb           # Visualize cell type biases
│   └── Figures.ipynb               # Generate cell type figures
│
├── notebooks_figures/               # Manuscript figure generation notebooks
├── notebook_rebuttal/              # Additional reviewer-requested analyses
│
├── Snakefile.bias                  # Snakemake workflow for bias calculation
├── Snakefile.circuit               # Snakemake workflow for circuit search
└── Snakefile.circuit.bootstrap     # Snakemake workflow for bootstrap analysis
```

## Analysis Workflows

GENCIC uses Snakemake to orchestrate the analysis pipeline. There are three main workflows:

### 1. Bias Calculation Workflow (`Snakefile.bias`)

Calculates ASD mutation biases for brain structures or cell types with statistical significance:

- **Step 1**: Generate random gene weights for null distributions
- **Step 2**: Calculate null bias distributions (sibling & random genes)
- **Step 3**: Calculate final bias scores with p-values

**Usage:**
```bash
# Run full bias analysis pipeline
snakemake -s Snakefile.bias --configfile config/config.yaml --cores 10

# Run specific gene set analysis
snakemake -s Snakefile.bias --configfile config/config.yaml --cores 10 \
    results/STR_ISH/ASD_All_bias_addP_sibling.csv
```

### 2. Circuit Search Workflow (`Snakefile.circuit`)

Identifies brain circuits using simulated annealing along Pareto fronts:

- **Step 1**: Generate bias limits for circuit sizes
- **Step 2**: Run simulated annealing to find optimal circuits
- **Step 3**: Extract best circuits and create Pareto fronts

**Usage:**
```bash
# Run circuit search for all datasets
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10

# Run for specific dataset
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml \
    --config dataset=ASD_All --cores 10
```

### 3. Bootstrap Analysis Workflow (`Snakefile.circuit.bootstrap`)

Performs bootstrap resampling for circuit robustness analysis.

**Usage:**
```bash
snakemake -s Snakefile.circuit.bootstrap \
    --configfile config/circuit_config_bootstrap.yaml --cores 10
```

## Usage

### Setting up the environment
```bash
# Clone the repository
git clone https://github.com/explorerwjy/ASD_Circuits.git
cd ASD_Circuits_CellType

# Create and activate conda environment
conda create -n gencic python=3.7
conda activate gencic

# Install dependencies
pip install -r requirements.txt
pip install snakemake
```

### Quick Start

**1. Calculate ASD mutation biases:**
```bash
# Configure your gene sets and expression matrices in config/config.yaml
# Then run the bias calculation pipeline
snakemake -s Snakefile.bias --configfile config/config.yaml --cores 10
```

**2. Run GENCIC circuit search:**
```bash
# Configure bias files and circuit parameters in config/circuit_config.yaml
# Then run the circuit search pipeline
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10
```

**3. Explore results in notebooks:**
```bash
jupyter notebook notebooks_mouse_str/05.circuit_search.ipynb
```

### Manual Script Execution

You can also run individual analysis steps:

```bash
# Generate random gene weights for null distribution
python scripts/script_generate_geneweights.py \
    --WeightDF data/gene_weights.csv \
    --SpecMat data/expression_matrix.parquet \
    --n_sims 10000 \
    --outfile results/random_weights.csv

# Calculate null bias distribution
python scripts/script_run_ctrl_sim.py \
    --SpecMat data/expression_matrix.parquet \
    --outfile results/null_bias.parquet \
    --mode STR \
    --Ctrl_Genes_Fil results/random_weights.csv \
    --n_processes 10

# Calculate final bias with p-values
python scripts/script_bias_cal.py \
    --SpecMat data/expression_matrix.parquet \
    --gw data/gene_weights.csv \
    --mode STR \
    --geneset ASD_All \
    --Bias_Null results/null_bias.parquet \
    --Bias_Out results/bias_final.csv

# Run circuit search
python scripts/script_circuit_search.SI.py \
    --size 46 \
    --min_bias 0.3 \
    --bias_file results/bias_final.csv \
    --weight_mat data/connectivity_matrix.csv \
    --info_mat data/info_matrix.csv \
    --output results/circuits.txt
```

## Configuration

### Analysis Types

GENCIC supports two main analysis modes configured in `config/config.yaml`:

- **`mouse_str_bias`**: Brain structure-level analysis using Allen Mouse Brain ISH data
  - Expression matrix: Allen Mouse Brain structural expression data
  - Output: Bias scores for 213 brain structures

- **`mouse_ct_bias`**: Cell type-level analysis using single-cell/spatial transcriptomics
  - Expression matrix: Cell type specificity data from Allen Brain Cell Atlas
  - Output: Bias scores for cell types/clusters

### Gene Sets

Configure gene sets in `config/config.yaml` under `gene_sets`:

```yaml
gene_sets:
  ASD_All:
    geneweights: "path/to/asd_gene_weights.csv"
    description: "All ASD de novo mutations"
  ASD_LGD:
    geneweights: "path/to/asd_lgd_weights.csv"
    description: "ASD likely gene-damaging mutations"
```

### Circuit Search Parameters

Configure in `config/circuit_config.yaml`:

- `circuit_sizes`: List of circuit sizes to search (e.g., [46, 50, 60])
- `top_n`: Number of top-biased structures to consider (default: 213)
- `sa_runtimes`: Number of simulated annealing runs per bias limit
- `sa_steps`: Number of optimization steps per SA run
- `measure`: Optimization measure ("SI" for self-information or "Connectivity")

## Notebook Organization

### Structure-Level Analysis (`notebooks_mouse_str/`)

Numbered notebooks follow the analysis workflow:

1. **01-04**: Data preprocessing (download, ISH data, connectivity data, bias calculation)
2. **05**: GENCIC circuit search and Pareto front generation
3. **06**: Phenotype analysis (IQ stratification)
4. **07**: Spatial analysis (distance stratification)

Additional analysis notebooks explore specific aspects (connectome properties, community structure, etc.)

### Cell Type Analysis (`notebooks_mouse_sc/`)

- **ABC_Bias_Cal.ipynb**: Calculate bias for Allen Brain Cell Atlas cell types
- **ASD_SC_MERFISH_Bias.ipynb**: MERFISH spatial transcriptomics analysis
- **BiasDisplay.ipynb**: Visualize and compare cell type biases
- **Figures.ipynb**: Generate manuscript figures for cell type analysis

### Figure Generation (`notebooks_figures/`)

Notebooks for generating publication-quality figures

### Rebuttal Analysis (`notebook_rebuttal/`)

Additional analyses requested during peer review (e.g., fMRI validation)

