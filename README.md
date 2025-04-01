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
https://vitkuplab.c2b2.columbia.edu/gencic/index.html

## Repository Structure
## Pipeline Overview

### 1. Data Preprocessing
- `preprocess_spark_genetic_data.py`: Process genetic data from SPARK study
- `preprocess_expression_data.py`: Process Allen Mouse Brain Atlas expression data
- `preprocess_connectome_data.py`: Process Allen Mouse Brain Connectivity Atlas data
- `preprocess_abc_atlas_data.py`: Process Allen Brain Cell Atlas single-cell data

### 2. Core Analysis
- `calculate_mutation_biases.py`: Calculate ASD mutation biases for brain structures
- `calculate_circuit_connectivity.py`: Calculate circuit connectivity scores (CCS)
- `run_gencic_search.py`: Perform GENCIC simulated annealing search for optimal circuits
- `generate_pareto_front.py`: Generate and evaluate Pareto fronts
- `analyze_long_distance_connections.py`: Analyze spatial distribution of circuit connections
- `cell_type_analysis.py`: Analyze ASD biases towards specific cell types

### 3. Cognitive Phenotype Analysis
- `analyze_iq_stratification.py`: Analyze ASD mutation biases stratified by IQ
- `calculate_phenotype_bias_sensitivity.py`: Calculate PBS scores for cell types

### 4. Visualization
- `plot_circuit_network.py`: Generate network visualizations of brain circuits
- `plot_mutation_biases.py`: Plot mutation biases across brain structures
- `plot_cell_type_biases.py`: Visualize cell type-specific biases
- `plot_brain_sections.py`: Generate brain section visualizations with spatial expression data

## Usage

### Setting up the environment
```bash
# Clone the repository
git clone https://github.com/explorerwjy/ASD_Circuits.git
cd ASD_Circuits

# Create and activate a virtual environment
conda create -n gencic python=3.7
conda activate gencic

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocess/preprocess_data.py

# Calculate ASD mutation biases
python src/analysis/calculate_mutation_biases.py

# Run GENCIC search for optimal circuits
python src/analysis/run_gencic_search.py --size 46 --min_bias 0.1

# Generate visualizations
python src/visualization/plot_circuit_network.py --output figures/asd_circuit.png

