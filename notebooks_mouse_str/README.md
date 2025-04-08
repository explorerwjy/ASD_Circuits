# Mouse Striatum Circuit Analysis Notebooks

This repository contains Jupyter notebooks for analyzing bias and circuit patterns in mouse striatum data.

## Notebooks Overview

### 1. Download ISH data from Allen Mouse Brain Atlas
- `01.Download_ISH_data.ipynb`: Download ISH data from Allen Mouse Brain Atlas and preprocess it for further analysis
  - Download structure level ISH data


### 2. Compute Mouse Strcture Expression level and "Z2" specificity score
- `02.Preprocessing_ISH_data.ipynb`: Download ISH data from Allen Mouse Brain Atlas and preprocess it for further analysis
  - Aggreate ISH data to structure level
  - Produce whole brain expression level for matching
  - Compute "Z2" specificity score

### 3. Compute Mouse Strcture Expression level and "Z2" specificity score
- `03.Preprocessing_Connectivity_data.ipynb`: Download ISH data from Allen Mouse Brain Atlas and preprocess it for further analysis
  - Input/output connection biases
  - Spatial distribution analysis
  - Cell type specific biases

### 4. Compute ASD biases for mouse brain structures based on ISH data
- `04.Weighted_ASD_bias.ipynb`: Identifies and analyzes circuit motifs
  - Common connectivity patterns
  - Feedforward and feedback loops
  - Cell-type specific circuit patterns
  - Statistical significance testing


### 5. Compute ASD biases for mouse brain structures based on ISH data
- `05.circuit_search.ipynb`: Identifies and analyzes circuit motifs
  - Common connectivity patterns
  - Feedforward and feedback loops
  - Cell-type specific circuit patterns
  - Statistical significance testing

### Other Notebooks
- `06.Phenotype_Analysis_seperating_HIQ_LIQ.ipynb`: Phenotype Analysis seperateing HIQ/LIQ
- `07.Stratified_distance_analysis.ipynb`: Stratified distance analysis

