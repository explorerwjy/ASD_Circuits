# Mouse Brain Structure-Level Analyses

Notebooks for analyzing ASD mutation bias at the level of mouse brain structures,
using Allen Brain Atlas ISH expression and connectivity data.

## Main Pipeline (Numbered)

These notebooks form the core analysis pipeline and should be run in order.
Notebooks 01-07 have been reworked for reproducibility: paired with `.py` files
via jupytext, dead code removed, paths made relative, and slow steps cached.

| Notebook | Description |
|----------|-------------|
| **01.Download_ISH_data** | Download in-situ hybridization (ISH) expression data from the Allen Brain Atlas API for ASD-associated genes. |
| **02.Preprocessing_ISH_data** | Preprocess downloaded ISH data into structure-level expression matrices (Z-scored, filtered). |
| **03.Preprocessing_Connectivity_data** | Process Allen Mouse Brain connectivity data into scoring matrices (WeightMat, InfoMat) with distance-based information content. |
| **04.Weighted_ASD_bias** | Compute mutation-weighted expression bias per brain structure and assess significance against null distributions (random and sibling controls). |
| **05.circuit_search** | Run simulated annealing to find circuits (structure sets) that maximize both ASD bias and connectivity. Pareto front analysis. |
| **06.Phenotype_Analysis_seperating_HIQ_LIQ** | Stratify ASD cases by IQ and sex, compare structural bias profiles between groups using bootstrap resampling and permutation tests. |
| **07.Stratified_distance_analysis** | Compare distance-stratified connectivity of ASD circuits vs sibling controls (top-bias structures and SA-derived circuits). |

## Exploratory / Development Notebooks

Older notebooks used during development and exploration. These have not been
reworked and may contain hardcoded paths or depend on deprecated functions.

| Notebook | Description |
|----------|-------------|
| CircuitsInformationScore | Circuit connectivity analysis using Shannon information scores. |
| Connectome | Allen Mouse Brain connectome loading and connectivity matrix generation. |
| debug_SA | Debugging simulated annealing cache desynchronization issues. |
| GENCIC | End-to-end GENCIC algorithm demo: bias limits, circuit search, ranking. |
| ipsi_vs_contra_weights | Comparison of ipsilateral vs contralateral connection weights. |
| NoteBook_Allen_Mouse_Connectivity | Exploratory queries of Allen connectivity data for specific structures. |
| NoteBook_CheckMutaionRate | Correlation between mutation rates and counts in sibling data. |
| NoteBook_Community_of_Circuits | Graph clustering to detect circuit community structure. |
| NoteBook_Compare_SA_SI_Conn | Comparison of circuit scoring methods across distance measures. |
| NoteBook_DistanceRangeAnalysis | Earlier version of distance-stratified connectivity analysis (superseded by 07). |
| NoteBook_N_ASD_Genes | ASD bias and network properties across varying gene set sizes. |
| NoteBook_Rank_vs_InfoContentScore | Relationship between structure ranking and information content scores. |
| NoteBook_SI_Score_Drop_Contra_wo_mirror | SI circuit scores excluding contralateral non-mirrored connections. |
| NoteBooks_topNvsCohesiveness | Circuit cohesiveness vs number of top-ranked structures. |
| Optimized_Circuits_Connectivity | Network connectivity metrics for optimized 46-structure ASD circuits. |
| Optimized_Circuits_Information_Score | Shannon Information scores for optimized circuits at different distance ranges. |
| Phenotype_Graph_New | High/low IQ ASD mapping onto optimized circuits (superseded by 06). |
| Preprocessing | Raw connectome processing and distance-based splitting (superseded by 03). |
| SA | Simulated annealing implementation and testing (superseded by 05). |
| Sibling_SI_score_significance | SI score significance testing against sibling control null distributions. |
| Test_Vlidation_fMRI | Validation of ASD circuit predictions using mouse fMRI data. |
| WeighedBias | Weighted expression bias with alternative weighting schemes (superseded by 04). |

## Data Dependencies

Key input data used by the main pipeline:

- `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` -- Structure-level Z2 expression matrix (genes x 213 structures)
- `dat/allen-mouse-conn/ConnectomeScoringMat/` -- WeightMat and InfoMat for ipsilateral connectivity
- `dat/allen-mouse-conn/Dist_CartesianDistance.csv` -- Pairwise structure distance matrix
- `dat/Genetics/GeneWeights/` -- Gene weight files (.gw format)
- `dat/Unionize_bias/` -- Bias results, null distributions, sibling controls, bootstrap data

## Cached Results

Slow computation steps cache results for fast re-execution:

- `results/Distance_analysis/subsib_distance_counts.npz` -- 10K subsampled sibling distance-binned counts (notebook 07)
- `results/Distance_analysis/sa_sib_distance_counts_bias0.37.npz` -- SA sibling circuit distance-binned counts (notebook 07)
- `results/Phenotype_bootstrap/` -- Bootstrap resampling results as parquet files (notebook 06)
- `results/Phenotype_permutation/` -- Permutation test results (notebook 06)
