# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: gencic
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/" # Change to your project directory
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *
import scipy.io as sio
from scipy.stats import spearmanr

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")

HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()


# %% [markdown]
# # Mouse fMRI data validation from 16 mouse model figure 4

# %%
def permutation_test_overlap(set1_range, set2_range, set1_size, set2_size, observed_overlap, n_permutations=10000):
    """
    Perform a permutation test to assess the significance of overlap between two sets.
    
    Parameters:
    -----------
    set1_range : array-like
        Range of possible values for set 1 (e.g., np.arange(1, 164))
    set2_range : array-like
        Range of possible values for set 2 (e.g., np.arange(1, 214))
    set1_size : int
        Size of set 1 sample
    set2_size : int
        Size of set 2 sample
    observed_overlap : int
        The observed overlap to test against
    n_permutations : int, default=10000
        Number of permutations to perform
    
    Returns:
    --------
    dict : Dictionary containing test results
    """
    
    # Generate all random samples at once
    set1_samples = np.array([np.random.choice(set1_range, size=set1_size, replace=False) for _ in range(n_permutations)])
    set2_samples = np.array([np.random.choice(set2_range, size=set2_size, replace=False) for _ in range(n_permutations)])

    # Vectorized intersection calculation
    intersections = np.array([len(np.intersect1d(set1_samples[i], set2_samples[i])) for i in range(n_permutations)])

    # Calculate p-value for overlap >= observed_overlap
    p_value = np.sum(intersections >= observed_overlap) / len(intersections)
    
    # Prepare results
    results = {
        'intersections': intersections,
        'mean_intersection': np.mean(intersections),
        'std_intersection': np.std(intersections),
        'min_intersection': np.min(intersections),
        'max_intersection': np.max(intersections),
        'observed_overlap': observed_overlap,
        'p_value': p_value,
        'n_significant': np.sum(intersections >= observed_overlap),
        'n_permutations': n_permutations
    }
    
    return results

def plot_permutation_results(results):
    """
    Plot the results of a permutation test.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from permutation_test_overlap function
    """
    intersections = results['intersections']
    observed_overlap = results['observed_overlap']
    mean_intersection = results['mean_intersection']
    n_permutations = results['n_permutations']
    
    plt.figure(figsize=(10, 6))
    plt.hist(intersections, bins=range(min(intersections), max(intersections)+2), 
             alpha=0.7, edgecolor='black', density=True)
    plt.axvline(observed_overlap, color='red', linestyle='--', linewidth=2, 
               label=f'Observed overlap = {observed_overlap}')
    plt.axvline(mean_intersection, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_intersection:.1f}')
    plt.xlabel('Intersection Length')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of Intersections from {n_permutations} Permutations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# %%
FMRI = pd.read_excel(os.path.join(ProjDIR, "dat/Clusters_Values.xlsx"), index_col="Name")

# %%
FMRI.head(10)

# %%
# Compute Correlation between 4 Clusters, and top STR in common
# Compute Spearman and Pearson correlation between Cluster1-4
cluster_cols = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']
cluster_data = FMRI[cluster_cols]

# Calculate Spearman and Pearson correlation matrices
spearman_corr = cluster_data.corr(method='spearman')
pearson_corr = cluster_data.corr(method='pearson')

# Import matplotlib with explicit backend setting to avoid backend_bases error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots for both correlation matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Spearman correlation heatmap
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax1)
ax1.set_title('Spearman Correlation between Clusters 1-4')

# Pearson correlation heatmap
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax2)
ax2.set_title('Pearson Correlation between Clusters 1-4')

plt.tight_layout()
plt.show()

# Display the correlation matrices
print("Spearman Correlation Matrix:")
print(spearman_corr.round(3))
print("\nPearson Correlation Matrix:")
print(pearson_corr.round(3))

# %%
GENCIC = pd.read_excel(os.path.join(ProjDIR, "results/SupTabs.v57.xlsx"), sheet_name="Table-S1- Structure Bias", index_col=0)
# Need Annotate Name


# %%
GENCIC.head(2)

# %%

# %%
# TODO: copy to dat/
ABA_Ontology = pd.read_csv(os.path.join(ProjDIR, "dat/Other/ontology.csv"), index_col="KEY")
ABA_Ontology.head(3)

# %%
for _str, row in GENCIC.iterrows():
    if _str in ABA_Ontology.index:
        GENCIC.loc[_str, "acronym"] = ABA_Ontology.loc[_str, "acronym"]
    else:
        print(f"{_str} not in ABA_Ontology")
GENCIC["Structure"] = GENCIC.index
GENCIC = GENCIC.set_index("acronym")
GENCIC.to_csv('../results/GENCIC_MouseSTRBias.csv')

# %%
GENCIC.head(5)

# %%
from scipy.stats import spearmanr, pearsonr

for cluster in ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']:
    # Get common acronyms between GENCIC and FMRI data
    common_acronyms = GENCIC.index.intersection(FMRI.index)
    
    if len(common_acronyms) > 0:
        # Extract values for common acronyms
        gencic_bias = GENCIC.loc[common_acronyms, 'Bias']
        fmri_cluster = FMRI.loc[common_acronyms, cluster]
        
        # Calculate Spearman correlation
        spearman_correlation, spearman_p_value = spearmanr(gencic_bias, fmri_cluster)
        
        # Calculate Pearson correlation
        pearson_correlation, pearson_p_value = pearsonr(gencic_bias, fmri_cluster)
        
        print(f"{cluster}:")
        print(f"  Number of common regions: {len(common_acronyms)}")
        print(f"  Spearman correlation: {spearman_correlation:.4f} (p = {spearman_p_value:.4f})")
        print(f"  Pearson correlation: {pearson_correlation:.4f} (p = {pearson_p_value:.4f})")
        print()
    else:
        print(f"{cluster}: No common acronyms found between GENCIC and FMRI data")
        print()


# %%
from scipy.stats import spearmanr, pearsonr

for cluster in ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']:
    # Get common acronyms between GENCIC and FMRI data
    common_acronyms = GENCIC.index.intersection(FMRI.index)
    
    if len(common_acronyms) > 0:
        # Extract values for common acronyms
        gencic_bias = GENCIC.loc[common_acronyms, 'Bias']
        fmri_cluster = FMRI.loc[common_acronyms, cluster].abs()
        
        # Calculate Spearman correlation
        spearman_correlation, spearman_p_value = spearmanr(gencic_bias, fmri_cluster)
        
        # Calculate Pearson correlation
        pearson_correlation, pearson_p_value = pearsonr(gencic_bias, fmri_cluster)
        
        print(f"{cluster}:")
        print(f"  Number of common regions: {len(common_acronyms)}")
        print(f"  Spearman correlation: {spearman_correlation:.4f} (p = {spearman_p_value:.4f})")
        print(f"  Pearson correlation: {pearson_correlation:.4f} (p = {pearson_p_value:.4f})")
        print()
    else:
        print(f"{cluster}: No common acronyms found between GENCIC and FMRI data")
        print()


# %%
from scipy.stats import hypergeom

topN = 50

for cluster in ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']:
    # Get common acronyms between GENCIC and FMRI data
    common_acronyms = GENCIC.index.intersection(FMRI.index)
    
    if len(common_acronyms) > 0:
        # Get top N structures from GENCIC (highest Bias values)
        gencic_topN = GENCIC.loc[common_acronyms].nlargest(topN, 'Bias').index
        
        # Get top N structures from fMRI cluster (highest values)
        fmri_topN_largest = FMRI.loc[common_acronyms].nlargest(topN, cluster).index
        
        # Get top N structures from fMRI cluster (lowest values)
        fmri_topN_smallest = FMRI.loc[common_acronyms].nsmallest(topN, cluster).index
        
        # Calculate overlap between GENCIC top N and fMRI top N (largest)
        overlap_largest = len(set(gencic_topN).intersection(set(fmri_topN_largest)))
        
        # Calculate overlap between GENCIC top N and fMRI top N (smallest)
        overlap_smallest = len(set(gencic_topN).intersection(set(fmri_topN_smallest)))
        
        # Calculate p-values using hypergeometric test
        # For largest values
        # Population size: total common regions
        # Successes in population: fMRI top N largest
        # Sample size: GENCIC top N
        # Observed successes: overlap_largest
        pval_largest = hypergeom.sf(overlap_largest - 1, len(common_acronyms), topN, topN)
        
        # For smallest values
        pval_smallest = hypergeom.sf(overlap_smallest - 1, len(common_acronyms), topN, topN)
        
        print(f"{cluster}:")
        print(f"  Number of common regions: {len(common_acronyms)}")
        print(f"  GENCIC top {topN} overlap with fMRI top {topN} (largest): {overlap_largest} (p = {pval_largest:.4f})")
        print(f"  GENCIC top {topN} overlap with fMRI top {topN} (smallest): {overlap_smallest} (p = {pval_smallest:.4f})")
        print()
    else:
        print(f"{cluster}: No common acronyms found between GENCIC and FMRI data")
        print()

# %%
# Add average and count columns to FMRI dataframe
FMRI['Average_Clusters'] = FMRI[['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']].mean(axis=1)
FMRI.head(5)

# %%
for cluster in ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Average_Clusters']:
    # Get common acronyms between GENCIC and FMRI data
    common_acronyms = GENCIC.index.intersection(FMRI.index)
    
    if len(common_acronyms) > 0:
        # Extract values for common acronyms
        gencic_bias = GENCIC.loc[common_acronyms, 'Bias']
        fmri_cluster = FMRI.loc[common_acronyms, cluster]
        
        # Calculate Spearman correlation
        correlation, p_value = spearmanr(gencic_bias, fmri_cluster)
        
        print(f"{cluster}:")
        print(f"  Number of common regions: {len(common_acronyms)}")
        print(f"  Spearman correlation: {correlation:.4f}")
        print(f"  P-value: {p_value:.8f}")
        print()
    else:
        print(f"{cluster}: No common acronyms found between GENCIC and FMRI data")
        print()

# %% [markdown]
# ### Sibling-Based Permutation P-Values
# Standard correlation p-values assume independence between brain structures.
# To address potential spatial autocorrelation, we compute empirical p-values
# from the sibling mutability null distribution (10,000 simulated bias profiles).

# %%
from scipy.stats import pearsonr, linregress, rankdata

# Load sibling null bias and map structure names to acronyms
sib_null = pd.read_parquet(os.path.join(ProjDIR, 'results/Sibling_bias/Mutability_61gene/sibling_mutability_bias.parquet'))
str2acr = {s: ABA_Ontology.loc[s, 'acronym'] for s in sib_null.index if s in ABA_Ontology.index}
sib_null_acr = sib_null.rename(index=str2acr)

# Common acronyms between GENCIC, FMRI, and sibling null
common_acr = GENCIC.index.intersection(FMRI.index).intersection(sib_null_acr.index)
X_null = sib_null_acr.loc[common_acr].values  # (N_common, 10000)
Y_fmri_null = FMRI.loc[common_acr, 'Average_Clusters'].values
X_obs_null = GENCIC.loc[common_acr, 'Bias'].values

# Observed correlations
obs_pear_r, obs_pear_p = pearsonr(X_obs_null, Y_fmri_null)
obs_spear_rho, obs_spear_p = spearmanr(X_obs_null, Y_fmri_null)

# Vectorized Pearson null correlations
n_sims = X_null.shape[1]
Y_c = Y_fmri_null - Y_fmri_null.mean()
Y_ss = np.sum(Y_c**2)
X_means = X_null.mean(axis=0)
X_c = X_null - X_means[np.newaxis, :]
X_ss = np.sum(X_c**2, axis=0)
null_pearson_r = (Y_c @ X_c) / np.sqrt(X_ss * Y_ss)

# Spearman null correlations (rank-based)
Y_ranks = rankdata(Y_fmri_null)
null_spearman_rho = np.empty(n_sims)
for i in range(n_sims):
    null_spearman_rho[i] = np.corrcoef(rankdata(X_null[:, i]), Y_ranks)[0, 1]

# One-tailed permutation p-values (testing negative correlation)
p_perm_pearson = (np.sum(null_pearson_r <= obs_pear_r) + 1) / (n_sims + 1)
p_perm_spearman = (np.sum(null_spearman_rho <= obs_spear_rho) + 1) / (n_sims + 1)
# Two-tailed alternative:
# p_perm_pearson = (np.sum(np.abs(null_pearson_r) >= np.abs(obs_pear_r)) + 1) / (n_sims + 1)
# p_perm_spearman = (np.sum(np.abs(null_spearman_rho) >= np.abs(obs_spear_rho)) + 1) / (n_sims + 1)

print(f"Sibling Permutation P-values (N={len(common_acr)} structures, {n_sims} sims):")
print(f"  Pearson:  r = {obs_pear_r:.4f}, p = {obs_pear_p:.4e}, p_perm = {p_perm_pearson:.4f}")
print(f"  Spearman: rho = {obs_spear_rho:.4f}, p = {obs_spear_p:.4e}, p_perm = {p_perm_spearman:.4f}")

# %%
# Get common acronyms between GENCIC and FMRI data
common_acronyms = GENCIC.index.intersection(FMRI.index)

if len(common_acronyms) > 0:
    # Extract values for common acronyms
    gencic_bias = GENCIC.loc[common_acronyms, 'Bias']
    fmri_average_clusters = FMRI.loc[common_acronyms, 'Average_Clusters']
    regions = GENCIC.loc[common_acronyms, 'REGION']

    # Calculate Pearson correlation and linear regression
    linres = linregress(gencic_bias, fmri_average_clusters)
    pear_r, pear_p = pearsonr(gencic_bias, fmri_average_clusters)

    import numpy as np
    import matplotlib.pyplot as plt

    # Define region color mapping (from DDD.ipynb)
    REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                    'Hippocampus','Amygdala','Striatum', 
                    "Thalamus", "Hypothalamus", "Midbrain", 
                    "Medulla", "Pallidum", "Pons", 
                    "Cerebellum"]
    REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                        "#2c9d39", "#742eb5", "#ed8921", 
                                        "#e82315", "#E6B0AA", "#f6b26b",  
                                        "#20124d", "#2ECC71", "#D2B4DE", 
                                        "#ffd966"]))
    
    # Map regions to colors
    # Handle naming differences between GENCIC and color map
    region_name_mapping = {
        'Amygdalar': 'Amygdala',
        'Olfactory': 'Olfactory_areas',
        'Hippocampal': 'Hippocampus'
    }
    
    colors = []
    for region in regions:
        # Apply name mapping if needed
        mapped_region = region_name_mapping.get(region, region)
        # Get color from dictionary, default to gray if not found
        colors.append(REG_COR_Dic.get(mapped_region, '#808080'))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=360)

    # Plot linear fit first (behind points)
    xfit = np.linspace(gencic_bias.min(), gencic_bias.max(), 100)
    yfit = linres.slope * xfit + linres.intercept
    ax.plot(
        xfit, yfit, color="#e6550d", linestyle="--", linewidth=2.2,
        label=f"Linear fit", zorder=1
    )

    # Create scatter plot with region colors
    for region_name in set(regions):
        mapped_region_name = region_name_mapping.get(region_name, region_name)
        color = REG_COR_Dic.get(mapped_region_name, '#808080')
        mask = regions == region_name
        ax.scatter(
            gencic_bias[mask], fmri_average_clusters[mask],
            color=color, edgecolor='k', s=80, alpha=0.9, zorder=2,
            label=region_name
        )

    # Expand x-axis to the right to give labels more room
    #x_lo, x_hi = ax.get_xlim()
    #ax.set_xlim(x_lo, x_hi + (x_hi - x_lo) * 0.15)

    # Add text annotations: ASD 46 circuit structures (blue) and high-fMRI non-circuit (green)
    from adjustText import adjust_text
    circuit_46 = set(GENCIC[GENCIC["Circuits.46"] == 1].index)
    texts = []
    for acronym in common_acronyms:
        x = gencic_bias[acronym]
        y = fmri_average_clusters[acronym]
        if acronym in circuit_46:
            texts.append(ax.text(x, y, acronym, fontsize=14, fontweight='bold',
                                 alpha=0.9, color='blue'))
        elif y <= -0.2:
            texts.append(ax.text(x, y, acronym, fontsize=14, fontweight='bold',
                                 alpha=0.9, color='green'))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel("ASD Mutation Bias", fontsize=17, fontweight='bold')
    plt.ylabel("Mean fMRI connectivity alterations", fontsize=17, fontweight='bold')

    # Show Pearson correlation in annotation, with a rectangular box around it
    stat_text = (
        f"r = {pear_r:.2f}\np = {pear_p:.0e}\n$p_{{perm}}$ = {p_perm_pearson:.4f}"
    )
    plt.text(
        0.12, 0.2, stat_text,
        transform=plt.gca().transAxes,
        fontsize=25,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.85)
    )

    # plt.title(
    #     "GENCIC Bias vs FMRI Average Clusters",
    #     fontsize=16,
    #     fontweight='bold'
    # )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, loc='best', frameon=True, ncol=2)
    plt.grid(alpha=0.3, linestyle=":", zorder=0)
    plt.tight_layout()
    plt.show()

    print(f"Number of common regions: {len(common_acronyms)}")
    print(f"Pearson correlation: {pear_r:.4f}")
    print(f"Pearson P-value: {pear_p:.8f}")
else:
    print("No common acronyms found between GENCIC and FMRI data")

# %%
# Create combined table with complete fMRI data and GENCIC bias for supplementary materials
# Get common acronyms between GENCIC and FMRI data
common_acronyms = GENCIC.index.intersection(FMRI.index)

# Create combined dataframe
combined_table = FMRI.loc[common_acronyms].copy()

# Add GENCIC Bias column
combined_table['GENCIC_Bias'] = GENCIC.loc[common_acronyms, 'Bias']

# Add other relevant GENCIC columns if needed (e.g., Structure name, REGION)
if 'Structure' in GENCIC.columns:
    combined_table['GENCIC_Structure'] = GENCIC.loc[common_acronyms, 'Structure']
if 'REGION' in GENCIC.columns:
    combined_table['GENCIC_REGION'] = GENCIC.loc[common_acronyms, 'REGION']

# Reorder columns to put GENCIC_Bias near the beginning for easier viewing
cols = combined_table.columns.tolist()
if 'GENCIC_Bias' in cols:
    cols.remove('GENCIC_Bias')
    cols.insert(1, 'GENCIC_Bias')  # Insert after index/name column
combined_table = combined_table[cols]

# Save to file for supplementary materials
output_file = os.path.join(ProjDIR, 'results/FMRI_GENCIC_Combined_Table.csv')
combined_table.to_csv(output_file)
print(f"Combined table saved to: {output_file}")
print(f"Table shape: {combined_table.shape}")
print(f"Number of regions: {len(combined_table)}")
print("\nFirst few rows:")
print(combined_table.head())


# %%

# %%
FMRI.head(2)

# %%
# Calculate sum of absolute Z-scores for each structure across all 4 clusters
# First, compute Z-scores for each cluster
from scipy.stats import zscore

cluster_cols = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']
#FMRI_zscores = FMRI[cluster_cols].apply(zscore, axis=0)

# Calculate sum of absolute Z-scores
FMRI['Sum_Abs_Zscore'] = FMRI[cluster_cols].abs().sum(axis=1)

# Get common acronyms between GENCIC and FMRI data
common_acronyms = GENCIC.index.intersection(FMRI.index)

if len(common_acronyms) > 0:
    # Extract values for common acronyms
    gencic_bias = GENCIC.loc[common_acronyms, 'Bias']
    fmri_sum_abs_zscore = FMRI.loc[common_acronyms, 'Sum_Abs_Zscore']
    regions = GENCIC.loc[common_acronyms, 'REGION']

    # Calculate Pearson correlation and linear regression
    from scipy.stats import pearsonr, linregress
    linres = linregress(gencic_bias, fmri_sum_abs_zscore)
    pear_r, pear_p = pearsonr(gencic_bias, fmri_sum_abs_zscore)

    import numpy as np
    import matplotlib.pyplot as plt

    # Define region color mapping (from DDD.ipynb)
    REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                    'Hippocampus','Amygdala','Striatum', 
                    "Thalamus", "Hypothalamus", "Midbrain", 
                    "Medulla", "Pallidum", "Pons", 
                    "Cerebellum"]
    REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                        "#2c9d39", "#742eb5", "#ed8921", 
                                        "#e82315", "#E6B0AA", "#f6b26b",  
                                        "#20124d", "#2ECC71", "#D2B4DE", 
                                        "#ffd966"]))
    
    # Map regions to colors
    # Handle naming differences between GENCIC and color map
    region_name_mapping = {
        'Amygdalar': 'Amygdala',
        'Olfactory': 'Olfactory_areas',
        'Hippocampal': 'Hippocampus'
    }
    
    colors = []
    for region in regions:
        # Apply name mapping if needed
        mapped_region = region_name_mapping.get(region, region)
        # Get color from dictionary, default to gray if not found
        colors.append(REG_COR_Dic.get(mapped_region, '#808080'))

    plt.figure(figsize=(11,8), dpi=360)
    
    # Plot linear fit first (behind points)
    xfit = np.linspace(gencic_bias.min(), gencic_bias.max(), 100)
    yfit = linres.slope * xfit + linres.intercept
    plt.plot(
        xfit, yfit, color="#e6550d", linestyle="--", linewidth=2.2,
        label=f"Linear fit (slope={linres.slope:.2f})", zorder=1
    )
    
    # Create scatter plot with region colors
    for region_name in set(regions):
        mapped_region_name = region_name_mapping.get(region_name, region_name)
        color = REG_COR_Dic.get(mapped_region_name, '#808080')
        mask = regions == region_name
        plt.scatter(
            gencic_bias[mask], fmri_sum_abs_zscore[mask],
            color=color, edgecolor='k', s=80, alpha=0.9, zorder=2,
            label=region_name
        )
    
    # Add text annotations for each point
    for acronym in common_acronyms:
        x = gencic_bias[acronym]
        y = fmri_sum_abs_zscore[acronym]
        plt.annotate(
            acronym, (x, y),
            fontsize=6, alpha=0.8,
            xytext=(3, 3), textcoords='offset points'
        )

    plt.xlabel("GENCIC Bias", fontsize=17, fontweight='bold')
    plt.ylabel("fMRI Sum of Absolute Z-scores", fontsize=17, fontweight='bold')

    # Only show Pearson correlation in annotation, without a box
    stat_text = (
        f"r = {pear_r:.2f}, p = {pear_p:.0e}"
    )
    plt.text(
        0.12, 0.2, stat_text,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left'
    )

    plt.title(
        "GENCIC Bias vs fMRI Sum of Absolute Z-scores",
        fontsize=16,
        fontweight='bold'
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=9, loc='best', frameon=True, ncol=2)
    plt.grid(alpha=0.3, linestyle=":", zorder=0)
    plt.tight_layout()
    plt.show()

    print(f"Number of common regions: {len(common_acronyms)}")
    print(f"Pearson correlation: {pear_r:.4f}")
    print(f"Pearson P-value: {pear_p:.8f}")
else:
    print("No common acronyms found between GENCIC and FMRI data")


# %%

# %%
GENCIC.columns

# %%
# Get common acronyms between GENCIC and FMRI data
common_acronyms = GENCIC.index.intersection(FMRI.index)

if len(common_acronyms) > 0:
    # Extract fMRI data for common acronyms
    fmri_average_clusters = FMRI.loc[common_acronyms, 'Average_Clusters']
    
    # Find all columns with "Bias" in the name
    bias_columns = [col for col in GENCIC.columns if 'Bias' in col]
    
    print(f"Number of common regions: {len(common_acronyms)}")
    print(f"Bias columns found: {bias_columns}")
    print()
    
    # Calculate correlations for each bias column
    for bias_col in bias_columns:
        # Extract values for common acronyms
        gencic_bias_values = GENCIC.loc[common_acronyms, bias_col]
        
        # Calculate Spearman correlation
        spearman_correlation, spearman_p_value = spearmanr(gencic_bias_values, fmri_average_clusters)
        
        # Calculate Pearson correlation
        pearson_correlation, pearson_p_value = pearsonr(gencic_bias_values, fmri_average_clusters)
        
        print(f"{bias_col}:")
        print(f"  Spearman correlation: {spearman_correlation:.4f}, p-value: {spearman_p_value:.2e}")
        print(f"  Pearson correlation: {pearson_correlation:.4f}, p-value: {pearson_p_value:.2e}")
        print()
else:
    print("No common acronyms found between GENCIC and FMRI data")


# %%

# %%

# %%
Cut = 0
FMRI['Count_Below_Threshold'] = (FMRI[['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']] < Cut).sum(axis=1)

# %%
# Create new 'Area' column based on SubArea if valid, otherwise MacroArea
FMRI['Area'] = FMRI['SubArea'].where(FMRI['SubArea'].notna() & (FMRI['SubArea'] != ''), FMRI['MacroArea'])

# %%
FMRI[FMRI["Count_Below_Threshold"]>=3].head(60)

# %%
# Aggregate Count_Below_Threshold at MacroArea level
macro_area_aggregation = FMRI.groupby('Area')['Count_Below_Threshold'].agg(['mean', 'std', 'count']).reset_index()

# Calculate total counts for each cluster below threshold by MacroArea
cluster_counts = FMRI.groupby('Area')[['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']].apply(lambda x: (x < Cut).sum()).reset_index()

# Create a stacked bar plot showing total counts for each cluster
plt.figure(figsize=(12, 6))
width = 0.6
x = range(len(cluster_counts))

# Define colors for each cluster
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# Create stacked bars
bottom = [0] * len(cluster_counts)
for i, cluster in enumerate(['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']):
    plt.bar(x, cluster_counts[cluster], width, bottom=bottom, 
            label=cluster, color=colors[i], alpha=0.8)
    bottom = [b + c for b, c in zip(bottom, cluster_counts[cluster])]

plt.xlabel('MaArearoArea')
plt.ylabel('Total Count Below Threshold')
plt.title('Total Count Below Threshold by Area (Colored by 4 Clusters)')
plt.xticks(x, cluster_counts['Area'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Display the aggregation table
print("Count_Below_Threshold aggregated by Area:")
print(macro_area_aggregation)
print("\nTotal counts for each cluster below threshold by Area:")
print(cluster_counts)

# %%
Number_gt = 3
test_set = FMRI[FMRI["Count_Below_Threshold"]>=Number_gt]
test_set_STRs = test_set.index.tolist()
GENCIC_STRs = GENCIC[GENCIC["Circuits.32"]==1].index.tolist()

# Calculate overlap between test_set_STRs and GENCIC_STRs
overlap_STRs = set(test_set_STRs).intersection(set(GENCIC_STRs))
overlap_count = len(overlap_STRs)

# Get the sizes of each dataset
fMRI_size = len(FMRI.index)  # 163 regions
GENCIC_size = len(GENCIC.index)  # 213 regions

test_set_count = len(test_set_STRs)
GENCIC_count = len(GENCIC_STRs)

# For hypergeometric test, we need to determine the correct population
# Since we're testing overlap between subsets from different populations,
# we use the intersection of both datasets as our universe
common_regions = set(FMRI.index).intersection(set(GENCIC.index))
universe_size = len(common_regions)

# Adjust counts to only include regions in the common universe
test_set_in_universe = set(test_set_STRs).intersection(common_regions)
GENCIC_in_universe = set(GENCIC_STRs).intersection(common_regions)
test_set_count_adj = len(test_set_in_universe)
GENCIC_count_adj = len(GENCIC_in_universe)

from scipy.stats import hypergeom
# P-value is probability of getting overlap_count or more overlaps by chance
# hypergeom.sf(k-1, N, K, n) gives P(X >= k)
# N = universe_size, K = GENCIC_count_adj, n = test_set_count_adj, k = overlap_count
p_value = hypergeom.sf(overlap_count - 1, universe_size, GENCIC_count_adj, test_set_count_adj)

print(f"fMRI dataset size: {fMRI_size} regions")
print(f"GENCIC dataset size: {GENCIC_size} regions")
print(f"Common regions (universe): {universe_size} regions")
print(f"Test set (Count_Below_Threshold >= {Number_gt}): {test_set_count} regions ({test_set_count_adj} in universe)")
print(f"GENCIC Circuits.46 = 1: {GENCIC_count} regions ({GENCIC_count_adj} in universe)")
print(f"Overlap: {overlap_count} regions")
print(f"Overlap regions: {list(overlap_STRs)}")
print(f"Hypergeometric test p-value: {p_value:.6f}")

# Calculate expected overlap under null hypothesis
expected_overlap = (test_set_count_adj * GENCIC_count_adj) / universe_size
print(f"Expected overlap under null hypothesis: {expected_overlap:.2f}")

# Calculate overlap statistics
overlap_percentage_test = (overlap_count / test_set_count_adj) * 100 if test_set_count_adj > 0 else 0
overlap_percentage_GENCIC = (overlap_count / GENCIC_count_adj) * 100 if GENCIC_count_adj > 0 else 0

print(f"Overlap as % of fMRI set: {overlap_percentage_test:.2f}%")
print(f"Overlap as % of GENCIC set: {overlap_percentage_GENCIC:.2f}%")

# %%
# Run the permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 214)
set1_size = 67
set2_size = 32
observed_overlap = 14

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)

# Print results
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"Std intersection length: {results['std_intersection']:.2f}")
print(f"Min intersection length: {results['min_intersection']}")
print(f"Max intersection length: {results['max_intersection']}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")
print(f"Number of permutations with overlap >= {results['observed_overlap']}: {results['n_significant']}")

# Plot the results
plot_permutation_results(results)

# %%

# %%

# %%

# %%

# %%
Number_gt = 3
test_set = FMRI[FMRI["Count_Below_Threshold"]>=Number_gt]
test_set_STRs = test_set.index.tolist()
GENCIC_STRs = GENCIC[GENCIC["Circuits.46"]==1].index.tolist()

# Calculate overlap between test_set_STRs and GENCIC_STRs
overlap_STRs = set(test_set_STRs).intersection(set(GENCIC_STRs))
overlap_count = len(overlap_STRs)

# Get the sizes of each dataset
fMRI_size = len(FMRI.index)  # 163 regions
GENCIC_size = len(GENCIC.index)  # 213 regions

test_set_count = len(test_set_STRs)
GENCIC_count = len(GENCIC_STRs)


common_regions = set(FMRI.index).intersection(set(GENCIC.index))
universe_size = len(common_regions)

# Adjust counts to only include regions in the common universe
test_set_in_universe = set(test_set_STRs).intersection(common_regions)
GENCIC_in_universe = set(GENCIC_STRs).intersection(common_regions)
test_set_count_adj = len(test_set_in_universe)
GENCIC_count_adj = len(GENCIC_in_universe)

#from scipy.stats import hypergeom
#p_value = hypergeom.sf(overlap_count - 1, universe_size, GENCIC_count_adj, test_set_count_adj)

print(f"fMRI dataset size: {fMRI_size} regions")
print(f"GENCIC dataset size: {GENCIC_size} regions")
print(f"Test set (Count_Below_Threshold >= {Number_gt}): {test_set_count} regions ({test_set_count_adj} in universe)")
print(f"GENCIC Circuits.46 = 1: {GENCIC_count} regions ({GENCIC_count_adj} in universe)")
print(f"Overlap: {overlap_count} regions")
print(f"Overlap regions: {list(overlap_STRs)}")

# Run the permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 214)
set1_size = 67
set2_size = 46
observed_overlap = 21
results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")

plot_permutation_results(results)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Run the permutation test
SET1 = np.arange(1, 164)
#SET1 = np.arange(1, 214)
SET2 = np.arange(1, 214)
set1_size = 67
set2_size = 46
observed_overlap = 21

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)

# Print results
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"Std intersection length: {results['std_intersection']:.2f}")
print(f"Min intersection length: {results['min_intersection']}")
print(f"Max intersection length: {results['max_intersection']}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")
print(f"Number of permutations with overlap >= {results['observed_overlap']}: {results['n_significant']}")

# Plot the results
plot_permutation_results(results)

# Create simple Venn diagram visualization using matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

# Draw two overlapping circles
circle1 = patches.Circle((0.35, 0.5), 0.3, alpha=0.5, color='blue', label='Set 1')
circle2 = patches.Circle((0.65, 0.5), 0.3, alpha=0.5, color='red', label='Set 2')

ax.add_patch(circle1)
ax.add_patch(circle2)

# Add text labels for each region
ax.text(0.2, 0.5, f'{set1_size - observed_overlap}', fontsize=14, ha='center', va='center')
ax.text(0.8, 0.5, f'{set2_size - observed_overlap}', fontsize=14, ha='center', va='center')
ax.text(0.5, 0.5, f'{observed_overlap}', fontsize=14, ha='center', va='center', weight='bold')

# Add set labels
ax.text(0.2, 0.2, 'Set 1', fontsize=12, ha='center', va='center', weight='bold')
ax.text(0.8, 0.2, 'Set 2', fontsize=12, ha='center', va='center', weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Venn Diagram of Set Overlap\nObserved Overlap: {observed_overlap}', fontsize=14)
plt.show()

# %%
# %% cell 25 code

# Set threshold for analysis
Number_gt = 3

# Define test sets
Region2Exclude = "Thalamus"
FMRI_filt = FMRI[FMRI["MacroArea"] != Region2Exclude]
GENCIC_filt = GENCIC[GENCIC["REGION"] != Region2Exclude]
test_set = FMRI_filt[FMRI_filt["Count_Below_Threshold"] >= Number_gt]
test_set_STRs = test_set.index.tolist()
GENCIC_STRs = GENCIC_filt[GENCIC_filt["Circuits.46"] == 1].index.tolist()

# Calculate overlap between test_set_STRs and GENCIC_STRs
overlap_STRs = set(test_set_STRs).intersection(set(GENCIC_STRs))
overlap_count = len(overlap_STRs)

# Get dataset sizes
fMRI_size = len(FMRI.index)  # 163 regions
GENCIC_size = len(GENCIC.index)  # 213 regions
test_set_count = len(test_set_STRs)
GENCIC_count = len(GENCIC_STRs)

# Define common universe and adjust counts
common_regions = set(FMRI.index).intersection(set(GENCIC.index))
universe_size = len(common_regions)

test_set_in_universe = set(test_set_STRs).intersection(common_regions)
GENCIC_in_universe = set(GENCIC_STRs).intersection(common_regions)
test_set_count_adj = len(test_set_in_universe)
GENCIC_count_adj = len(GENCIC_in_universe)

# Calculate hypergeometric p-value
from scipy.stats import hypergeom
p_value = hypergeom.sf(overlap_count - 1, universe_size, GENCIC_count_adj, test_set_count_adj)

# Print results
print(f"fMRI dataset size: {fMRI_size} regions")
print(f"GENCIC dataset size: {GENCIC_size} regions")
print(f"Test set (Count_Below_Threshold >= {Number_gt}): {test_set_count} regions ({test_set_count_adj} in universe)")
print(f"GENCIC Circuits.46 = 1: {GENCIC_count} regions ({GENCIC_count_adj} in universe)")
print(f"Overlap: {overlap_count} regions")
print(f"Overlap regions: {list(overlap_STRs)}")

# Run permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 214)
set1_size = 53
set2_size = 38
observed_overlap = 19

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")

plot_permutation_results(results)

# %%
19 / 35

# %%


# Run the permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 214)
set1_size = 67
set2_size = 46
observed_overlap = 21

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)

# Print results
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"Std intersection length: {results['std_intersection']:.2f}")
print(f"Min intersection length: {results['min_intersection']}")
print(f"Max intersection length: {results['max_intersection']}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")
print(f"Number of permutations with overlap >= {results['observed_overlap']}: {results['n_significant']}")

# Plot the results
plot_permutation_results(results)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Run the permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 214)
set1_size = 67
set2_size = 46
observed_overlap = 21

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)

# Print results
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"Std intersection length: {results['std_intersection']:.2f}")
print(f"Min intersection length: {results['min_intersection']}")
print(f"Max intersection length: {results['max_intersection']}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")
print(f"Number of permutations with overlap >= {results['observed_overlap']}: {results['n_significant']}")

# Plot the results
plot_permutation_results(results)

# Create simple Venn diagram visualization using matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

# Draw two overlapping circles
circle1 = patches.Circle((0.35, 0.5), 0.3, alpha=0.5, color='blue', label='Set 1')
circle2 = patches.Circle((0.65, 0.5), 0.3, alpha=0.5, color='red', label='Set 2')

ax.add_patch(circle1)
ax.add_patch(circle2)

# Add text labels for each region
ax.text(0.2, 0.5, f'{set1_size - observed_overlap}', fontsize=14, ha='center', va='center')
ax.text(0.8, 0.5, f'{set2_size - observed_overlap}', fontsize=14, ha='center', va='center')
ax.text(0.5, 0.5, f'{observed_overlap}', fontsize=14, ha='center', va='center', weight='bold')

# Add set labels
ax.text(0.2, 0.2, 'Set 1', fontsize=12, ha='center', va='center', weight='bold')
ax.text(0.8, 0.2, 'Set 2', fontsize=12, ha='center', va='center', weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Venn Diagram of Set Overlap\nObserved Overlap: {observed_overlap}', fontsize=14)
plt.show()

# %%
# Run the permutation test
SET1 = np.arange(1, 164)
SET2 = np.arange(1, 164)
set1_size = 67
set2_size = 43
observed_overlap = 21

results = permutation_test_overlap(SET1, SET2, set1_size, set2_size, observed_overlap, n_permutations=10000)

# Print results
print(f"Mean intersection length: {results['mean_intersection']:.2f}")
print(f"Std intersection length: {results['std_intersection']:.2f}")
print(f"Min intersection length: {results['min_intersection']}")
print(f"Max intersection length: {results['max_intersection']}")
print(f"P-value for overlap >= {results['observed_overlap']}: {results['p_value']:.6f}")
print(f"Number of permutations with overlap >= {results['observed_overlap']}: {results['n_significant']}")

# Plot the results
plot_permutation_results(results)

# %%

# %%
# OK Let test if each individual cluster is overlap with Test set. 
for cluster in ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']:
    #_sub_test_set = FMRI.sort_values(by=cluster, ascending=False).head(46).index.values
    _sub_test_set = FMRI.sort_values(by=cluster, ascending=False).tail(46).index.values
    # Test overlap between _sub_test_set and test_set_STRs
    _sub_overlap = set(_sub_test_set).intersection(set(test_set_STRs))
    _sub_overlap_count = len(_sub_overlap)
    
    _sub_test_count = len(_sub_test_set)
    _test_set_count = len(test_set_STRs)
    _pool_size = 163  # total regions in the pool
    
    _sub_p_value = hypergeom.sf(_sub_overlap_count - 1, _pool_size, _test_set_count, _sub_test_count)

    SET1 = np.arange(1, 164)
    SET2 = np.arange(1, 164)
    set1_size = 67
    set2_size = 46
    observed_overlap = _sub_overlap_count
    perm_p_value = permutation_test_overlap(SET1, SET2, set1_size, set2_size, _sub_overlap_count, n_permutations=10000)
    
    print(f"\n{cluster} Analysis:")
    print(f"Cluster regions: {_sub_test_count}")
    print(f"Test set regions: {_test_set_count}")
    print(f"Overlap: {_sub_overlap_count} regions")
    #print(f"Overlap regions: {list(_sub_overlap)}")
    #print(f"Hypergeometric p-value: {_sub_p_value:.6f}")
    print(f"Permutation p-value: {perm_p_value['p_value']:.6f}")

# %%
# Try another way. Under and Over connectivity above certain threshold. 
Cut = 0.5
FMRI['Count_Below_Threshold_v2'] = (FMRI[['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']].abs() > Cut).sum(axis=1)

# %%
Number_gt = 2
test_set = FMRI[FMRI["Count_Below_Threshold_v2"]>=Number_gt]
test_set_STRs = test_set.index.tolist()
GENCIC_STRs = GENCIC[GENCIC["Circuits.46"]==1].index.tolist()

# Calculate overlap between test_set_STRs and GENCIC_STRs
overlap_STRs = set(test_set_STRs).intersection(set(GENCIC_STRs))
overlap_count = len(overlap_STRs)


# First, let's get the union of all regions from both datasets to define our universe
all_fMRI_regions = set(FMRI.index)
all_GENCIC_regions = set(GENCIC.index)
universe_regions = all_fMRI_regions.union(all_GENCIC_regions)
universe_size = len(universe_regions)

test_set_count = len(test_set_STRs)
GENCIC_count = len(GENCIC_STRs)

from scipy.stats import hypergeom
p_value = hypergeom.sf(overlap_count - 1, universe_size, GENCIC_count, test_set_count)

print(f"Universe size (union of both datasets): {universe_size} regions")
print(f"Test set (Count_Below_Threshold >= {Number_gt}): {test_set_count} regions")
print(f"GENCIC Circuits.46 = 1: {GENCIC_count} regions")
print(f"Overlap: {overlap_count} regions")
print(f"Overlap regions: {list(overlap_STRs)}")


# Calculate expected overlap under null hypothesis
expected_overlap = (test_set_count * GENCIC_count) / universe_size
print(f"Expected overlap under null hypothesis: {expected_overlap:.2f}")

# Calculate overlap statistics
overlap_percentage_test = (overlap_count / test_set_count) * 100 if test_set_count > 0 else 0
overlap_percentage_GENCIC = (overlap_count / GENCIC_count) * 100 if GENCIC_count > 0 else 0

print(f"Overlap as % of fMRI set: {overlap_percentage_test:.2f}%")
print(f"Overlap as % of GENCIC set: {overlap_percentage_GENCIC:.2f}%")
print(f"Hypergeometric test p-value: {p_value:.6f}")

# %%
GENCIC.head(2)

# %%
# Annotate FMRI dataframe with GENCIC Bias and Circuits.46 membership
FMRI_annotated = FMRI.copy()

# Add GENCIC Bias column
FMRI_annotated['GENCIC_Bias'] = FMRI_annotated.index.map(GENCIC['Bias'])
FMRI_annotated['GENCIC_Circuits_46'] = FMRI_annotated.index.map(GENCIC['Circuits.46'])
FMRI_annotated['FullName'] = FMRI_annotated.index.map(GENCIC['Structure'])

# Fill NaN values with 0 for regions not in GENCIC
FMRI_annotated['GENCIC_Bias'] = FMRI_annotated['GENCIC_Bias'].fillna(0)
FMRI_annotated['GENCIC_Circuits_46'] = FMRI_annotated['GENCIC_Circuits_46'].fillna(0)

FMRI_annotated

# %%
FMRI_annotated.to_csv(os.path.join(ProjDIR, 'results/FMRI_annotated.csv'))

# %%
FMRI_annotated.head(2)


# %% [markdown]
# # Conncetion overlap 

# %%
def load_connectivity_data(score_mat_dir):
    """Load connectivity matrices"""
    ipsi_info_mat = pd.read_csv(score_mat_dir + "InfoMat.Ipsi.csv", index_col=0)
    weight_mat = pd.read_csv(score_mat_dir + "WeightMat.Ipsi.csv", index_col=0)
    ipsi_info_mat_short = pd.read_csv(score_mat_dir + "InfoMat.Ipsi.Short.3900.csv", index_col=0)
    ipsi_info_mat_long = pd.read_csv(score_mat_dir + "InfoMat.Ipsi.Long.3900.csv", index_col=0)
    
    return ipsi_info_mat, weight_mat, ipsi_info_mat_short, ipsi_info_mat_long


# %%
score_mat_dir = os.path.join(ProjDIR, "dat/allen-mouse-conn/ConnectomeScoringMat/")
ipsi_info_mat, weight_mat, ipsi_info_mat_short, ipsi_info_mat_long = load_connectivity_data(score_mat_dir)

# %%
weight_mat

# %%
GENCIC.head(2)

# %%
cir_struc = GENCIC.loc[GENCIC["Circuits.46"]==1, "Structure"]
cir_struc_wm = weight_mat.loc[cir_struc, cir_struc]

# %%
cir_fmri = FMRI_annotated.loc[FMRI_annotated["Count_Below_Threshold"]>=3, "FullName"]
cir_fmri_wm = weight_mat.loc[cir_fmri, cir_fmri]

# %%
all_struc = set(cir_struc_wm.index) | set(cir_fmri_wm.index)
print(len(all_struc))

# %%
# initialize counters and flags
common = 0
gencic_only = 0
fmri_only = 0

for str_i in all_struc:
    for str_j in all_struc:
        if str_i == str_j:
            continue

        gencic_flag = False
        fmri_flag = False

        if str_i in cir_struc_wm.index and str_j in cir_struc_wm.index:
            if cir_struc_wm.loc[str_i, str_j] != 0:
                gencic_flag = True
        if str_i in cir_fmri_wm.index and str_j in cir_fmri_wm.index:
            if cir_fmri_wm.loc[str_i, str_j] != 0:
                fmri_flag = True

        if gencic_flag and fmri_flag:
            common += 1
        elif gencic_flag:
            gencic_only += 1
        elif fmri_flag:
            fmri_only += 1
                
print(f"Common: {common}")
print(f"Gencic only: {gencic_only}")
print(f"Fmri only: {fmri_only}")


# %%

# %% [markdown]
# # Mouse Model fMRI

# %%
DataDIR = os.path.join(ProjDIR, "dat/MouseFMRI/")
data_csf = sio.loadmat(DataDIR + "global_connectivity_allsubjs_CSF.mat")
data_gsr = sio.loadmat(DataDIR + "global_connectivity_allsubjs_GSR.mat")

parcel_indices = pd.read_csv(DataDIR + "parcel_indices_424.csv")
parcel_labels = pd.read_csv(DataDIR + "parc_labels_424_LR.csv")

# %%
data_csf

# %%
gc_csf = data_csf['global_connectivity_allsubjs']

# %%
gc_csf.shape

# %%
from pathlib import Path

class MouseGlobalConnectivity:
    def __init__(self, mat_file, parcel_idx_file, parcel_label_file):
        self.mat_file = Path(mat_file)
        self.mouse_models = ['shank3b', 'chd8', 'cntnap2', 'mecp2']
        self.genotypes = ['mutant', 'wt']

        # Load parcel metadata
        self.parcel_indices = pd.read_csv(parcel_idx_file, header=None, names=['index'])  # 1-based indices
        self.parcel_labels = pd.read_csv(parcel_label_file)  # has 'name' column

        # Map indices to names (adjust 1-based to 0-based indexing)
        idx_zero_based = self.parcel_indices['index'].values - 1
        self.parcel_names = self.parcel_labels.iloc[idx_zero_based]['name'].tolist()

        # Load MATLAB data
        self.data = self._load_mat()

    def _load_mat(self):
        """Load MATLAB .mat file (v7.2 or older)"""
        mat = sio.loadmat(self.mat_file, squeeze_me=True)
        return {k: v for k, v in mat.items() if not k.startswith('__')}

    def get_connectivity(self, mouse_model, genotype):
        """Return connectivity matrix for a given mouse model and genotype"""
        if mouse_model not in self.mouse_models:
            raise ValueError(f"Unknown mouse model: {mouse_model}")
        if genotype not in self.genotypes:
            raise ValueError(f"Genotype must be one of {self.genotypes}")

        arr = self.data['global_connectivity_allsubjs']  # 4×2 array of matrices
        row = self.mouse_models.index(mouse_model)
        col = self.genotypes.index(genotype)

        mat = arr[row, col]
        return np.array(mat)

    def _merge_hemispheres(self, df, strategy="average"):
        """Merge left/right hemisphere parcels. If only one side exists, keep as is."""
        base_names = df.index.str.replace(r'_(L|R)$', '', regex=True)

        if strategy == "average":
            # For each base name, average L/R if both exist, else just keep the one present
            df = df.copy()
            df['__base__'] = base_names
            merged = []
            for base, group in df.groupby('__base__'):
                if len(group) == 2:
                    merged_row = group.drop(columns='__base__').mean()
                else:
                    merged_row = group.drop(columns='__base__').iloc[0]
                merged.append((base, merged_row))
            merged_df = pd.DataFrame([row for _, row in merged], index=[base for base, _ in merged])
            return merged_df

        elif strategy == "concat":
            # For each base name, concat L and R columns if both exist, else just keep the one present
            left_mask = df.index.str.endswith('_L')
            right_mask = df.index.str.endswith('_R')
            left_df = df[left_mask].copy()
            right_df = df[right_mask].copy()

            left_df.index = left_df.index.str.replace(r'_L$', '', regex=True)
            right_df.index = right_df.index.str.replace(r'_R$', '', regex=True)

            left_df.columns = [f"{c}_L" for c in left_df.columns]
            right_df.columns = [f"{c}_R" for c in right_df.columns]

            # Find all base names
            all_bases = set(left_df.index) | set(right_df.index)
            concat_rows = []
            concat_index = []
            for base in sorted(all_bases):
                left_row = left_df.loc[base] if base in left_df.index else None
                right_row = right_df.loc[base] if base in right_df.index else None
                if left_row is not None and right_row is not None:
                    row = pd.concat([left_row, right_row])
                elif left_row is not None:
                    row = left_row
                elif right_row is not None:
                    row = right_row
                else:
                    continue  # Should not happen
                concat_rows.append(row)
                concat_index.append(base)
            expanded = pd.DataFrame(concat_rows, index=concat_index)
            expanded = expanded.sort_index()
            return expanded

        elif strategy is None:
            return df

        else:
            raise ValueError("merge strategy must be 'average', 'concat', or None")

    def get_dataframe(self, mouse_model, genotype, merge=None):
        """
        Return DataFrame with parcel names and connectivity values.
        merge: None, 'average', or 'concat'
        """
        mat = self.get_connectivity(mouse_model, genotype)
        df = pd.DataFrame(
            mat,
            index=self.parcel_names,
            columns=[f"subj_{i+1}" for i in range(mat.shape[1])]
        )
        df.index.name = 'parcel_name'

        if merge is not None:
            df = self._merge_hemispheres(df, strategy=merge)

        return df


# %%
from scipy.stats import mannwhitneyu
import numpy as np

def connectivity_test(data, method, gene):
    mut_df = data[method][gene]["mutant"]
    wt_df = data[method][gene]["wt"]
    results = []
    for i, _str in enumerate(mut_df.index.values):
        mut_conn = mut_df.iloc[i, :]
        wt_conn = wt_df.iloc[i, :]
        # Exclude invalid values (NaN, inf, -inf)
        mut_conn_valid = mut_conn[~np.isnan(mut_conn) & np.isfinite(mut_conn)]
        wt_conn_valid = wt_conn[~np.isnan(wt_conn) & np.isfinite(wt_conn)]
        mut_conn_mean = mut_conn_valid.mean()
        wt_conn_mean = wt_conn_valid.mean()
        mut_conn_std = mut_conn_valid.std()
        wt_conn_std = wt_conn_valid.std()
        # Cohen's D
        if len(mut_conn_valid) > 1 and len(wt_conn_valid) > 1:
            # pooled std
            n1 = len(mut_conn_valid)
            n2 = len(wt_conn_valid)
            s1 = mut_conn_std
            s2 = wt_conn_std
            pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
            cohens_d = (mut_conn_mean - wt_conn_mean) / pooled_std if pooled_std > 0 else np.nan
        else:
            cohens_d = np.nan
        # Only perform test if both groups have at least one valid value
        if len(mut_conn_valid) > 0 and len(wt_conn_valid) > 0:
            stat, p = mannwhitneyu(mut_conn_valid, wt_conn_valid, alternative='two-sided')
        else:
            p = np.nan
        results.append({
            'parcel_name': _str,
            'mut_mean': mut_conn_mean,
            'wt_mean': wt_conn_mean,
            'conn_diff': mut_conn_mean - wt_conn_mean,
            'mut_std': mut_conn_std,
            'wt_std': wt_conn_std,
            'mwu_p': p,
            "cohens_d": cohens_d
        })
    results_df = pd.DataFrame(results).set_index('parcel_name')
    results_df = results_df.sort_values(by='mwu_p')
    return results_df

def collapse_hemispheres(df, strategy='mean'):
    if strategy == 'mean':
        # Strip _L / _R suffix from parcel names
        collapsed = df.copy()
        collapsed.index = collapsed.index.str.replace(r'_(L|R)$', '', regex=True)
        # Group by the new parcel name and take the mean across L and R
        collapsed = collapsed.groupby(collapsed.index).mean()
        return collapsed
    elif strategy == 'concat':
        # Strip _L / _R suffix from parcel names
        collapsed = df.copy()
        collapsed.index = collapsed.index.str.replace(r'_(L|R)$', '', regex=True)
        # Group by the new parcel name and take the mean across L and R
        collapsed = collapsed.groupby(collapsed.index).mean()
        return collapsed
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
        
def print_data_treeview(data, indent=0):
    for preproc in data:
        print("  " * indent + f"{preproc}/")
        for gene in data[preproc]:
            print("  " * (indent + 1) + f"{gene}/")
            for group in data[preproc][gene]:
                print("  " * (indent + 2) + f"{group}: DataFrame shape {data[preproc][gene][group].shape}")


# %%
# Cleaned up data structure: use nested dicts for easy access
# Structure: data[preproc][gene][group] = dataframe

data = {}
for preproc, loader in {
    "CSF": MouseGlobalConnectivity(
        mat_file=DataDIR + "global_connectivity_allsubjs_CSF.mat",
        parcel_idx_file=DataDIR + "parcel_indices_424.csv",
        parcel_label_file=DataDIR + "parc_labels_424_LR.csv"
    ),
    "GSR": MouseGlobalConnectivity(
        mat_file=DataDIR + "global_connectivity_allsubjs_GSR.mat",
        parcel_idx_file=DataDIR + "parcel_indices_424.csv",
        parcel_label_file=DataDIR + "parc_labels_424_LR.csv"
    )
}.items():
    data[preproc] = {}
    for gene in ["shank3b", "cntnap2", "chd8", "mecp2"]:
        data[preproc][gene] = {}
        for group in ["mutant", "wt"]:
            data[preproc][gene][group] = loader.get_dataframe(gene, group, merge=None)
print_data_treeview(data)

# %%
data_LR_merge = {}
for preproc, loader in {
    "CSF": MouseGlobalConnectivity(
        mat_file=DataDIR + "global_connectivity_allsubjs_CSF.mat",
        parcel_idx_file=DataDIR + "parcel_indices_424.csv",
        parcel_label_file=DataDIR + "parc_labels_424_LR.csv"
    ),
    "GSR": MouseGlobalConnectivity(
        mat_file=DataDIR + "global_connectivity_allsubjs_GSR.mat",
        parcel_idx_file=DataDIR + "parcel_indices_424.csv",
        parcel_label_file=DataDIR + "parc_labels_424_LR.csv"
    )
}.items():
    data_LR_merge[preproc] = {}
    for gene in ["shank3b", "cntnap2", "chd8", "mecp2"]:
        data_LR_merge[preproc][gene] = {}
        for group in ["mutant", "wt"]:
            data_LR_merge[preproc][gene][group] = loader.get_dataframe(gene, group, merge="average")
print_data_treeview(data_LR_merge)

# %%
#shank3b_mut.head(5)
data["CSF"]["shank3b"]["mutant"].head(5)

# %%
CSF_shank3b_res = connectivity_test(data, "CSF", "shank3b")
CSF_chd8_res = connectivity_test(data, "CSF", "chd8")
CSF_cntnap2_res = connectivity_test(data, "CSF", "cntnap2")
CSF_mecp2_res = connectivity_test(data, "CSF", "mecp2")

GSR_shank3b_res = connectivity_test(data, "GSR", "shank3b")
GSR_chd8_res = connectivity_test(data, "GSR", "chd8")
GSR_cntnap2_res = connectivity_test(data, "GSR", "cntnap2")
GSR_mecp2_res = connectivity_test(data, "GSR", "mecp2")

# %%
CSF_shank3b_res.head(10)

# %%
import matplotlib.pyplot as plt
import numpy as np

def compare_lr_correlations(results_df, plot=True, title=None):
    """
    Compare left vs right hemisphere correlation for cohens_d and mwu_p,
    and make scatter plots of L vs R for both cohens_d and -log10(mwu_p).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Must have index with `_L` and `_R` suffixes and columns 'cohens_d' and 'mwu_p'.
    plot : bool
        If True, show scatter plots.
    
    Returns
    -------
    pd.DataFrame
        Spearman correlations for cohens_d and mwu_p.
    """
    # Make sure we only use parcels with both L and R
    base_names = results_df.index.str.replace(r'_(L|R)$', '', regex=True)
    results_df = results_df.assign(base=base_names)
    
    left_df = results_df[results_df.index.str.endswith('_L')].copy()
    right_df = results_df[results_df.index.str.endswith('_R')].copy()
    
    # Align on base name
    left_df.index = left_df['base']
    right_df.index = right_df['base']
    
    # Intersect bases to be safe
    common = left_df.index.intersection(right_df.index)
    left_df = left_df.loc[common]
    right_df = right_df.loc[common]
    
    # Spearman correlations
    rho_cd, p_cd = spearmanr(left_df['cohens_d'], right_df['cohens_d'])
    rho_pval, p_pval = spearmanr(left_df['mwu_p'], right_df['mwu_p'])
    
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Scatter for cohens_d
        axs[0].scatter(left_df['cohens_d'], right_df['cohens_d'], alpha=0.7)
        axs[0].set_xlabel('Left cohens_d')
        axs[0].set_ylabel('Right cohens_d')
        axs[0].set_title(f'cohens_d L vs R\nSpearman r={rho_cd:.2f}, p={p_cd:.2g}')
        axs[0].axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)
        
        # Scatter for -log10(mwu_p)
        left_logp = -np.log10(left_df['mwu_p'].clip(lower=1e-20))
        right_logp = -np.log10(right_df['mwu_p'].clip(lower=1e-20))
        axs[1].scatter(left_logp, right_logp, alpha=0.7)
        axs[1].set_xlabel('-log10(mwu_p) Left')
        axs[1].set_ylabel('-log10(mwu_p) Right')
        axs[1].set_title(f'-log10(mwu_p) L vs R\nSpearman r={rho_pval:.2f}, p={p_pval:.2g}')
        axs[1].axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        if title is not None:
            fig.suptitle(title)
            plt.subplots_adjust(top=0.85)
        plt.show()
    
    res = {
        "n_pairs": len(common),
        "cohens_d_spearman_rho": rho_cd,
        "cohens_d_pval": p_cd,
        "mwu_p_spearman_rho": rho_pval,
        "mwu_p_pval": p_pval
    }
    return pd.DataFrame([res])


# %%
for res_df, name in zip([CSF_shank3b_res, GSR_shank3b_res, CSF_chd8_res, GSR_chd8_res, CSF_cntnap2_res, GSR_cntnap2_res, CSF_mecp2_res, GSR_mecp2_res], ["CSF Shank3b", "GSR Shank3b", "CSF Chd8", "GSR Chd8", "CSF Cntnap2", "GSR Cntnap2", "CSF Mepc2", "GSR Mepc2"]) :
    lr_corr = compare_lr_correlations(res_df, title=name)
#print(lr_corr)

# %%
# Separate each parcel's L and R entries into separate DataFrames for all 4 models, assuming index ends with "_L" or "_R"
CSF_shank3b_L = CSF_shank3b_res[CSF_shank3b_res.index.str.endswith("_L")].copy()
CSF_shank3b_L.index = CSF_shank3b_L.index.str[:-2]

CSF_shank3b_R = CSF_shank3b_res[CSF_shank3b_res.index.str.endswith("_R")].copy()
CSF_shank3b_R.index = CSF_shank3b_R.index.str[:-2]

CSF_chd8_L = CSF_chd8_res[CSF_chd8_res.index.str.endswith("_L")].copy()
CSF_chd8_L.index = CSF_chd8_L.index.str[:-2]

CSF_chd8_R = CSF_chd8_res[CSF_chd8_res.index.str.endswith("_R")].copy()
CSF_chd8_R.index = CSF_chd8_R.index.str[:-2]

CSF_cntnap2_L = CSF_cntnap2_res[CSF_cntnap2_res.index.str.endswith("_L")].copy()
CSF_cntnap2_L.index = CSF_cntnap2_L.index.str[:-2]

CSF_cntnap2_R = CSF_cntnap2_res[CSF_cntnap2_res.index.str.endswith("_R")].copy()
CSF_cntnap2_R.index = CSF_cntnap2_R.index.str[:-2]

CSF_mecp2_L = CSF_mecp2_res[CSF_mecp2_res.index.str.endswith("_L")].copy()
CSF_mecp2_L.index = CSF_mecp2_L.index.str[:-2]

CSF_mecp2_R = CSF_mecp2_res[CSF_mecp2_res.index.str.endswith("_R")].copy()
CSF_mecp2_R.index = CSF_mecp2_R.index.str[:-2]

# %%
# Store L/R separated DataFrames in a dict: Data[gene]["L"], Data[gene]["R"]
genes = ['shank3b', 'chd8', 'cntnap2', 'mecp2']
csf_results = {
    'shank3b': CSF_shank3b_res,
    'chd8': CSF_chd8_res,
    'cntnap2': CSF_cntnap2_res,
    'mecp2': CSF_mecp2_res
}

Data = {}
for gene in genes:
    Data[gene] = {}
    df = csf_results[gene]
    Data[gene]['L'] = df[df.index.str.endswith('_L')].copy()
    Data[gene]['L'].index = Data[gene]['L'].index.str[:-2]
    Data[gene]['R'] = df[df.index.str.endswith('_R')].copy()
    Data[gene]['R'].index = Data[gene]['R'].index.str[:-2]

# %%
Data["shank3b"]["L"].head(5)

# %%
# Create a DataFrame to collect cohens_d for each gene, left hemisphere (L)
cohens_d_L = pd.DataFrame({
    gene: Data[gene]["L"]["cohens_d"] for gene in ['shank3b', 'chd8', 'cntnap2', 'mecp2']
})
# Add a column indicating the number of negative values (0-4) for each row across the 4 genes
cohens_d_L["Count_Negative"] = (cohens_d_L < 0).sum(axis=1)

# Create a DataFrame to collect cohens_d for each gene, right hemisphere (R)
cohens_d_R = pd.DataFrame({
    gene: Data[gene]["R"]["cohens_d"] for gene in ['shank3b', 'chd8', 'cntnap2', 'mecp2']
})
cohens_d_R["Count_Negative"] = (cohens_d_R < 0).sum(axis=1)

# %%
# Find parcels with at least 3 negative values in L and R
cohens_d_L_gt3 = cohens_d_L[cohens_d_L["Count_Negative"] >= 3].index.values
cohens_d_R_gt3 = cohens_d_R[cohens_d_R["Count_Negative"] >= 3].index.values

# Find the overlap (intersection) between L and R
overlap_parcels = np.intersect1d(cohens_d_L_gt3, cohens_d_R_gt3)
n_overlap = len(overlap_parcels)

# Compute the p-value: 
# Null hypothesis: overlap as large as observed or greater occurs by chance,
# given the sizes of two sets (hypergeometric test)
from scipy.stats import hypergeom

M = len(cohens_d_L.index)        # total number of parcels (universe)
n = len(cohens_d_L_gt3)          # L "successes"
N = len(cohens_d_R_gt3)          # R "draws"
X = n_overlap                    # observed overlap

# p-value: probability of overlap >= X by chance
p_overlap = hypergeom.sf(X-1, M, n, N)

# Print the results
print("Number of parcels with >=3 negative values (L):", n)
print("Number of parcels with >=3 negative values (R):", N)
print("Number of parcels overlapping (L ∩ R):", n_overlap)
print("Overlapping parcels:", overlap_parcels)
print("Hypergeometric p-value for this overlap:", p_overlap)


# %%
44 / 73

# %%
cohens_d_R_gt3

# %%

# %%

# %%

# %%
CSF_merge_shank3b_res = connectivity_test(data_LR_merge, "CSF", "shank3b")
CSF_merge_chd8_res = connectivity_test(data_LR_merge, "CSF", "chd8")
CSF_merge_cntnap2_res = connectivity_test(data_LR_merge, "CSF", "cntnap2")
CSF_merge_mecp2_res = connectivity_test(data_LR_merge, "CSF", "mecp2")

GSR_merge_shank3b_res = connectivity_test(data_LR_merge, "GSR", "shank3b")
GSR_merge_chd8_res = connectivity_test(data_LR_merge, "GSR", "chd8")
GSR_merge_cntnap2_res = connectivity_test(data_LR_merge, "GSR", "cntnap2")
GSR_merge_mecp2_res = connectivity_test(data_LR_merge, "GSR", "mecp2")

merge_res_dict = {
    "CSF_merge": {
        "shank3b": CSF_merge_shank3b_res,
        "chd8": CSF_merge_chd8_res,
        "cntnap2": CSF_merge_cntnap2_res,
        "mecp2": CSF_merge_mecp2_res
    },
    "GSR_merge": {
        "shank3b": GSR_merge_shank3b_res,
        "chd8": GSR_merge_chd8_res,
        "cntnap2": GSR_merge_cntnap2_res,
        "mecp2": GSR_merge_mecp2_res
    }
}

# %%
merge_res_dict["CSF_merge"]["shank3b"].sort_values(by="conn_diff").head(50)

# %%
merge_res_dict["CSF_merge"]["chd8"].sort_values(by="conn_diff").tail(50)


# %%
def compare_models_spearman(merge_results_dict, stat_type="conn_diff", merge_key="CSF_merge"):
    """
    Compare models pairwise by Spearman correlation of MWU p-values using the new merge_res_dict structure.

    merge_results_dict: dict
        Outer keys = merge type (e.g., "CSF_merge", "GSR_merge")
        Inner keys = model name (str)
        Inner values = results DataFrame (must have 'conn_diff' column and same index)
    stat_type: str
        The column to use for correlation (default "conn_diff")
    merge_key: str
        Which merge type to use from merge_results_dict (default "CSF_merge")

    Returns:
        DataFrame of pairwise Spearman correlation coefficients.
    """
    # Use the specified merge_key to get the inner dict of models
    results_dict = merge_results_dict[merge_key]
    models = list(results_dict.keys())
    corr_df = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1, m2 in combinations(models, 2):
        # Align by parcel_name index
        s1 = results_dict[m1][stat_type]
        s2 = results_dict[m2][stat_type]
        aligned = pd.concat([s1, s2], axis=1, join='inner').dropna()

        rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        corr_df.loc[m1, m2] = rho
        corr_df.loc[m2, m1] = rho

    # Fill diagonal
    for m in models:
        corr_df.loc[m, m] = 1.0

    return corr_df


# %%
rho_table = compare_models_spearman(merge_res_dict, merge_key="CSF_merge")
sns.heatmap(rho_table.astype(float), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman correlation of MWU p-values across models")
plt.show()

# %%
rho_table = compare_models_spearman(merge_res_dict, merge_key="GSR_merge")
sns.heatmap(rho_table.astype(float), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman correlation of MWU p-values across models")
plt.show()


# %%
def compare_models_topN_overlap(results_dict, N=50, merge_key=None):
    """
    Compare models by % overlap of top-N parcels (lowest MWU p-values).

    Parameters
    ----------
    results_dict : dict
        If merge_key is None:
            Keys = model name (str)
            Values = results DataFrame (must have 'mwu_p' column)
        If merge_key is not None:
            Outer keys = merge type (e.g., "CSF_merge", "GSR_merge")
            Inner keys = model name (str)
            Inner values = results DataFrame (must have 'mwu_p' column)
    N : int
        Number of top parcels to compare.
    merge_key : str or None
        If not None, use this key to select the inner dict from results_dict.

    Returns
    -------
    pd.DataFrame
        Pairwise % overlap of top-N parcels.
    """
    # Handle merge_res_dict format if merge_key is provided
    if merge_key is not None:
        results_dict = results_dict[merge_key]

    models = list(results_dict.keys())
    overlap_df = pd.DataFrame(index=models, columns=models, dtype=float)

    # Precompute top-N sets for each model
    top_sets = {}
    for model, df in results_dict.items():
        top_sets[model] = set(df.sort_values('mwu_p').head(N).index)

    for m1, m2 in combinations(models, 2):
        overlap_count = len(top_sets[m1] & top_sets[m2])
        overlap_pct = overlap_count / N
        overlap_df.loc[m1, m2] = overlap_pct
        overlap_df.loc[m2, m1] = overlap_pct

    # Fill diagonal with 1.0
    for m in models:
        overlap_df.loc[m, m] = 1.0

    return overlap_df


# %%
compare_models_topN_overlap(merge_res_dict, merge_key="CSF_merge", N=50)

# %%
compare_models_topN_overlap(merge_res_dict, merge_key="GSR_merge", N=50)

# %%
GENCIC = pd.read_excel(os.path.join(ProjDIR, "results/SupTabs.v57.xlsx"), sheet_name="Table-S1- Structure Bias", index_col=0)

# %%
GENCIC.head(2)

# %%
CommonSTRs = merge_res_dict["CSF_merge"]["shank3b"].index.intersection(GENCIC.index)
GENCIC_intersect = GENCIC.loc[CommonSTRs]

# %%
# Annotate conn_diff and mwu_p for each mouse model and each method to GENCIC DataFrame

# Define the methods and mouse models to annotate
methods = ["CSF_merge", "GSR_merge"]
mouse_models = list(merge_res_dict["CSF_merge"].keys())

for method in methods:
    for model in mouse_models:
        # Prepare column names for conn_diff and mwu_p
        conn_col = f"{model}_{method}_conn_diff"
        pval_col = f"{model}_{method}_mwu_p"
        # Initialize columns if not present
        if conn_col not in GENCIC_intersect.columns:
            GENCIC_intersect[conn_col] = pd.NA
        if pval_col not in GENCIC_intersect.columns:
            GENCIC_intersect[pval_col] = pd.NA
        # Get the result DataFrame for this model/method
        res_df = merge_res_dict[method][model]
        for STR in GENCIC_intersect.index:
            if STR in res_df.index:
                GENCIC_intersect.at[STR, conn_col] = res_df.at[STR, "conn_diff"] if "conn_diff" in res_df.columns else pd.NA
                GENCIC_intersect.at[STR, pval_col] = res_df.at[STR, "mwu_p"] if "mwu_p" in res_df.columns else pd.NA
            else:
                GENCIC_intersect.at[STR, conn_col] = pd.NA
                GENCIC_intersect.at[STR, pval_col] = pd.NA

# %%
GENCIC_intersect.head(2)

# %%
GENCIC_intersect.columns.values

# %%
import matplotlib.pyplot as plt

mousemodels = ["shank3b", "chd8", "cntnap2", "mecp2"]
methods = ["CSF_merge", "GSR_merge"]

fig, axes = plt.subplots(len(mousemodels), len(methods), figsize=(10, 16), dpi=150, sharex=True, sharey=False)
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, mousemodel in enumerate(mousemodels):
    for j, method in enumerate(methods):
        ax = axes[i, j]
        x = GENCIC_intersect["Bias"]
        y = GENCIC_intersect[f"{mousemodel}_{method}_conn_diff"]
        valid = x.notna() & y.notna()
        if valid.sum() > 1:
            corr, p = spearmanr(x[valid], y[valid])
            ax.scatter(x[valid], y[valid], alpha=0.7, s=20)
            ax.set_title(f"{mousemodel} - {method}")
            ax.annotate(f"r={corr:.2f}\np={p:.2g}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", fontsize=10)
            ax.set_title(f"{mousemodel} - {method}")
        if i == len(mousemodels) - 1:
            ax.set_xlabel("GENCIC Bias")
        if j == 0:
            ax.set_ylabel("Conn Diff")
plt.suptitle("GENCIC Bias vs Mouse Model Conn Diff\n(Spearman r and p shown)", fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Collect all relevant columns for pairwise correlation
cols = ["Bias"]
for mousemodel in ["shank3b", "chd8", "cntnap2", "mecp2"]:
    for method in ["CSF_merge", "GSR_merge"]:
        col = f"{mousemodel}_{method}_conn_diff"
        if col in GENCIC_intersect.columns:
            cols.append(col)

# Subset and drop rows with all-NA
df_corr = GENCIC_intersect[cols].copy()
df_corr = df_corr.dropna(how="all", subset=cols)

# Compute pairwise Spearman correlation
corr_matrix = df_corr.corr(method="spearman")

# Cluster the correlation matrix and show clustered heatmap
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns

# Compute linkage for rows and columns
linkage_rows = linkage(corr_matrix, method='average')
linkage_cols = linkage(corr_matrix.T, method='average')

# Get the order of rows and columns after clustering
row_order = leaves_list(linkage_rows)
col_order = leaves_list(linkage_cols)

# Reorder the correlation matrix
corr_matrix_clustered = corr_matrix.iloc[row_order, col_order]

plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(
    corr_matrix_clustered, annot=True, cmap="vlag", center=0,
    linewidths=0.5, cbar_kws={"label": "Spearman r"}
)
plt.title("Clustered Spearman Correlation: GENCIC Bias & Mouse Model Conn Diff")
plt.tight_layout()
plt.show()

# %%
# Top HypoConnected vs GENCIC Bias 
from scipy.stats import hypergeom

def compute_hypergeometric_pvalue(N_total, N_set1, N_set2, N_common):
    """
    Compute the p-value for observing at least N_common overlap between two sets
    of size N_set1 and N_set2 drawn from a population of size N_total.
    """
    # P(X >= N_common)
    # sf is "survival function" = 1 - cdf, so sf(N_common-1) = P(X >= N_common)
    pval = hypergeom.sf(N_common-1, N_total, N_set1, N_set2)
    return pval

GENCIC_STRs = GENCIC_intersect[GENCIC_intersect["Circuits.46"] == 1].index.values
N_total_STR = 211
N_GENCIC = len(GENCIC_STRs)
N_top = 44
N_bottom = 44

for mousemodel in ["shank3b", "chd8", "cntnap2", "mecp2"]:
    for method in ["CSF_merge", "GSR_merge"]:
        col = f"{mousemodel}_{method}_conn_diff"
        col = GENCIC_intersect[col].sort_values(ascending=False)
        top46 = col.head(N_top)
        bottom44 = col.tail(N_bottom)
        Common_hyper = set(GENCIC_STRs).intersection(set(top46.index))
        Common_hypo = set(GENCIC_STRs).intersection(set(bottom44.index))
        pval_hyper = compute_hypergeometric_pvalue(N_total_STR, N_GENCIC, N_top, len(Common_hyper))
        pval_hypo = compute_hypergeometric_pvalue(N_total_STR, N_GENCIC, N_bottom, len(Common_hypo))
        print(f"{mousemodel} {method} | Hyper: {len(Common_hyper)} (p={pval_hyper:.4g}), Hypo: {len(Common_hypo)} (p={pval_hypo:.4g})")
        #print(Common_hyper)
        #print(Common_hypo)


# %%
