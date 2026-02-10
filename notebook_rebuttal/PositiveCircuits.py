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
from plot import *

try:
    os.chdir(f"{ProjDIR}/notebook_rebuttal/")
    print(f"Current working directory: {os.getcwd()}")
except FileNotFoundError as e:
    print(f"Error: Could not change directory - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
STR_BiasMat = pd.read_parquet("../dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet")
Anno = STR2Region()

# %%
csv_fil = "NeuralSystem.csv"
NeuralSystem = pd.read_csv(csv_fil, index_col=0)
ExpLevel = pd.read_csv("../dat/allen-mouse-exp/ExpMatchFeatures.csv", index_col=0)

NeuralSystem['entrez_id'] = NeuralSystem['entrez_id'].astype(int)
ExpLevel.index = ExpLevel.index.astype(int)

# Map the EXP value based on entrez_id
NeuralSystem['EXP'] = NeuralSystem['entrez_id'].map(ExpLevel['EXP'])

# %%
[NeuralSystem["neurotransmitter_system"].unique()]

# %%
NeuralSystem = NeuralSystem[~((NeuralSystem["neurotransmitter_system"] == "oxytocin") & (NeuralSystem.index == "PCSK1"))]

# %%
NeuralSystem[NeuralSystem["neurotransmitter_system"]=="oxytocin"]

# %%
STR_BiasMat.loc[1128,:].sort_values(ascending=False).head(10)

# %%
# Import functions from clean ASD_Circuits.py module
from ASD_Circuits import (
    analyze_neurotransmitter_systems_source_target, 
    plot_source_target_heatmap, 
    plot_source_target_only,
    save_neurotransmitter_results
)

# Define category groupings for this analysis
SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release', 'processing']
TARGET_CATEGORIES = ['receptor', 'degradation']

# Run the source-target analysis using imported function
print("="*70)
nt_results = analyze_neurotransmitter_systems_source_target(
    NeuralSystem, 
    STR_BiasMat, 
    SOURCE_CATEGORIES=SOURCE_CATEGORIES, 
    TARGET_CATEGORIES=TARGET_CATEGORIES,
    weights = "EXP",
    generate_bias=True
)

# %%
nt_results.keys()

# %%
nt = "acetylcholine"
nt_results[nt]["source"].head(10)

# %%
# drop  "acetylcholine" from nt_results

# %%
nt_results[nt]["target"].head(10)

# %%
# Load NT STR Bias results into a dictionary
NT_Systems = ["Dopamine", "Serotonin", "Oxytocin"]
NT_BiasDict = {}
for system in NT_Systems:
    NT_BiasDict[system] = {
        'source': pd.read_csv(f"{ProjDIR}/results/STR_ISH/NT_{system}_source_bias_addP_sibling.csv", index_col=0),
        'target': pd.read_csv(f"{ProjDIR}/results/STR_ISH/NT_{system}_target_bias_addP_sibling.csv", index_col=0),
        'combined': pd.read_csv(f"{ProjDIR}/results/STR_ISH/NT_{system}_combined_bias_addP_sibling.csv", index_col=0),
    }

# %%
NT_BiasDict["Dopamine"]["source"].head(10)

# %%
# Create visualizations and save results using imported functions
print("Creating source-target visualization plots...")

# Plot source vs target vs combined comparison and save to results directory
fig1 = plot_source_target_only(NT_BiasDict, top_n=10, save_plot=True, dpi=480, pvalue_label="P-value")

display_summary = False
if display_summary:
    # Display summary statistics
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS - SOURCE vs TARGET ANALYSIS")
    print("="*70)
    print(f"SOURCE categories: {SOURCE_CATEGORIES}")
    print(f"TARGET categories: {TARGET_CATEGORIES}")

    for system, system_data in NT_BiasDict.items():
        print(f"\n{system.upper()} SYSTEM:")
        
        if 'source' in system_data:
            top_structure = system_data['source'].index[0]
            top_effect = system_data['source'].iloc[0]['EFFECT']
            print(f"  Source: Top structure = {top_structure.replace('_', ' ')} (Effect: {top_effect:.3f})")
        
        if 'target' in system_data:
            top_structure = system_data['target'].index[0]
            top_effect = system_data['target'].iloc[0]['EFFECT']
            print(f"  Target: Top structure = {top_structure.replace('_', ' ')} (Effect: {top_effect:.3f})")
        
        if 'combined' in system_data:
            top_structure = system_data['combined'].index[0]
            top_effect = system_data['combined'].iloc[0]['EFFECT']
            print(f"  Combined: Top structure = {top_structure.replace('_', ' ')} (Effect: {top_effect:.3f})")

    print("\nSource-Target analysis complete!")
    print(f"All results saved to ./results/ directory")
    print("  - CSV files: Individual bias results for each system and category")
    print("  - PNG files: High-resolution plots for publication")
    print("\nEach system now has 3 DataFrames:")
    print("  - 'source': bias from synthesis + transporter + storage_release genes")
    print("  - 'target': bias from receptor + degradation genes") 
    print("  - 'combined': bias from source + target genes (excluding metabolism, processing)")

# %%
NT_BiasDict


# %%
# plot_combined_bias_only: moved to src/plot.py
fig1 = plot_combined_bias_only(NT_BiasDict, top_n=15, save_plot=True, dpi=120, pvalue_label="P-value")

# %% [markdown]
# ### Test Bias vs CCS 

# %%
ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v3/"
WeightMat = pd.read_csv(ScoreMatDir + "WeightMat.Ipsi.csv", index_col=0)
IpsiInfoMat=pd.read_csv(ScoreMatDir + "InfoMat.Ipsi.csv", index_col=0)
IpsiInfoMatShort_v1=pd.read_csv(ScoreMatDir + "InfoMat.Ipsi.Short.3900.csv", index_col=0)
IpsiInfoMatLong_v1=pd.read_csv(ScoreMatDir + "InfoMat.Ipsi.Long.3900.csv", index_col=0)

topNs = np.arange(200, 5, -1)
DIR = "/home/jw3514/Work/ASD_Circuits/scripts/RankScores/"
Cont_Distance = np.load("{}/RankScore.Ipsi.Cont.npy".format(DIR))
Cont_DistanceShort = np.load("{}/RankScore.Ipsi.Short.3900.Cont.npy".format(DIR))
Cont_DistanceLong = np.load("{}/RankScore.Ipsi.Long.3900.Cont.npy".format(DIR))


# %%
def calculate_circuit_scores(pc_scores_df, IpsiInfoMat, sort_by="PC1"):
    STR_Ranks = pc_scores_df.sort_values(sort_by, ascending=False).index.values
    topNs = list(range(200, 5, -1))
    SC_Agg_topN_score = []
    
    for topN in topNs:
        top_strs = STR_Ranks[:topN]
        score = ScoreCircuit_SI_Joint(top_strs, IpsiInfoMat)
        SC_Agg_topN_score.append(score)
        
    return np.array(SC_Agg_topN_score)


# %%
print("Keys in nt_results_detailed:")
for key in nt_results.keys():
    print(f"  {key}: {list(nt_results[key].keys())}")

# %%
#NT_BiasDict = nt_results

# %%
# For each neurotransmitter system, create a new DataFrame where each structure's EFFECT is the max of source/target, but in final df use column name "EFFECT"
NT_BiasDict_max = {}
for system in NT_BiasDict.keys():
    source = NT_BiasDict[system].get("source")
    target = NT_BiasDict[system].get("target")
    if source is not None and target is not None:
        # Align source and target indices, take max per structure
        aligned = source["EFFECT"].to_frame("source").join(
            target["EFFECT"].to_frame("target"), how="outer"
        )
        aligned["EFFECT"] = aligned[["source", "target"]].max(axis=1)
        # You might want to add back region/other columns if needed
        df_max = aligned[["EFFECT"]].copy()
        # Optionally add Region info from source or target
        if "Region" in source.columns:
            df_max["Region"] = source["Region"]
        elif "Region" in target.columns:
            df_max["Region"] = target["Region"]
        df_max = df_max.sort_values("EFFECT", ascending=False)
        NT_BiasDict_max[system] = df_max

# %%
neuro_system_scores = {}
for system in NT_BiasDict.keys():
    for category, df in NT_BiasDict[system].items():
        neuro_system_scores[f"{system}_{category}"] = calculate_circuit_scores(df, IpsiInfoMat, sort_by="EFFECT")

neuro_system_scores2 = {}
for system in NT_BiasDict_max.keys():
    df = NT_BiasDict_max[system]
    neuro_system_scores2[f"{system}"] = calculate_circuit_scores(df, IpsiInfoMat, sort_by="EFFECT")

# %%
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1,1, dpi=480, figsize=(12,6), facecolor='none')

fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1
#BarLen = 47.5

# Define colors for each neurotransmitter system
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

topNs = list(range(200, 5, -1))  # Define topNs based on the range used in calculate_circuit_scores
                        
for i, (system_category, scores) in enumerate(neuro_system_scores.items()):
    if "combined" in system_category:
        label = system_category.replace("_combined", "")
        color = colors[i % len(colors)]
        ax1.plot(topNs, scores, color=color, marker="o", markersize=5, lw=1,
                            ls="dashed", label=label, alpha = 0.5)

cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)

# Place legend outside of plot
ax1.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent legend cutoff

# %%
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1,1, dpi=480, figsize=(12,6), facecolor='none')

fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1
#BarLen = 47.5

# Define colors for each neurotransmitter system
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

topNs = np.arange(200, 5, -1)  # Define topNs based on the range used in calculate_circuit_scores
                        
for i, (system_category, scores) in enumerate(neuro_system_scores2.items()):

    color = colors[i % len(colors)]
    ax1.plot(topNs, scores, color=color, marker="o", markersize=5, lw=1,
                        ls="dashed", label=system_category, alpha = 0.5)

cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)

# Place legend outside of plot
ax1.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent legend cutoff


# %%
# plot_CCS_pvalues_at_N: moved to src/plot.py

# %%
#plot_CCS_pvalues_at_N(46, neuro_system_scores, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(46, neuro_system_scores2, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(40, neuro_system_scores2, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(20, neuro_system_scores2, topNs, Cont_Distance, colors)

# %%
plot_CCS_pvalues_at_N(40, neuro_system_scores2, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(20, neuro_system_scores2, topNs, Cont_Distance, colors)

# %%
plot_CCS_histogram_with_NT_bars(20, neuro_system_scores2, Cont_Distance, topNs)
plot_CCS_histogram_with_NT_bars(40, neuro_system_scores2, Cont_Distance, topNs)


# %%
plot_CCS_histogram_with_NT_bars(50, neuro_system_scores2, Cont_Distance, topNs)

# %% [markdown]
# # Other disorders 

# %%
PKList = ["SNCA", "PRKN", "PARK7", "PINK1", "LRRK2"]
PK_GW = dict(zip([GeneSymbol2Entrez[x] for x in PKList], [1]*len(PKList)))
Dict2Fil(PK_GW, "../dat/Genetics/GeneWeights/Parkinson.gw")
PK_STR_Bias  = MouseSTR_AvgZ_Weighted(STR_BiasMat, PK_GW)
PK_STR_Bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in PK_STR_Bias.index.values]

# %%
PK_STR_Bias.head(50)

# %%
# NegativeCircuits
from asyncio import Handle


DIR = "~/Work/ASD_Circuits_CellType/results/STR_ISH/"
T2D = pd.read_csv(DIR + "T2D_bias_addP_sibling.csv", index_col=0)
Parkinson = pd.read_csv(DIR + "Parkinson_bias_addP_sibling.csv", index_col=0)
HBALC = pd.read_csv(DIR + "hba1c_bias_addP_sibling.csv", index_col=0)
IBD = pd.read_csv(DIR + "IBD_bias_addP_sibling.csv", index_col=0)
HDL_C = pd.read_csv(DIR + "HDL_C_bias_addP_sibling.csv", index_col=0)
Alzheimer = pd.read_csv(DIR + "Alzheimer_bias_addP_sibling.csv", index_col=0)
#ASD = pd.read_csv(DIR + "ASD_bias_addP_sibling.csv", index_col=0)
ASD = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")

# %%
# Calculate circuit scores for each disorder dataset
disorder_scores = {}
disorder_datasets = {
    "T2D": T2D,
    # "Parkinson": Parkinson,
    # "HBALC": HBALC,
    "IBD": IBD,
    "HDL_C": HDL_C,
    # "Alzheimer": Alzheimer,
    "ASD": ASD
}

# Fixed color dictionary - ASD is blue, other disorders each get a specific color
disorder_colors = {
    "ASD": "#1f77b4",    # blue
    "T2D": "#ff7f0e",    # orange
    "IBD": "#2ca02c",    # green
    "HDL_C": "#d62728",  # red
    # Add more if enabling more disorders (e.g., Parkinson, HBALC, Alzheimer)
    "Parkinson": "#9467bd", # purple
    "HBALC": "#8c564b",     # brown
    "Alzheimer": "#e377c2"  # pink
}

for disorder_name, disorder_df in disorder_datasets.items():
    disorder_scores[disorder_name] = calculate_circuit_scores(disorder_df, IpsiInfoMat, sort_by="EFFECT")


# %%
# Plot rank vs CCS for all disorders
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1,1, dpi=480, figsize=(12,6), facecolor='none')

fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1

# Define a color palette, but force ASD to blue (#1f77b4), others to use different colors
asd_color = '#1f77b4'
other_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

topNs = list(range(200, 5, -1))  # Define topNs based on the range used in calculate_circuit_scores
color_idx = 0
for disorder_name, scores in disorder_scores.items():
    if disorder_name.upper() == "ASD":
        color = asd_color
        zorder = 10  # Bring ASD to the front if overlapped
        alpha = 1.0
        linewidth = 2.5
    else:
        color = other_colors[color_idx % len(other_colors)]
        color_idx += 1
        zorder = 2
        alpha = 0.7
        linewidth = 1
    ax1.plot(topNs, scores, color=color, marker="o", markersize=5, lw=linewidth,
             ls="dashed", label=disorder_name, alpha=alpha, zorder=zorder)

# Add sibling controls
cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)

# Place legend outside of plot
ax1.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent legend cutoff


# %%
# Plot CCS p-values for negative controls (disorders)
# Ensure topNs is a numpy array for the function (it's already defined in cell 17, but ensure it's numpy array)
topNs = np.arange(200, 5, -1)

# Define colors if not already defined (needed for plot_CCS_pvalues_at_N)
if 'colors' not in globals():
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

# Plot p-values at different ranks
plot_CCS_pvalues_at_N(46, disorder_scores, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(20, disorder_scores, topNs, Cont_Distance, colors)


# %%
# plot_CCS_histogram_with_NT_bars: moved to src/plot.py (with overlap fix)
fig2, ax2 = plot_CCS_histogram_with_NT_bars(46, disorder_scores, Cont_Distance, topNs)
plt.show()

# %%
