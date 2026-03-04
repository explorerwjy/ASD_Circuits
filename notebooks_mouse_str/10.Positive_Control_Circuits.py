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

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
STR_BiasMat = pd.read_parquet("../dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet")
Anno = STR2Region()

# %%
csv_fil = "../dat/NeuralSystem.csv"
NeuralSystem = pd.read_csv(csv_fil, index_col=0)
ExpLevel = pd.read_csv("../dat/allen-mouse-exp/ExpMatchFeatures.csv", index_col=0)

NeuralSystem['entrez_id'] = NeuralSystem['entrez_id'].astype(int)
ExpLevel.index = ExpLevel.index.astype(int)

# Map the EXP value based on entrez_id
NeuralSystem['EXP'] = NeuralSystem['entrez_id'].map(ExpLevel['EXP'])

# %%
# Remove PCSK1 from oxytocin (pleiotropic proprotein convertase, not oxytocin-specific)
NeuralSystem = NeuralSystem[~((NeuralSystem["neurotransmitter_system"] == "oxytocin") & (NeuralSystem.index == "PCSK1"))]

# %%
# Define category groupings for source-target analysis
# Metabolism genes (COMT, MAOA, MAOB) excluded — broadly expressed catecholamine
# degradation enzymes that dilute spatial specificity of dopamine circuit signal
SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release', 'processing']
TARGET_CATEGORIES = ['receptor', 'degradation']

# Weight mode: "EXP" = expression-level weights (accounts for variation in receptor
# subtype abundance, e.g. DRD1/DRD2 >> DRD3-5, HTR1A/HTR2A >> other HTRs)
WEIGHT_MODE = "EXP"

# Run the source-target analysis — generates .gw files and computes bias
nt_results = analyze_neurotransmitter_systems_source_target(
    NeuralSystem,
    STR_BiasMat,
    SOURCE_CATEGORIES=SOURCE_CATEGORIES,
    TARGET_CATEGORIES=TARGET_CATEGORIES,
    weights=WEIGHT_MODE,
    generate_bias=True
)

# %% [markdown]
# ### Run Snakemake bias pipeline
# Compute null distributions and p-values for each NT system (combined gene set).
# This takes ~5-10 minutes with 10 cores (10K permutations per gene set).
# Skip this cell if results already exist in `results/STR_ISH/`.

# %%
import subprocess

NT_Systems = ["Dopamine", "Serotonin", "Oxytocin"]
targets = []
for system in NT_Systems:
    for null in ["sibling", "random"]:
        targets.append(f"results/STR_ISH/NT_{system}_combined_bias_addP_{null}.csv")

# Only run if any target is missing
missing = [t for t in targets if not os.path.exists(os.path.join(ProjDIR, t))]
if missing:
    print(f"Running Snakemake for {len(missing)} missing targets...")
    cmd = ["snakemake", "-s", "Snakefile.bias", "--forceall", "--cores", "10"] + targets
    result = subprocess.run(cmd, cwd=ProjDIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Snakemake failed with return code {result.returncode}")
    print("Snakemake completed successfully.")
else:
    print(f"All {len(targets)} NT bias files already exist, skipping Snakemake.")

# %%
# Load NT combined bias results (with sibling null p-values)
NT_BiasDict = {}
for system in NT_Systems:
    NT_BiasDict[system] = pd.read_csv(
        f"{ProjDIR}/results/STR_ISH/NT_{system}_combined_bias_addP_sibling.csv", index_col=0
    )

# %%
# Plot combined bias across NT systems
# plot_combined_bias_only expects {system: {'combined': df}} format
NT_BiasDict_nested = {system: {'combined': df} for system, df in NT_BiasDict.items()}
fig1 = plot_combined_bias_only(NT_BiasDict_nested, top_n=15, save_plot=True, dpi=120, pvalue_label="P-value")

# %% [markdown]
# ## Circuit Connectivity Score (CCS) analysis

# %%
ScoreMatDir = os.path.join(ProjDIR, "dat/allen-mouse-conn/ConnectomeScoringMat")
IpsiInfoMat = pd.read_csv(os.path.join(ScoreMatDir, "InfoMat.Ipsi.csv"), index_col=0)

topNs = np.arange(200, 5, -1)
DIR = os.path.join(ProjDIR, "dat/allen-mouse-conn/RankScores")
Cont_Distance = np.load(os.path.join(DIR, "RankScore.Ipsi.Cont.npy"))

# %%
def calculate_circuit_scores(bias_df, IpsiInfoMat, sort_by="EFFECT"):
    """Compute CCS profile across circuit sizes from 200 down to 6."""
    STR_Ranks = bias_df.sort_values(sort_by, ascending=False).index.values
    topNs = list(range(200, 5, -1))
    return np.array([ScoreCircuit_SI_Joint(STR_Ranks[:n], IpsiInfoMat) for n in topNs])

# %%
# Compute CCS profiles for each NT system (combined gene set)
nt_ccs_scores = {}
for system, bias_df in NT_BiasDict.items():
    nt_ccs_scores[system] = calculate_circuit_scores(bias_df, IpsiInfoMat)

# %%
# CCS profile plot: NT systems vs sibling null
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(1, 1, dpi=480, figsize=(12, 6), facecolor='none')
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)

BarLen = 34.1
for i, (system, scores) in enumerate(nt_ccs_scores.items()):
    ax1.plot(topNs, scores, color=colors[i], marker="o", markersize=5, lw=1,
             ls="dashed", label=system, alpha=0.5)

cont = np.median(Cont_Distance, axis=0)
lower = np.percentile(Cont_Distance, 50 - BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50 + BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
             yerr=(cont - lower, upper - cont), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 121)
ax1.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

# %%
# CCS p-values at key circuit sizes
plot_CCS_pvalues_at_N(50, nt_ccs_scores, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(20, nt_ccs_scores, topNs, Cont_Distance, colors)

# %%
# CCS histogram with NT bars
plot_CCS_histogram_with_NT_bars(50, nt_ccs_scores, Cont_Distance, topNs)
plot_CCS_histogram_with_NT_bars(20, nt_ccs_scores, Cont_Distance, topNs)

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
# Ensure negative control bias files exist (run Snakemake if needed)
neg_ctrl_sets = ["T2D", "Parkinson", "hba1c", "IBD", "HDL_C", "Alzheimer"]
neg_targets = [f"results/STR_ISH/{s}_bias_addP_sibling.csv" for s in neg_ctrl_sets]
neg_missing = [t for t in neg_targets if not os.path.exists(os.path.join(ProjDIR, t))]
if neg_missing:
    print(f"Running Snakemake for {len(neg_missing)} missing negative control targets...")
    cmd = ["snakemake", "-s", "Snakefile.bias", "--forceall", "--cores", "10"] + neg_missing
    result = subprocess.run(cmd, cwd=ProjDIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Snakemake failed with return code {result.returncode}")
    print("Done.")
else:
    print(f"All {len(neg_targets)} negative control bias files exist.")

DIR = os.path.join(ProjDIR, "results/STR_ISH")
T2D = pd.read_csv(os.path.join(DIR, "T2D_bias_addP_sibling.csv"), index_col=0)
Parkinson = pd.read_csv(os.path.join(DIR, "Parkinson_bias_addP_sibling.csv"), index_col=0)
HBALC = pd.read_csv(os.path.join(DIR, "hba1c_bias_addP_sibling.csv"), index_col=0)
IBD = pd.read_csv(os.path.join(DIR, "IBD_bias_addP_sibling.csv"), index_col=0)
HDL_C = pd.read_csv(os.path.join(DIR, "HDL_C_bias_addP_sibling.csv"), index_col=0)
Alzheimer = pd.read_csv(os.path.join(DIR, "Alzheimer_bias_addP_sibling.csv"), index_col=0)
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
# CCS p-values for negative controls
plot_CCS_pvalues_at_N(46, disorder_scores, topNs, Cont_Distance, colors)
plot_CCS_pvalues_at_N(20, disorder_scores, topNs, Cont_Distance, colors)

# %%
fig2, ax2 = plot_CCS_histogram_with_NT_bars(46, disorder_scores, Cont_Distance, topNs)
plt.show()
