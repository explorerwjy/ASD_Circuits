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
#     display_name: Python (gencic)
#     language: python
#     name: gencic
# ---

# %% [markdown]
# This is a notebook demonstrating CENCIC algorithm for circuit search with using ASD mutations

# %% [markdown]
# Requirement data file:
# 1. ASD mutation bias file: "Spark_Meta_EWS.Z2.bias.FDR.csv"
# 2. Information Score for connetome "Spark_Meta_EWS.Z2.info.csv"
#
# Scripts used in this notebook:
# 1. scripts/workflow/generate_bias_limits.py
# 2. scripts/workflow/run_sa_search.py
# 3. scripts/workflow/extract_best_circuits.py
# 4. scripts/workflow/create_pareto_front.py

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os

RootDIR = "/home/jw3514/Work/ASD_Circuits_CellType/" # put this in the right place
os.chdir(RootDIR + "/notebooks_mouse_str") # put this in the right place
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(1, RootDIR + 'src')
# Need to add src directory to Python path first

#sys.path.append("../src")
from ASD_Circuits import *

# %% [markdown]
# # 1. Calculate mutation strcture biases with ASD mutations

# %%
Spark_ASD_STR_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col=0)
Spark_ASD_STR_Bias.head(2)

# %% [markdown]
# # 2. Circuit Search with SA

# %% [markdown]
# ### 3.1 Calculate Biaslim for SA search

# %% [markdown]
# This will generate bias limits for different circuit sizes. In our paper we use size 46 as main search size since it has highest CCS score.
#
# The bias step size is 0.005 when bias > 0.3, 0.01 when bias > 0.2 and 0.05 when bias <= 0.2, to decrease computation burden. (we care less about the bias limits for low bias)

# %%
# Calculate bias limits for different circuit sizes
OutDIR = "../dat/CircuitSearch/Biaslims/"
BiasDF = Spark_ASD_STR_Bias

sizes = np.arange(10, 100, 1)
for i, t in enumerate(sizes):
    fout = open(OutDIR + "biaslim.size.{}.txt".format(t), 'w')
    writer = csv.writer(fout)
    lims = BiasLim(BiasDF, t)
    for size, bias in lims:
        writer.writerow([size, bias])
    fout.close()

# %%
Selected_BiasLim = pd.read_csv(OutDIR + "biaslim.size.46.txt", names=["size", "bias"])
Selected_BiasLim = Selected_BiasLim[Selected_BiasLim["bias"] >= 0.3] # select bias >= 0.3 to reduce number of jobs
Selected_BiasLim.reset_index(inplace=True, drop=True)
Selected_BiasLim.to_csv(OutDIR + "biaslim.size.46.top17.txt", index=False)


# %%
Selected_BiasLim


# %% [markdown]
# ### 3.2 run bash script to search circuits

# %% [markdown]
# # Circuit Search Using Simulated Annealing
# Now we can run the bash script `scripts/submit_job_run_pareto.SI.sh` to search for circuits using the selected bias limits.
#
# This is a computationally intensive process that may take 1-2 days to complete, depending on:
# - Number of parallel threads used
# - Size of the search space
# - Number of bias limits being explored

# %% [markdown]
# Here is the list of files/variables used in the bash script:
# - `BiasDF=../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv`: ASD mutation bias file
# - `AdjMat=../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv`: Connection weights for connetome
# - `InfoMat=../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv`: Information Score for connetome
# - `BiasLim=../dat/CircuitSearch/Biaslims/biaslim.size.46.txt`: Bias limits for different circuit sizes
# - `DIR=../dat/CircuitSearch/results/ASD_Pareto_46`:  Output directory
# - `NJob`: Number of total searches to complete the Pareto front (number of bias limits), calculated using `wc -l $BiasLim | cut -f 1 -d ' '`
# - `Nparallel=20`: Number of parallel threads used
#
# Fill the variables and run the bash script.
#

# %% [markdown]
# ### 3.3 Collect results and visualize

# %%
def searchFil(text, DIR):
    #print(text)
    RES = []
    for file in os.listdir(DIR):
        if text in file:
            RES.append(file)
    return RES

def LoadSA3(fname, DIR, InfoMat, minbias, topL=100):
    fin = open(DIR+fname, 'rt')
    max_score, max_bias, max_STRs = 0, 0, []
    for i, l in enumerate(fin):
        if i > topL:
            break
        l = l.strip().split()
        bias = float(l[1])
        if bias < minbias:
            continue
        STRs = l[2].split(",")
        score = ScoreCircuit_SI_Joint(STRs, InfoMat)
        if score > max_score:
            max_score = score
            max_bias = bias
            max_STRs = STRs
    return max_score, max_bias, max_STRs

def GetData2(params, size, DIR, adj_mat, InfoMat):
    SCORES, CutBias, RealBias, STRS = [],[],[],[]
    for i, row in params.iterrows():
        fil = searchFil("keepN_{}-minbias_{}.txt".format(size, row["bias"]), DIR)[0]
        score, real_minbias, STRs = LoadSA3(fil, DIR, InfoMat, row["bias"])
        score = ScoreCircuit_SI_Joint(STRs, InfoMat)
        if score == 0:
            continue
        SCORES.append(score)
        CutBias.append(row["bias"])
        RealBias.append(real_minbias)
        STRS.append(STRs)
    return SCORES, CutBias, RealBias, STRS

def XXXX_cont(BiasDF, BiasDF2, biaslim_df, size, DIR, adj_mat, InfoMat):
    #fil = searchFil("keepN_{}-minbias_{}.txt".format(size, bias), DIR)[0]
    SCORES, CutBias, RealBias, STRS = GetData2(biaslim_df, size, DIR, adj_mat, InfoMat)
    New_RealBias = []
    for STRSET in STRS:
        xx = BiasDF.loc[STRSET, "EFFECT"].mean()
        New_RealBias.append(xx)
    # Add top size STRs
    topNSTRs = BiasDF.index.values[:size]
    bias = BiasDF.head(size)["EFFECT"].mean()
    score = ScoreCircuit_SI_Joint(topNSTRs, InfoMat)
    SCORES.append(score)
    CutBias.append(bias)
    New_RealBias.append(bias)
    STRS.append(topNSTRs)    
    return SCORES, CutBias, New_RealBias, STRS

def LoadProfiles(BiasDF, BiasDF2, biaslim_df, size, DIR, adj_mat, InfoMat):
    Scores, CutBias, RealBias, STRS = GetData2(biaslim_df, size, DIR, adj_mat, InfoMat)
    # Add top size STRs
    topNSTRs = BiasDF.index.values[:size]
    bias = BiasDF2.head(size)["EFFECT"].mean()
    score = ScoreCircuit_SI_Joint(topNSTRs, InfoMat)
    Scores.append(score)
    CutBias.append(bias)
    RealBias.append(bias)
    STRS.append(topNSTRs)    
    return Scores, CutBias, RealBias, STRS


# %%
# Read connectome files
InfoMat = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv", index_col=0)
adj_mat = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv", index_col=0)
InfoMat_short = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.short.csv", index_col=0)
InfoMat_long = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.long.csv", index_col=0)

# %%
size = 46
#biaslim_df = pd.read_csv(biaslim_dir + "biaslim.size.{}.txt".format(size), names=["size", "bias"])
ASD_DIR = "../dat/CircuitSearch/SA/ASD_Pareto_SI_Size46/"
ASD_BiasDF = Spark_ASD_STR_Bias
biaslim_df = pd.read_csv(OutDIR + "biaslim.size.46.top17.txt")
COHESPeak, CutBiasPeak, RealBiasPeak, STRSPeak = LoadProfiles(ASD_BiasDF, ASD_BiasDF, biaslim_df, size, 
                                              ASD_DIR, adj_mat, InfoMat)
ASD_DFPeak = pd.DataFrame(data={"Cohe":COHESPeak, "minBias":CutBiasPeak, "Bias":RealBiasPeak})

# %%
plt.figure(dpi=120, figsize=(5,5))
plt.plot(ASD_DFPeak["Cohe"].values, ASD_DFPeak["Bias"].values, marker=".", color="#542788",  lw=2, markersize=8,
             ls = "-", label="ASD")
plt.scatter(ASD_DFPeak["Cohe"].values[-3], ASD_DFPeak["Bias"].values[-3], marker="x", s=50, color="red",
           zorder=100, label="Selected Circuits")

plt.xlabel("Circuit Score")
plt.ylabel("Mean Structure bias")
plt.grid()
plt.ylim((0.05, 0.4))
plt.legend()

# %%
# print the selected circuits
print(RegionDistributionsList(STRSPeak[-3]))

# %% [markdown]
# ## Plotting ASD and Sibling Circuit Data
#
# The following analysis compares ASD circuits with sibling control data.
#
# **Note:** The sibling data shown here is for visualization purposes only. The full analysis used data generated by running simulated annealing (SA) search with the same procedure on 10,000 subsampled sibling sets, which is too large to include here.
#
# To generate your own sibling data:
# 1. Use the bash script `scripts/submit_job_run_pareto.SI.sh` 
# 2. Run it with different sibling sets as input
#
# For details on the original procedure used to generate these profiles, see `Optimized_Circuits_Information_Score.ipynb`.
#

# %%
# Load variables from numpy file
sibling_data = np.load('../dat/CircuitSearch/SA/ASD_Pareto_SI_Size46/circuit_analysis_data.sibling.SA.npz')
meanbias = sibling_data['meanbias']
meanSI = sibling_data['meanSI']
topbias_sub = sibling_data['topbias_sub']

# %%
fig, ax = plt.subplots(dpi=480, figsize=(4.2,4))

ax.plot(ASD_DFPeak["Cohe"].values, ASD_DFPeak["Bias"].values, marker=".", color="#542788",  lw=2, markersize=8,
             ls = "-", label="ASD")
ax.scatter(ASD_DFPeak["Cohe"].values[-4], ASD_DFPeak["Bias"].values[-4], marker="x", s=70, color="red", lw=2,
           zorder=100)
ax.text(ASD_DFPeak["Cohe"].values[-4], 0.01 + ASD_DFPeak["Bias"].values[-4], s="Selected\n Circuit")

ax.plot(topbias_sub[:,0,:].T, topbias_sub[:,1,:].T, color="grey", markersize=1, lw=0.5,
             ls = "-", alpha=0.05)
#ax.plot(topbias_sub[0,0,:].T, topbias_sub[0,1,:].T, color="grey", markersize=1, lw=1,
#             ls = "-", alpha=1, label="Sibling Circuit")

ax.plot(meanSI, meanbias, marker=".", color="Orange", lw=2, markersize=8,
             ls = "-", alpha=1, label="Average Sibling Circuit")
ax.plot(meanSI, meanbias, color="grey", lw=2, markersize=8,
             ls = "-", alpha=1, label="Sibling Circuit", zorder=0)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
ax.legend(loc="lower left", frameon=False)

plt.xlabel("Circuit Connectivity Score", fontsize=14)
plt.ylabel("Average Mutation Bias", fontsize=14)
plt.grid(True, alpha=0.2)
plt.ylim(0.05, 0.42)

plt.show()

# %% [markdown]
# ## Bootstrap ASD

# %% [markdown]
# ### 3.5 Extract and Visualize Bootstrap Pareto Fronts
#
# In this section, we extract pareto front data from all bootstrap samples and visualize the variability across different bootstrap replicates.
#

# %%
# Extract all pareto front CSV files from different boot IDs
import glob
import shutil

# Find all pareto front CSV files
Boot_DIR = "../results/CircuitSearch_Bootstrap/"
pareto_files = glob.glob(Boot_DIR + "*/pareto_fronts/*_pareto_front.csv")
pareto_files.sort()

print(f"Found {len(pareto_files)} pareto front files:")
for f in pareto_files:
    print(f"  - {os.path.basename(os.path.dirname(os.path.dirname(f)))}: {os.path.basename(f)}")



# %%
# Read and combine all pareto front CSV files
all_pareto_data = []

for pf_file in pareto_files:
    # Extract boot ID from directory name
    boot_id = os.path.basename(os.path.dirname(os.path.dirname(pf_file)))
    
    # Read the CSV file
    df = pd.read_csv(pf_file)
    
    # Add boot_id column to track which bootstrap sample this is from
    df['boot_id'] = boot_id
    
    all_pareto_data.append(df)
    print(f"Loaded {boot_id}: {len(df)} rows")

# Combine all dataframes
combined_pareto_df = pd.concat(all_pareto_data, ignore_index=True)
print(f"\nTotal rows in combined dataframe: {len(combined_pareto_df)}")
print(f"Columns: {list(combined_pareto_df.columns)}")

combined_pareto_df.head()



# %%
Boot_DIR = "../results/CircuitSearch_Bootstrap/"

# %%
# Summary of pareto front data by boot_id
summary_by_boot = combined_pareto_df.groupby('boot_id').size().reset_index(name='num_points')
print("Number of pareto front points per bootstrap sample:")
summary_by_boot



# %%
# Separate plot for ASD_Boot samples vs main SPARK samples with 95% CI
from scipy import interpolate

fig, ax = plt.subplots(dpi=120, figsize=(8, 6))

# Filter to get only bootstrap samples (exclude SPARK_Main and SPARK_61)
boot_samples = [f'ASD_Boot{i}' for i in range(1000)]
boot_samples_in_data = [bid for bid in boot_samples if bid in combined_pareto_df['boot_id'].values]

# Create a common grid of circuit_score values for interpolation
all_circuit_scores = []
for boot_id in boot_samples_in_data:
    boot_data = combined_pareto_df[combined_pareto_df['boot_id'] == boot_id]
    all_circuit_scores.extend(boot_data['circuit_score'].values)

# Define interpolation grid
circuit_score_min = np.min(all_circuit_scores)
circuit_score_max = np.max(all_circuit_scores)
circuit_score_grid = np.linspace(circuit_score_min, circuit_score_max, 100)

# Interpolate each bootstrap sample onto the common grid
interpolated_bias_values = []
for boot_id in boot_samples_in_data:
    boot_data = combined_pareto_df[combined_pareto_df['boot_id'] == boot_id]
    boot_data = boot_data.sort_values('circuit_score')
    
    # Only interpolate if we have enough points
    if len(boot_data) >= 2:
        # Linear interpolation
        f_interp = interpolate.interp1d(
            boot_data['circuit_score'].values, 
            boot_data['mean_bias'].values,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        interpolated_bias = f_interp(circuit_score_grid)
        interpolated_bias_values.append(interpolated_bias)

# Convert to array: shape (n_bootstrap, n_grid_points)
interpolated_bias_array = np.array(interpolated_bias_values)

# Compute 95% CI (ignoring NaN values)
lower_ci = np.nanpercentile(interpolated_bias_array, 2.5, axis=0)
upper_ci = np.nanpercentile(interpolated_bias_array, 97.5, axis=0)
median_bias = np.nanmedian(interpolated_bias_array, axis=0)

# Plot 95% CI as shaded region
ax.fill_between(circuit_score_grid, lower_ci, upper_ci, 
                color='grey', alpha=0.3, label='95% CI (Bootstrap)', zorder=1)

# Plot median bootstrap pareto front
ax.plot(circuit_score_grid, median_bias, 
        color='grey', lw=2, ls='--', label='Median Bootstrap', zorder=5)

# Plot main SPARK samples in color
spark_samples = ['ASD_SPARK_Main', 'ASD_SPARK_61']
colors_spark = ['#542788', '#d95f02']
for i, spark_id in enumerate(spark_samples):
    if spark_id in combined_pareto_df['boot_id'].values:
        spark_data = combined_pareto_df[combined_pareto_df['boot_id'] == spark_id]
        spark_data = spark_data.sort_values('circuit_score')
        ax.plot(spark_data['circuit_score'], spark_data['mean_bias'], 
                marker='o', markersize=6, lw=2, 
                label=spark_id, color=colors_spark[i], zorder=10)

ax.set_xlabel("Circuit Connectivity Score", fontsize=12)
ax.set_ylabel("Average Mutation Bias", fontsize=12)
ax.set_title("Bootstrap Pareto Fronts with 95% CI", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

print(f"Computed 95% CI from {len(interpolated_bias_values)} bootstrap samples")



# %%
# Alternative plot: Show individual bootstrap lines WITH 95% CI
fig, ax = plt.subplots(dpi=120, figsize=(8, 6))

# Plot individual bootstrap samples as gray lines with low alpha
for boot_id in boot_samples_in_data[:100]:  # Show subset to avoid overcrowding
    boot_data = combined_pareto_df[combined_pareto_df['boot_id'] == boot_id]
    boot_data = boot_data.sort_values('circuit_score')
    ax.plot(boot_data['circuit_score'], boot_data['mean_bias'], 
            color='grey', lw=0.5, alpha=0.15, zorder=1)

# Plot 95% CI as shaded region (on top of individual lines)
ax.fill_between(circuit_score_grid, lower_ci, upper_ci, 
                color='lightblue', alpha=0.5, label='95% CI (Bootstrap)', zorder=5)

# Plot median bootstrap pareto front
ax.plot(circuit_score_grid, median_bias, 
        color='navy', lw=2.5, ls='-', label='Median Bootstrap', zorder=8)

# Plot main SPARK samples in color
for i, spark_id in enumerate(spark_samples):
    if spark_id in combined_pareto_df['boot_id'].values:
        spark_data = combined_pareto_df[combined_pareto_df['boot_id'] == spark_id]
        spark_data = spark_data.sort_values('circuit_score')
        ax.plot(spark_data['circuit_score'], spark_data['mean_bias'], 
                marker='o', markersize=6, lw=2.5, 
                label=spark_id, color=colors_spark[i], zorder=10)

ax.set_xlabel("Circuit Connectivity Score", fontsize=12)
ax.set_ylabel("Average Mutation Bias", fontsize=12)
ax.set_title("Bootstrap Pareto Fronts: Individual Lines + 95% CI", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()


# %%
# Save combined pareto front data to a summary file
summary_dir = "../results/CircuitSearch_Bootstrap_Summary/"
os.makedirs(summary_dir, exist_ok=True)

output_file = summary_dir + "all_pareto_fronts_size_46_combined.csv"
combined_pareto_df.to_csv(output_file, index=False)
print(f"Saved combined pareto front data to: {output_file}")



# %%
combined_pareto_df

# %% [markdown]
# ### Add Sibling

# %%
size = 46

# biaslim_df = pd.read_csv(
#     "../dat/Circuits/SA/biaslims2/biaslim.size.46.top17.txt", names=["size", "bias"])

# TODO: reproduce sibling SA data via pipeline (scripts/script_generate_geneweights.py
# with SiblingGenes(), then Snakefile.circuit). These paths point to the old repository.
SIB_SA_DIR = "/home/jw3514/Work/ASD_Circuits/dat/Circuits/SA/SubSib_Score_SI_Nov27_2023/"
SIB_BIAS_DIR = "/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib/"

dat_score = []
dat_bias = []

for i, file in enumerate(os.listdir(SIB_SA_DIR)):
    #try:
    d = os.path.join(SIB_SA_DIR, file)
    if os.path.isdir(d):
        biasdf = SIB_BIAS_DIR + file + ".csv"
        Sib_BiasDF = pd.read_csv(biasdf, index_col="STR")
        try:
            ASD_cont_Dir = SIB_SA_DIR + file + "/"
            COHES55, CutBias55, RealBias55, STRS55 = XXXX_cont(Sib_BiasDF, ASD_BiasDF, Selected_BiasLim, size, 
                                                          ASD_cont_Dir, adj_mat, InfoMat)
            dat_score.append(COHES55)
            dat_bias.append(RealBias55)
        except:
            continue

# %%
dat_score = np.array(dat_score)
dat_score_flat = dat_score.flatten()
dat_bias = np.array(dat_bias)
dat_bias_flat = dat_bias.flatten()

# %%
Sib_DF55 = pd.DataFrame(data={"SI score":dat_score_flat, "Bias":dat_bias_flat})

# %%
profiles = []
for bias, score in zip(dat_bias, dat_score):
    meanbias = np.mean(bias)
    meanscore = np.mean(score)
    meantotal = meanbias + meanscore
    profiles.append([meanbias, meanscore, meantotal, bias, score])

# %%
rank_bias = sorted(profiles, key = lambda x:x[0], reverse=True)
rank_score = sorted(profiles, key = lambda x:x[1], reverse=True)

# %%
topbias = []
for i in range(len(profiles)):
    topbias.append((rank_bias[i][4], rank_bias[i][3]))
topbias = np.array(topbias)
topSI = []
for i in range(len(profiles)):
    topSI.append((rank_score[i][4], rank_score[i][3]))
topSI = np.array(topSI)

# %%
meanbias = []
meanSI = []
xerr = []
yerr = []
for i in range(topbias.shape[2]):
    meanbias.append(np.nanmean(topbias[:,1,i]))
    meanSI.append(np.nanmean(topbias[:,0,i]))
    xerr.append(topbias[:,0,i].std())
    yerr.append(topbias[:,1,i].std())

# %%
rand_indexes = np.random.randint(0, topbias.shape[0], 1000)
topbias_sub = topbias[rand_indexes]

# %%
fig, ax = plt.subplots(dpi=480, figsize=(4.2,4))

ax.plot(ASD_DFPeak["Cohe"].values, ASD_DFPeak["Bias"].values, marker=".", color="#542788",  lw=2, markersize=8,
             ls = "-", label="ASD")
ax.scatter(ASD_DFPeak["Cohe"].values[-4], ASD_DFPeak["Bias"].values[-4], marker="x", s=70, color="red", lw=2,
           zorder=100)
# Plot 95% CI as shaded region (on top of individual lines)
ax.fill_between(circuit_score_grid, lower_ci, upper_ci, 
                color='lightblue', alpha=0.5, label='95% CI (Bootstrap)', zorder=5)

ax.plot(topbias_sub[:,0,:].T, topbias_sub[:,1,:].T, color="grey", markersize=1, lw=0.5,
             ls = "-", alpha=0.05)
#ax.plot(topbias_sub[0,0,:].T, topbias_sub[0,1,:].T, color="grey", markersize=1, lw=1,
#             ls = "-", alpha=1, label="Sibling Circuit")

ax.plot(meanSI, meanbias, marker=".", color="Orange", lw=2, markersize=8,
             ls = "-", alpha=1, label="Average Sibling Circuit")
ax.plot(meanSI, meanbias, color="grey", lw=2, markersize=8,
             ls = "-", alpha=1, label="Sibling Circuit", zorder=0)

# Draw the label ON TOP OF ALL LAYERS by setting high zorder
ax.text(
    ASD_DFPeak["Cohe"].values[-4],
    0.01 + ASD_DFPeak["Bias"].values[-4],
    s="Selected\n Circuit",
    zorder=1000,  # greater than any other element
    fontsize=12,
    ha='left'
)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
ax.legend(loc="lower left", frameon=False)

plt.xlabel("Circuit Connectivity Score", fontsize=14)
plt.ylabel("Average Mutation Bias", fontsize=14)
plt.grid(True, alpha=0.2)
plt.ylim(0.00, 0.44)

plt.show()

# %% [markdown]
# # Other Size

# %%
ASD_Pareto_45 = pd.read_csv("../results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_45_pareto_front.csv")
ASD_Pareto_45 = ASD_Pareto_45.sort_values('mean_bias', ascending=False).reset_index(drop=True)
ASD_Pareto_45.head()


# %%
# Plot main SPARK samples in color
fig, ax = plt.subplots(dpi=120, figsize=(8, 6))
ax.plot(ASD_Pareto_45['circuit_score'], ASD_Pareto_45['mean_bias'],
        marker='o', markersize=6, lw=2,
        label="ASD_SPARK_61 (size 45)", color='#542788', zorder=10)

ax.set_xlabel("Circuit Connectivity Score", fontsize=12)
ax.set_ylabel("Average Mutation Bias", fontsize=12)
ax.set_title("Pareto Front (size 45)", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

# %%
print(ASD_Pareto_45.loc[3, "mean_bias"], ASD_Pareto_45.loc[3, "circuit_score"])
print(RegionDistributionsList(ASD_Pareto_45.loc[3, "structures"].split(",")))

# %%
ASD_Pareto_41 = pd.read_csv("../results/CircuitSearch/ASD_SPARK_61/pareto_fronts/ASD_SPARK_61_size_41_pareto_front.csv")
ASD_Pareto_41 = ASD_Pareto_41.sort_values('mean_bias', ascending=False).reset_index(drop=True)
ASD_Pareto_41.head()

# %%
# Plot main SPARK samples in color
fig, ax = plt.subplots(dpi=120, figsize=(8, 6))
ax.plot(ASD_Pareto_41['circuit_score'], ASD_Pareto_41['mean_bias'],
        marker='o', markersize=6, lw=2,
        label="ASD_SPARK_61 (size 41)", color='#542788', zorder=10)

ax.set_xlabel("Circuit Connectivity Score", fontsize=12)
ax.set_ylabel("Average Mutation Bias", fontsize=12)
ax.set_title("Pareto Front (size 41)", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

# %%
print(RegionDistributionsList(ASD_Pareto_41.loc[2, "structures"].split(",")))

# %%
# print the selected circuits (size 46)
print(RegionDistributionsList(STRSPeak[-3]))

# %% [markdown]
# # Structure Overlap Analysis: Selected vs Bootstrap Circuits
#
# This section compares structure overlap between:
# 1. The selected circuit (3rd point on main pareto front)
# 2. Each bootstrap circuit (3rd point on each bootstrap pareto front)
#
# Also analyzes how frequently the top 10 structures (by bias) from the main circuit appear in bootstrap circuits.
#

# %%
# Get the selected circuit structures (3rd point from end, which is index -3)
selected_circuit_structures = set(STRSPeak[-3])
print(f"Selected circuit has {len(selected_circuit_structures)} structures")
print(f"Selected circuit structures: {sorted(selected_circuit_structures)}")


# %%
# Extract 3rd point (index 2) from each bootstrap pareto front
# Sort each bootstrap's pareto front by mean_bias descending to match the main circuit ordering
bootstrap_3rd_point_structures = []
bootstrap_3rd_point_data = []

boot_ids = combined_pareto_df['boot_id'].unique()
boot_ids = [bid for bid in boot_ids if bid.startswith('ASD_Boot')]  # Filter to bootstrap samples only
boot_ids.sort()

for boot_id in boot_ids:
    boot_data = combined_pareto_df[combined_pareto_df['boot_id'] == boot_id].copy()
    # Sort by mean_bias descending (highest bias first, matching main circuit order)
    boot_data = boot_data.sort_values('mean_bias', ascending=False).reset_index(drop=True)
    
    if len(boot_data) >= 3:  # Make sure we have at least 3 points
        # Get the 3rd point (index 2, 0-indexed)
        third_point = boot_data.iloc[2]
        structures_str = third_point['structures']
        structures_set = set(structures_str.split(','))
        
        bootstrap_3rd_point_structures.append(structures_set)
        bootstrap_3rd_point_data.append({
            'boot_id': boot_id,
            'structures': structures_set,
            'mean_bias': third_point['mean_bias'],
            'circuit_score': third_point['circuit_score']
        })
    else:
        print(f"Warning: {boot_id} has only {len(boot_data)} points, skipping")

print(f"Extracted 3rd point from {len(bootstrap_3rd_point_structures)} bootstrap samples")


# %%
# Calculate structure overlap between selected circuit and each bootstrap's 3rd point
overlap_scores = []
overlap_details = []

for i, boot_structures in enumerate(bootstrap_3rd_point_structures):
    # Calculate Jaccard similarity (intersection over union)
    intersection = selected_circuit_structures.intersection(boot_structures)
    union = selected_circuit_structures.union(boot_structures)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    
    # Calculate overlap percentage (intersection over selected circuit size)
    overlap_pct = len(intersection) / len(selected_circuit_structures) if len(selected_circuit_structures) > 0 else 0
    
    overlap_scores.append({
        'boot_id': bootstrap_3rd_point_data[i]['boot_id'],
        'jaccard_similarity': jaccard,
        'overlap_percentage': overlap_pct,
        'n_intersection': len(intersection),
        'n_selected': len(selected_circuit_structures),
        'n_bootstrap': len(boot_structures)
    })
    
    overlap_details.append({
        'boot_id': bootstrap_3rd_point_data[i]['boot_id'],
        'intersection': intersection,
        'only_in_selected': selected_circuit_structures - boot_structures,
        'only_in_bootstrap': boot_structures - selected_circuit_structures
    })

overlap_df = pd.DataFrame(overlap_scores)
print(f"Overlap statistics:")
print(f"  Mean Jaccard similarity: {overlap_df['jaccard_similarity'].mean():.4f}")
print(f"  Mean overlap percentage: {overlap_df['overlap_percentage'].mean():.4f}")
print(f"  Mean intersection size: {overlap_df['n_intersection'].mean():.2f}")
print(f"\nOverlap distribution:")
print(overlap_df[['jaccard_similarity', 'overlap_percentage', 'n_intersection']].describe())


# %%
# Visualize number of structures overlap
fig, ax = plt.subplots(dpi=120, figsize=(8, 5))

# Histogram of number of overlapping structures
mean_intersection = overlap_df['n_intersection'].mean()
ax.hist(overlap_df['n_intersection'], bins=30, edgecolor='black', alpha=0.7, color='#542788')
ax.axvline(mean_intersection, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_intersection:.1f}')
ax.set_xlabel('Number of Overlapping Structures', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
#ax.set_title('Structure Overlap: Number of Overlapping Structures', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean number of overlapping structures: {mean_intersection:.2f}")
print(f"Min: {overlap_df['n_intersection'].min()}, Max: {overlap_df['n_intersection'].max()}")
print(f"Median: {overlap_df['n_intersection'].median():.2f}")


# %%
# Visualize overlap distribution
fig, axes = plt.subplots(1, 2, dpi=120, figsize=(12, 5))

# Histogram of Jaccard similarity
axes[0].hist(overlap_df['jaccard_similarity'], bins=30, edgecolor='black', alpha=0.7, color='#542788')
axes[0].axvline(overlap_df['jaccard_similarity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {overlap_df["jaccard_similarity"].mean():.3f}')
axes[0].set_xlabel('Jaccard Similarity', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Structure Overlap: Jaccard Similarity', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogram of overlap percentage
axes[1].hist(overlap_df['overlap_percentage'], bins=30, edgecolor='black', alpha=0.7, color='#542788')
axes[1].axvline(overlap_df['overlap_percentage'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {overlap_df["overlap_percentage"].mean():.3f}')
axes[1].set_xlabel('Overlap Percentage', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Structure Overlap: Percentage of Selected Circuit', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Analysis 2: Top 20 structures by bias in main circuit
# Get top 20 structures from the SELECTED CIRCUIT, sorted by bias value
selected_circuit_structures_list = list(STRSPeak[-3])
# Get bias values for structures in the selected circuit
selected_circuit_bias = [(struct, Spark_ASD_STR_Bias.loc[struct, 'EFFECT']) for struct in selected_circuit_structures_list]
# Sort by bias descending
selected_circuit_bias_sorted = sorted(selected_circuit_bias, key=lambda x: x[1], reverse=True)
# Get top 20 structures from selected circuit by bias
top20_structures_by_bias = [struct for struct, bias in selected_circuit_bias_sorted[:20]]
top20_structures_by_bias_set = set(top20_structures_by_bias)

print(f"Top 20 structures by bias in the selected circuit:")
for i, (struct, bias) in enumerate(selected_circuit_bias_sorted[:20], 1):
    print(f"  {i}. {struct} (bias: {bias:.4f})")


# %%
# Check how many bootstrap circuits include each of the top 20 structures
top20_presence = {struct: [] for struct in top20_structures_by_bias_set}

for boot_structures in bootstrap_3rd_point_structures:
    for struct in top20_structures_by_bias_set:
        top20_presence[struct].append(1 if struct in boot_structures else 0)

# Calculate frequency for each structure
top20_frequency = {}
for struct in top20_structures_by_bias_set:
    presence_array = np.array(top20_presence[struct])
    frequency = presence_array.mean()
    top20_frequency[struct] = frequency
    print(f"{struct}: {frequency*100:.1f}% ({presence_array.sum()}/{len(presence_array)} bootstrap circuits)")

# Overall: how many bootstrap circuits include ALL top 20 structures?
n_bootstraps_with_all_top20 = 0
for boot_structures in bootstrap_3rd_point_structures:
    if top20_structures_by_bias_set.issubset(boot_structures):
        n_bootstraps_with_all_top20 += 1

pct_with_all_top20 = n_bootstraps_with_all_top20 / len(bootstrap_3rd_point_structures) * 100
print(f"\nBootstrap circuits that include ALL top 20 structures: {n_bootstraps_with_all_top20}/{len(bootstrap_3rd_point_structures)} ({pct_with_all_top20:.1f}%)")


# %% [markdown]
# ### Fraction of Bootstrap Circuits with All Top 10 Structures Together
#

# %%
# Derive top 10 from the top 20 structures already computed above
top10_structures_by_bias = top20_structures_by_bias[:10]
top10_structures_by_bias_set = set(top10_structures_by_bias)

# Also compute frequency for top 10 (needed later)
top10_frequency = {struct: top20_frequency[struct] for struct in top10_structures_by_bias_set}

# %%
# Calculate what fraction of bootstrap circuits contain ALL top 10 structures together
n_bootstraps_with_all_top10_together = 0
bootstraps_with_all_top10 = []
bootstraps_missing_some_top10 = []

for i, boot_structures in enumerate(bootstrap_3rd_point_structures):
    if top10_structures_by_bias_set.issubset(boot_structures):
        n_bootstraps_with_all_top10_together += 1
        bootstraps_with_all_top10.append(bootstrap_3rd_point_data[i]['boot_id'])
    else:
        missing = top10_structures_by_bias_set - boot_structures
        bootstraps_missing_some_top10.append({
            'boot_id': bootstrap_3rd_point_data[i]['boot_id'],
            'missing_structures': missing,
            'n_missing': len(missing)
        })

total_bootstraps = len(bootstrap_3rd_point_structures)
fraction_with_all_top10 = n_bootstraps_with_all_top10_together / total_bootstraps

print("=" * 70)
print("FRACTION OF BOOTSTRAP CIRCUITS WITH ALL TOP 10 STRUCTURES TOGETHER")
print("=" * 70)
print(f"\nTop 10 structures (by bias) in selected circuit:")
for i, struct in enumerate(top10_structures_by_bias, 1):
    print(f"  {i}. {struct}")

print(f"\n📊 Results:")
print(f"  Bootstrap circuits with ALL top 10 structures: {n_bootstraps_with_all_top10_together}/{total_bootstraps}")
print(f"  Fraction: {fraction_with_all_top10:.4f} ({fraction_with_all_top10*100:.2f}%)")
print(f"  Bootstrap circuits missing at least one top 10 structure: {total_bootstraps - n_bootstraps_with_all_top10_together}/{total_bootstraps}")

# Show which structures are most commonly missing
if bootstraps_missing_some_top10:
    missing_counts = {struct: 0 for struct in top10_structures_by_bias_set}
    for entry in bootstraps_missing_some_top10:
        for struct in entry['missing_structures']:
            if struct in missing_counts:
                missing_counts[struct] += 1
    
    print(f"\n📉 Most commonly missing structures:")
    missing_sorted = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)
    for struct, count in missing_sorted:
        if count > 0:
            pct_missing = count / total_bootstraps * 100
            print(f"  - {struct}: missing in {count}/{total_bootstraps} ({pct_missing:.1f}%) bootstrap circuits")


# %%
# Visualize the fraction of bootstrap circuits with all top 10 structures
fig, ax = plt.subplots(dpi=120, figsize=(8, 6))

# Create pie chart
labels = ['Has all top 10', 'Missing some top 10']
sizes = [n_bootstraps_with_all_top10_together, total_bootstraps - n_bootstraps_with_all_top10_together]
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)  # explode the first slice

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})

# Add count labels
for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
    autotext.set_text(f'{sizes[i]}/{total_bootstraps}\n({autotext.get_text()})')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax.set_title(f'Fraction of Bootstrap Circuits with All Top 10 Structures\n(Total: {total_bootstraps} bootstrap circuits)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# Also create a bar chart showing the breakdown
fig, ax = plt.subplots(dpi=120, figsize=(10, 6))

categories = ['Has all top 10\nstructures', 'Missing some top 10\nstructures']
counts = [n_bootstraps_with_all_top10_together, total_bootstraps - n_bootstraps_with_all_top10_together]
bar_colors = ['#2ecc71', '#e74c3c']

bars = ax.bar(categories, counts, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/total_bootstraps*100:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Number of Bootstrap Circuits', fontsize=12)
ax.set_title('Bootstrap Circuits: Presence of All Top 10 Structures Together', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# %%
# Check if 90% of bootstrap circuits include the top 10 structures
# Count how many of the top 10 structures appear in at least 90% of bootstrap circuits
structures_in_90pct = []
for struct in top10_structures_by_bias_set:
    frequency = top10_frequency[struct]
    if frequency >= 0.90:
        structures_in_90pct.append(struct)
        print(f"✓ {struct}: {frequency*100:.1f}% (meets 90% threshold)")
    else:
        print(f"✗ {struct}: {frequency*100:.1f}% (below 90% threshold)")

print(f"\nSummary:")
print(f"  Structures appearing in ≥90% of bootstrap circuits: {len(structures_in_90pct)}/10")
print(f"  Structures appearing in <90% of bootstrap circuits: {10 - len(structures_in_90pct)}/10")


# %%
# Enhanced visualization of top 20 structure frequency in bootstrap circuits

import matplotlib.patheffects as pe

fig, ax = plt.subplots(dpi=150, figsize=(11, 7))

# Use the top20_structures_by_bias that was calculated from the selected circuit (from previous cell)
# Make sure top20_structures_by_bias and top20_frequency are available from previous cells
structures_list = top20_structures_by_bias  # Keep original order (sorted by bias)
# Use top20_frequency which should be calculated in a previous cell
frequencies = [top20_frequency.get(s, 0) * 100 for s in structures_list]

# Prepare display labels: replace underscores with spaces
structures_list_display = [s.replace("_", " ") for s in structures_list]

# Define nice color palette for 20 values (using "crest" for background consistency)
import matplotlib.colors as mcolors
import seaborn as sns

base_cmap = sns.color_palette("crest", 20).as_hex()
color_map = []
for f in frequencies:
    if f >= 90:
        color_map.append(sns.color_palette("dark:#17B978", 15).as_hex()[8])  # green tone
    elif f >= 75:
        color_map.append(sns.color_palette("dark:#FFC300", 15).as_hex()[8])  # warm yellow/orange
    else:
        color_map.append(sns.color_palette("dark:#FF5733", 15).as_hex()[8])  # orange/red

bars = ax.barh(range(len(structures_list)), frequencies, color=color_map, alpha=0.87, 
               edgecolor='k', linewidth=1.3, height=0.67,
               zorder=10)

# Add value labels on the bars, with shadow
for i, (bar, freq) in enumerate(zip(bars, frequencies)):
    ax.text(freq + 1, bar.get_y() + bar.get_height() / 2, f'{freq:.1f}%',
            va='center', ha='left', fontsize=11, weight='bold', color=color_map[i],
            path_effects=[pe.withStroke(linewidth=2.7, foreground="white")])

# Enhance y-axis
ax.set_yticks(range(len(structures_list)))
ax.set_yticklabels(structures_list_display, fontsize=11, fontfamily='monospace')
ax.invert_yaxis()  # highest bias at top

# Titles and axis
ax.set_xlabel('Frequency in Bootstrap Circuits (%)', fontsize=13, labelpad=13)
#ax.set_title('Top 20 Structures (by Bias):\nPresence Across Bootstrap Circuits', 
#             fontsize=17, weight='bold', pad=18)
ax.set_xlim(0, 105)

# Remove box frame on top/right and prettify spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.1)
ax.spines['bottom'].set_linewidth(1.1)

# Grid and layout
ax.grid(True, alpha=0.18, axis='x', linestyle='-', zorder=1, color='#888C')
plt.tight_layout(rect=[0, 0, 0.88, 1])  # add right margin for legend, if desired
plt.show()


# %% [markdown]
# # Summary statistics table
# summary_stats = pd.DataFrame({
#     'Structure': structures_list,
#     'Bias_Value': [Spark_ASD_STR_Bias.loc[s, 'EFFECT'] for s in structures_list],
#     'Frequency_Percent': frequencies,
#     'In_Selected_Circuit': [s in selected_circuit_structures for s in structures_list],
#     'Meets_90pct_Threshold': [f >= 90 for f in frequencies]
# })
#
# # Already sorted by bias (descending) from the list creation
# print("Summary Statistics:")
# print(summary_stats.to_string(index=False))
#

# %%
# Plot presence across bootstrap for ALL structures in the selected circuit
# Calculate frequency for each structure in the selected circuit
all_structures_presence = {struct: [] for struct in selected_circuit_structures}

for boot_structures in bootstrap_3rd_point_structures:
    for struct in selected_circuit_structures:
        all_structures_presence[struct].append(1 if struct in boot_structures else 0)

# Calculate frequency for each structure
all_structures_frequency = {}
all_structures_bias = {}
for struct in selected_circuit_structures:
    presence_array = np.array(all_structures_presence[struct])
    frequency = presence_array.mean()
    all_structures_frequency[struct] = frequency
    all_structures_bias[struct] = Spark_ASD_STR_Bias.loc[struct, 'EFFECT']

# Sort structures by strongest bias (descending) for plotting (strongest = largest EFFECT)
all_structures_sorted = sorted(selected_circuit_structures, 
                               key=lambda x: all_structures_bias[x], 
                               reverse=True)
all_frequencies = [all_structures_frequency[s] * 100 for s in all_structures_sorted]
all_bias_values = [all_structures_bias[s] for s in all_structures_sorted]

# Prepare yticks to match strongest bias (top) to weakest bias (bottom, highest negative or lowest positive)
fig, ax = plt.subplots(dpi=120, figsize=(14, 8))

# Color code by frequency: green >=90%, orange >=75%, red <75%
colors = ['green' if f >= 90 else 'orange' if f >= 75 else 'red' for f in all_frequencies]
bars = ax.barh(range(len(all_structures_sorted)), all_frequencies, 
               color=colors, alpha=0.7, edgecolor='black')

# Add 90% threshold line
ax.axvline(90, color='red', linestyle='--', linewidth=2, label='90% threshold', zorder=0)
ax.axvline(75, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='75% threshold', zorder=0)

# Add value labels on bars
for i, (bar, freq) in enumerate(zip(bars, all_frequencies)):
    ax.text(freq + 1, i, f'{freq:.1f}%', va='center', fontsize=8)

# Set strongest bias on top (reverse yticks order)
ax.set_yticks(range(len(all_structures_sorted)))
ax.set_yticklabels(all_structures_sorted, fontsize=8)
ax.invert_yaxis()  # This puts the strongest biased at the top

ax.set_xlabel('Frequency in Bootstrap Circuits (%)', fontsize=12)
ax.set_title('All Selected Circuit Structures: Frequency in Bootstrap Circuits', fontsize=14)
ax.set_xlim(0, 105)
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Print summary
n_structures_90pct = sum(1 for f in all_frequencies if f >= 90)
n_structures_75pct = sum(1 for f in all_frequencies if f >= 75)
print(f"\nSummary for all {len(all_structures_sorted)} structures in selected circuit:")
print(f"  Structures appearing in ≥90% of bootstrap circuits: {n_structures_90pct}/{len(all_structures_sorted)} ({n_structures_90pct/len(all_structures_sorted)*100:.1f}%)")
print(f"  Structures appearing in ≥75% of bootstrap circuits: {n_structures_75pct}/{len(all_structures_sorted)} ({n_structures_75pct/len(all_structures_sorted)*100:.1f}%)")
print(f"  Structures appearing in <75% of bootstrap circuits: {len(all_structures_sorted)-n_structures_75pct}/{len(all_structures_sorted)} ({(len(all_structures_sorted)-n_structures_75pct)/len(all_structures_sorted)*100:.1f}%)")
