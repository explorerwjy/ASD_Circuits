"""Re-aggregate all 10 batches of sibling results with correct batch numbering."""
import os, sys, numpy as np, pandas as pd

RESULT_DIR = "results/CircuitSearch_Sibling_Mutability"
AGG_DIR = "results/CircuitSearch_Sibling_Summary/Mutability/size_46"
PARQUET = "results/STR_ISH/null_bias/ASD_All_null_bias_sibling.parquet"
PREFIX = "ASD_Sib_Mutability_"
SIZE = 46

print("Loading full parquet...", flush=True)
bias_full = pd.read_parquet(PARQUET)
print(f"  Shape: {bias_full.shape}", flush=True)

for batch_idx in range(10):
    start = batch_idx * 1000
    end = (batch_idx + 1) * 1000

    sibling_data = []
    for sim_id in range(start, end):
        dataset_name = f"{PREFIX}{sim_id}"
        pf_path = os.path.join(RESULT_DIR, dataset_name, "pareto_fronts",
                               f"{dataset_name}_size_{SIZE}_pareto_front.csv")
        if not os.path.exists(pf_path):
            continue

        df = pd.read_csv(pf_path)
        df['sim_id'] = sim_id

        col = str(sim_id)
        real_biases = []
        for _, row in df.iterrows():
            if pd.notna(row.get('structures', np.nan)) and col in bias_full.columns:
                structs = row['structures'].split(',')
                valid = [s for s in structs if s in bias_full.index]
                real_bias = bias_full.loc[valid, col].mean() if valid else np.nan
            else:
                real_bias = np.nan
            real_biases.append(real_bias)

        df['transferred_bias'] = df['mean_bias']
        df['mean_bias'] = real_biases
        sibling_data.append(df)

    if not sibling_data:
        print(f"Batch {batch_idx}: NO DATA", flush=True)
        continue

    df_all = pd.concat(sibling_data, ignore_index=True)
    optimized = df_all[df_all['circuit_type'] == 'optimized']
    bias_limits = sorted(optimized['bias_limit'].unique())
    n_bias_limits = len(bias_limits)
    n_loaded = len(sibling_data)

    all_profiles = np.full((n_loaded, 2, n_bias_limits), np.nan)
    sim_ids_loaded = sorted(set(df_all['sim_id']))
    for idx, sim_id in enumerate(sim_ids_loaded):
        sib_opt = optimized[optimized['sim_id'] == sim_id]
        for j, bl in enumerate(bias_limits):
            row = sib_opt[sib_opt['bias_limit'] == bl]
            if len(row) > 0:
                all_profiles[idx, 0, j] = row['mean_bias'].iloc[0]
                all_profiles[idx, 1, j] = row['circuit_score'].iloc[0]

    meanbias = np.nanmean(all_profiles[:, 0, :], axis=0)
    meanSI = np.nanmean(all_profiles[:, 1, :], axis=0)

    batch_dir = os.path.join(AGG_DIR, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)
    npz_path = os.path.join(batch_dir, "sibling_profiles.npz")
    np.savez_compressed(npz_path, meanbias=meanbias, meanSI=meanSI,
                        topbias_sub=all_profiles, bias_limits=np.array(bias_limits))
    print(f"Batch {batch_idx} ({start}-{end-1}): {n_loaded} siblings, {n_bias_limits} limits, "
          f"bias=[{meanbias.min():.4f},{meanbias.max():.4f}], score=[{meanSI.min():.4f},{meanSI.max():.4f}]",
          flush=True)

# Create combined batch_all
print("\nCreating combined batch_all...", flush=True)
all_data = []
for batch_idx in range(10):
    npz = os.path.join(AGG_DIR, f"batch_{batch_idx}", "sibling_profiles.npz")
    d = np.load(npz)
    all_data.append(d['topbias_sub'])

combined = np.concatenate(all_data, axis=0)
meanbias_all = np.nanmean(combined[:, 0, :], axis=0)
meanSI_all = np.nanmean(combined[:, 1, :], axis=0)
bias_limits = np.load(os.path.join(AGG_DIR, "batch_0", "sibling_profiles.npz"))['bias_limits']

batch_all_dir = os.path.join(AGG_DIR, "batch_all")
os.makedirs(batch_all_dir, exist_ok=True)
np.savez_compressed(os.path.join(batch_all_dir, "sibling_profiles.npz"),
                    meanbias=meanbias_all, meanSI=meanSI_all,
                    topbias_sub=combined, bias_limits=bias_limits)
print(f"Combined: {combined.shape[0]} siblings, bias=[{meanbias_all.min():.4f},{meanbias_all.max():.4f}], "
      f"score=[{meanSI_all.min():.4f},{meanSI_all.max():.4f}]", flush=True)
print("DONE", flush=True)
