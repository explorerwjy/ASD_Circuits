#!/usr/bin/env python3
"""Plot CCS profiles from the seed-42 matching validation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/jw3514/Work/ASD_Circuits_CellType/notebook_validation"
CACHE_DIR = f"{OUTPUT_DIR}/cache_z2_comparison"

# Load the main profiles
ccs = pd.read_csv(f"{OUTPUT_DIR}/z2_seed42_ccs_profiles.csv")
seeds = pd.read_csv(f"{OUTPUT_DIR}/z2_seed_sensitivity.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_alpha(0)
for ax in axes:
    ax.patch.set_alpha(0)

# --- Panel 1: All four main pipelines ---
ax = axes[0]
ax.plot(ccs["Size"], ccs["Production"], "k-", lw=2.5, label="A) Production (Jon+legacy)", zorder=5)
ax.plot(ccs["Size"], ccs["Ours_legacy"], "--", color="#2196F3", lw=2, label="B) Ours+legacy match")
ax.plot(ccs["Size"], ccs["Ours_seed42"], "-", color="#E91E63", lw=2, label="C) Ours+seed42")
ax.plot(ccs["Size"], ccs["Jon_seed42"], "--", color="#FF9800", lw=1.5, label="D) Jon+seed42")

# Mark peaks
for col, color, marker in [
    ("Production", "k", "s"),
    ("Ours_legacy", "#2196F3", "D"),
    ("Ours_seed42", "#E91E63", "o"),
    ("Jon_seed42", "#FF9800", "^"),
]:
    sub = ccs[(ccs["Size"] >= 20) & (ccs["Size"] <= 60)]
    idx = sub[col].idxmax()
    ax.plot(sub.loc[idx, "Size"], sub.loc[idx, col], marker, color=color,
            markersize=10, zorder=6, markeredgecolor="white", markeredgewidth=1)

ax.axvline(46, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Circuit size (N structures)")
ax.set_ylabel("CCS")
ax.set_title("CCS Profiles: All Pipelines")
ax.legend(fontsize=8, loc="upper right")

# --- Panel 2: Zoom into 20-60 range ---
ax = axes[1]
sub = ccs[(ccs["Size"] >= 15) & (ccs["Size"] <= 65)]
ax.plot(sub["Size"], sub["Production"], "k-", lw=2.5, label="Production (peak=46)")
ax.plot(sub["Size"], sub["Ours_legacy"], "--", color="#2196F3", lw=2, label="Ours+legacy (peak=47)")
ax.plot(sub["Size"], sub["Ours_seed42"], "-", color="#E91E63", lw=2, label="Ours+seed42 (peak=27)")
ax.plot(sub["Size"], sub["Jon_seed42"], "--", color="#FF9800", lw=1.5, label="Jon+seed42 (peak=28)")

ax.axvline(46, color="gray", ls=":", alpha=0.5, label="size=46")
ax.set_xlabel("Circuit size (N structures)")
ax.set_ylabel("CCS")
ax.set_title("CCS Profiles: Zoomed (15-65)")
ax.legend(fontsize=8, loc="upper right")

# --- Panel 3: Seed sensitivity (seeds 0-9, all with our raw) ---
ax = axes[2]
# Recompute CCS for each seed from cached Z2
import sys, os
sys.path.insert(1, os.path.join("/home/jw3514/Work/ASD_Circuits_CellType", "src"))
import yaml
with open("/home/jw3514/Work/ASD_Circuits_CellType/config/config.yaml") as f:
    config = yaml.safe_load(f)
from ASD_Circuits import MouseSTR_AvgZ_Weighted, ScoreCircuit_SI_Joint, Fil2Dict

InfoMat = pd.read_csv(os.path.join("/home/jw3514/Work/ASD_Circuits_CellType",
                                    config["data_files"]["infomat_ipsi"]), index_col=0)
ASD_Weights = Fil2Dict(os.path.join("/home/jw3514/Work/ASD_Circuits_CellType",
                                     config["data_files"]["asd_gene_weights_v2"]))

colors_seed = plt.cm.tab10(np.linspace(0, 1, 10))
for seed_i in range(10):
    z2_path = os.path.join(CACHE_DIR, f"Z2_Ours_seed{seed_i}.parquet")
    if not os.path.exists(z2_path):
        continue
    z2 = pd.read_parquet(z2_path)
    bias = MouseSTR_AvgZ_Weighted(z2, ASD_Weights)
    bias_sorted = bias.sort_values("EFFECT", ascending=False)
    valid_strs = [s for s in bias_sorted.index if s in InfoMat.index]
    sizes = list(range(2, 81))
    ccs_vals = []
    for n in sizes:
        if n > len(valid_strs):
            ccs_vals.append(np.nan)
        else:
            ccs_vals.append(ScoreCircuit_SI_Joint(valid_strs[:n], InfoMat))
    ax.plot(sizes, ccs_vals, color=colors_seed[seed_i], alpha=0.5, lw=1,
            label=f"seed {seed_i}" if seed_i < 5 else None)
    # Mark peak
    sub_s = pd.DataFrame({"Size": sizes, "CCS": ccs_vals})
    sub_s = sub_s[(sub_s["Size"] >= 20) & (sub_s["Size"] <= 60)]
    idx = sub_s["CCS"].idxmax()
    pk_size = int(sub_s.loc[idx, "Size"])
    ax.plot(pk_size, sub_s.loc[idx, "CCS"], "o", color=colors_seed[seed_i],
            markersize=5, zorder=6)

# Overlay production for reference
ax.plot(ccs["Size"], ccs["Production"], "k-", lw=2.5, label="Production", zorder=5)
ax.axvline(46, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Circuit size (N structures)")
ax.set_ylabel("CCS")
ax.set_title("Seed Sensitivity (seeds 0-9, fresh match)")
ax.legend(fontsize=7, loc="upper right", ncol=2)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "z2_seed42_ccs_profiles.png")
plt.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
