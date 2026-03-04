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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import STR2Region, MouseSTR_AvgZ_Weighted, Fil2Dict
from plot import plot_circuit_network, REGION_COLORS

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# # Circuit Network Visualization
#
# Automated generation of circuit network graphs (Figure 4 / Figure S10b).
# Nodes represent brain structures, colored by region and sized by mutation bias.
# Directed edges show significant ipsilateral connections (p<0.05, WeightMat).

# %% [markdown]
# ## 1. Load Data

# %%
with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# Connectivity matrix
WeightMat = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv", index_col=0)

# Bias
STR_BiasMat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")
ASD_GW = Fil2Dict(ProjDIR + "dat/Genetics/GeneWeights/Spark_Meta_EWS.GeneWeight.csv")
ASD_bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, ASD_GW)

# Region annotations
Anno = STR2Region()

print(f"WeightMat: {WeightMat.shape}")
print(f"Bias: {len(ASD_bias)} structures, top={ASD_bias['EFFECT'].iloc[0]:.4f} ({ASD_bias.index[0]})")

# %% [markdown]
# ## 2. Load Selected Circuits

# %%
RESULT_DIR = "../results/CircuitSearch/ASD_SPARK_61/pareto_fronts"
SELECTED_IDX = 2  # 3rd row after descending sort = high-bias knee (same as notebook 05)

circuits = {}
for size in [37, 46]:
    pareto = pd.read_csv(os.path.join(RESULT_DIR, f"ASD_SPARK_61_size_{size}_pareto_front.csv"))
    pareto = pareto.sort_values("mean_bias", ascending=False).reset_index(drop=True)
    row = pareto.iloc[SELECTED_IDX]
    strs = row["structures"].split(",")
    circuits[size] = strs
    n_edges = (WeightMat.loc[strs, strs] > 0).sum().sum()
    print(f"Size {size}: {len(strs)} structures, {n_edges} directed edges, "
          f"mean_bias={row['mean_bias']:.4f}, SI={row['circuit_score']:.4f}")

# %% [markdown]
# ## 3. Hand-Tuned Layout Positions
#
# Node positions tuned to reproduce the Cytoscape layout of Figure 4.
# Coordinate system: x [-1, 1] left-to-right, y [-1, 1] bottom-to-top.

# %%
# Positions tuned to match the published Figure 4 layout.
# Grouped by anatomical region / spatial band.
FIG4_POS = {
    # --- Top row: cortical areas ---
    'Agranular_insular_area_posterior_part':      (-0.30, 0.95),
    'Agranular_insular_area_ventral_part':        (-0.42, 0.82),
    'Infralimbic_area':                           (-0.12, 0.88),
    'Prelimbic_area':                             (0.05, 0.95),
    'Primary_somatosensory_area_trunk':           (-0.05, 0.75),
    'Primary_motor_area':                         (0.18, 0.82),
    'Posterior_parietal_association_areas':        (0.35, 0.90),
    'Secondary_motor_area':                       (0.30, 0.70),
    'Retrosplenial_area_lateral_agranular_part':  (0.62, 0.85),
    'Primary_visual_area':                        (0.72, 0.68),
    # --- Second row: cortical + gustatory ---
    'Orbital_area_lateral_part':                  (-0.55, 0.72),
    'Anterior_cingulate_area_dorsal_part':        (-0.20, 0.68),
    'Primary_somatosensory_area_lower_limb':      (0.08, 0.62),
    'Anteromedial_visual_area':                   (0.42, 0.58),
    'Lateral_visual_area':                        (0.55, 0.50),
    'Gustatory_areas':                            (0.58, 0.40),
    # --- Left column: orbital / olfactory / visceral ---
    'Orbital_area_ventrolateral_part':            (-0.68, 0.55),
    'Orbital_area_medial_part':                   (-0.62, 0.35),
    'Visceral_area':                              (-0.45, 0.58),
    'Claustrum':                                  (-0.50, 0.36),
    # --- Central hub: striatum ---
    'Nucleus_accumbens':                          (-0.15, 0.50),
    'Caudoputamen':                               (0.28, 0.50),
    # --- Thalamus cluster ---
    'Parataenial_nucleus':                        (0.05, 0.38),
    'Anteriomedial_nucleus_dorsal_part':          (0.35, 0.35),
    'Mediodorsal_nucleus_of_thalamus':            (-0.05, 0.25),
    'Nucleus_of_reuniens':                        (0.18, 0.22),
    'Lateral_posterior_nucleus_of_the_thalamus':  (0.62, 0.28),
    'Parafascicular_nucleus':                     (-0.20, 0.12),
    'Submedial_nucleus_of_the_thalamus':          (-0.35, 0.18),
    'Rhomboid_nucleus':                           (0.45, 0.10),
    # --- Cortical subplate ---
    'Endopiriform_nucleus_dorsal_part':           (-0.35, 0.28),
    # --- Septal / transition ---
    'Dorsal_peduncular_area':                     (-0.52, 0.00),
    'Lateral_septal_nucleus_rostral_rostroventral_part': (0.20, 0.02),
    'Lateral_septal_nucleus_ventral_part':        (0.00, 0.02),
    'Bed_nuclei_of_the_stria_terminalis':         (-0.05, 0.08),
    'Piriform_area':                              (-0.35, -0.08),
    # --- Hippocampus (lower right) ---
    'Dentate_gyrus':                              (0.45, -0.05),
    'Field_CA2':                                  (0.55, -0.12),
    'Field_CA3':                                  (0.45, -0.20),
    'Field_CA1':                                  (0.58, -0.22),
    'Subiculum_dorsal_part':                      (0.38, -0.25),
    'Subiculum_ventral_part':                     (0.25, -0.32),
    'Entorhinal_area_lateral_part':               (0.10, -0.12),
    # --- Amygdala (lower center) ---
    'Basolateral_amygdalar_nucleus':              (-0.10, -0.22),
    'Basomedial_amygdalar_nucleus':               (0.02, -0.28),
    'Lateral_amygdalar_nucleus':                  (0.15, -0.32),
    # --- Bottom periphery ---
    'Anterior_olfactory_nucleus':                 (-0.45, -0.20),
    'Anterior_pretectal_nucleus':                 (-0.52, -0.30),
}

# %% [markdown]
# ## 4. Size 46 — Main Circuit (Figure 4)

# %%
fig, ax = plot_circuit_network(
    circuits[46], ASD_bias, WeightMat, Anno,
    pos=FIG4_POS, figsize=(18, 12), font_size=7, node_scale=2500,
    title="ASD Circuit (Size 46) — Selected Operating Point",
)
fig.savefig("../results/figs/Figure4_circuit_network_46.png",
            dpi=300, transparent=True, bbox_inches='tight')
fig.savefig("../results/figs/Figure4_circuit_network_46.pdf",
            transparent=True, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Size 37 — FDR<0.10 Circuit
#
# 36 of 37 structures overlap with the size-46 circuit, so the same
# hand-tuned positions are reused (subset automatically).

# %%
fig, ax = plot_circuit_network(
    circuits[37], ASD_bias, WeightMat, Anno,
    pos=FIG4_POS, figsize=(18, 12), font_size=7, node_scale=2500,
    title="ASD Circuit (Size 37, FDR<0.10) — Selected Operating Point",
)
fig.savefig("../results/figs/circuit_network_37_fdr.png",
            dpi=300, transparent=True, bbox_inches='tight')
fig.savefig("../results/figs/circuit_network_37_fdr.pdf",
            transparent=True, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Graph Statistics

# %%
import networkx as nx

for size, strs in circuits.items():
    sub = WeightMat.loc[strs, strs]
    G = nx.DiGraph()
    G.add_nodes_from(strs)
    for src in strs:
        for tgt in strs:
            if src != tgt and sub.loc[src, tgt] > 0:
                G.add_edge(src, tgt)

    print(f"\n=== Size {size} ===")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Directed edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    print(f"  Avg clustering (undirected): {nx.average_clustering(G.to_undirected()):.3f}")

    # Top 5 by in-degree (most targeted)
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top in-degree: {[(s.replace('_',' '), d) for s, d in in_deg]}")

    # Top 5 by out-degree (most projecting)
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top out-degree: {[(s.replace('_',' '), d) for s, d in out_deg]}")
