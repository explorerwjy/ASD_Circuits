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

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/" # Change to your project directory
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

try:
    os.chdir(f"{ProjDIR}/notebooks_figures/")
    print(f"Current working directory: {os.getcwd()}")
except FileNotFoundError as e:
    print(f"Error: Could not change directory - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
import pandas as pd

# Define the input and output Excel file paths
input_excel = "Supplementary_Tables_updated_sub.xlsx"
output_excel = "Supplementary_Tables_updated_sub_new.xlsx"  # Different output file

# Read in the existing supplementary Excel file (if sheet names are unknown, this will scan them)
try:
    with pd.ExcelFile(input_excel) as xls:
        sheet_names = xls.sheet_names
        existing_tables = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names}
        print(f"Successfully read {len(existing_tables)} existing tables from {input_excel}")
except FileNotFoundError:
    # File does not exist yet, so start with empty tables dictionary
    existing_tables = {}
    print(f"Warning: {input_excel} not found. Starting with empty tables dictionary.")

# Initialize a dictionary to store new tables you want to add
new_tables = {}

# Function to save all tables (existing + new) to a new Excel file
def save_supplementary_tables(output_filepath, new_tables_dict, existing_tables_dict):
    """
    Combine existing tables with new ones and save to a new Excel file.
    New tables can overwrite existing sheets of the same name.
    """
    # Combine existing tables with new ones (new ones can overwrite existing sheets of the same name)
    all_tables = {**existing_tables_dict, **new_tables_dict}
    with pd.ExcelWriter(output_filepath, engine='openpyxl', mode='w') as writer:
        for sheet, df in all_tables.items():
            # Truncate sheet names to 31 characters to avoid Excel warnings
            sheet_name = sheet[:31] if len(sheet) > 31 else sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Successfully saved {len(all_tables)} tables to {output_filepath}")



# %%
# # Supplementary Table 6: Neurotransmitter genes included in validation analysis.
# TableS6 = pd.read_csv("NeuralSystem.csv", index_col=0)
# TableS6_name = 'Table-S6-NeuroTransmitter Genes'

# # Add TableS6 to the dict of new tables to be added to the Excel file
# new_tables[TableS6_name] = TableS6

# # Save all supplementary tables including the new one
# save_supplementary_tables(V1_TabS, new_tables, existing_tables)

# %%
# Update Table1. I need replace orignal column of "Bias.ASC102" to "Bias.Fu_72" and add a new column of "Bias.Fu_185"
# Use column "EFFECT" from Fu_72 and Fu_185 to replace the original column "Bias.ASC102" in Table1.
# Need to insert at proper column position.
Fu_72 = pd.read_csv("../results/STR_ISH/Fu_ASD_72_bias_addP_sibling.csv")
Fu_185 = pd.read_csv("../results/STR_ISH/Fu_ASD_185_bias_addP_sibling.csv")

# Find Table1 in existing_tables (look for sheet names containing "Table-1" or "Table1" or "Table 1")
Table1_sheet_name = 'Table-S1- Structure Bias'
for sheet_name in existing_tables.keys():
    if 'Table-1' in sheet_name or 'Table1' in sheet_name or 'Table 1' in sheet_name:
        Table1_sheet_name = sheet_name
        break

if Table1_sheet_name is None:
    print("Warning: Could not find Table1 in existing tables. Available sheets:", list(existing_tables.keys()))
else:
    print(f"Found Table1: {Table1_sheet_name}")
    Table1 = existing_tables[Table1_sheet_name].copy()
    
    # Check if Bias.ASC102 column exists
    if 'Bias.ASC102' not in Table1.columns:
        print(f"Warning: Column 'Bias.ASC102' not found in Table1. Available columns: {list(Table1.columns)}")
    else:
        # Find the position of Bias.ASC102 column
        bias_asc102_idx = list(Table1.columns).index('Bias.ASC102')
        print(f"Found 'Bias.ASC102' at column index: {bias_asc102_idx}")
        
        # Find the matching column for Structure (could be 'STR', 'Structure', 'Acronym', etc.)
        merge_col_table1 = None
        for col in ['STR', 'Structure', 'structure', 'Acronym', 'ACRONYM']:
            if col in Table1.columns:
                merge_col_table1 = col
                break
        
        if merge_col_table1 is None:
            print(f"Warning: Could not find Structure column in Table1. Available columns: {list(Table1.columns)}")
        else:
            print(f"Using '{merge_col_table1}' column from Table1 for alignment")
            
            # Create mapping dictionaries from Structure to EFFECT
            Fu_72_map = dict(zip(Fu_72['Structure'], Fu_72['EFFECT']))
            Fu_185_map = dict(zip(Fu_185['Structure'], Fu_185['EFFECT']))
            
            # Map EFFECT values based on Structure alignment
            Table1['Bias.Fu_72'] = Table1[merge_col_table1].map(Fu_72_map)
            Table1['Bias.Fu_185'] = Table1[merge_col_table1].map(Fu_185_map)
            
            # Remove the old Bias.ASC102 column
            Table1 = Table1.drop(columns=['Bias.ASC102'])
            
            # Reorder columns to insert Bias.Fu_72 and Bias.Fu_185 at the position where Bias.ASC102 was
            cols = list(Table1.columns)
            # Remove Bias.Fu_72 and Bias.Fu_185 from their current positions
            cols = [c for c in cols if c not in ['Bias.Fu_72', 'Bias.Fu_185']]
            # Insert them at the original Bias.ASC102 position
            cols.insert(bias_asc102_idx, 'Bias.Fu_72')
            cols.insert(bias_asc102_idx + 1, 'Bias.Fu_185')
            Table1 = Table1[cols]
            
            # Update Table1 in existing_tables and also add to new_tables to ensure it gets saved
            existing_tables[Table1_sheet_name] = Table1
            new_tables[Table1_sheet_name] = Table1
            print(f"Successfully updated Table1: replaced 'Bias.ASC102' with 'Bias.Fu_72' and added 'Bias.Fu_185'")
            print(f"New columns around position {bias_asc102_idx}: {cols[bias_asc102_idx:bias_asc102_idx+3]}")



# %%

# %%
# Supplementary Table 7: Dopamine Bias

TableS7 = pd.read_csv("../results/STR_ISH/NT_Dopamine_combined_bias_addP_sibling.csv")
TableS7_name = 'Table-S7-Dopamine Bias'

# Add TableS7 to the dict of new tables to be added to the Excel file
new_tables[TableS7_name] = TableS7

# %%
# Supplementary Table 8: Serotonin Bias

TableS8 = pd.read_csv("../results/STR_ISH/NT_Serotonin_combined_bias_addP_sibling.csv")
TableS8_name = 'Table-S8-Serotonin Bias'

# Add TableS8 to the dict of new tables to be added to the Excel file
new_tables[TableS8_name] = TableS8

# %%
# Supplementary Table 9: Oxytocin Bias

TableS9 = pd.read_csv("../results/STR_ISH/NT_Oxytocin_combined_bias_addP_sibling.csv")
TableS9_name = 'Table-S9-Oxytocin Bias'

# Add TableS9 to the dict of new tables to be added to the Excel file
new_tables[TableS9_name] = TableS9

# %%
# Supplementary Table 10: DDD (exclude ASD) Structure Bias
TableS10 = pd.read_csv("../results/STR_ISH/DDD_293_ExcludeASD_bias_addP_sibling.csv")
TableS10_name = 'Table-S10-DDD (exclude ASD) Structure Bias'

# Add TableS10 to the dict of new tables to be added to the Excel file
new_tables[TableS10_name] = TableS10

# %%
# Supplementary Table 11: DDD (exclude ASD) Cell type Bias

TableS11 = pd.read_csv("../results/CT_Z2/DDD_293_ExcludeASD_bias_addP_sibling.csv")
TableS11_name = 'Table-S11-DDD (exclude ASD) Cell type Bias'

# Add TableS11 to the dict of new tables to be added to the Excel file
new_tables[TableS11_name] = TableS11

# %%
# Supplementary Table 12: Constraint gene Structure Bias
TableS12 = pd.read_csv("../results/STR_ISH/Constraint_top25_LOEUF_bias_addP_random.csv")
TableS12_name = 'Table-S12-Constraint gene Structure Bias'

# Add TableS12 to the dict of new tables to be added to the Excel file
new_tables[TableS12_name] = TableS12

# %%
# Supplementary Table 13: Constraint gene Cell type Bias
TableS13 = pd.read_csv("../results/CT_Z2/Constraint_top25_LOEUF_bias_addP_random.csv")
TableS13_name = 'Table-S13-Constraint gene Cell type Bias'

# Add TableS13 to the dict of new tables to be added to the Excel file
new_tables[TableS13_name] = TableS13

# %%
# # Supplementary Table 14: Gencic vs Mouse fMRI
# TableS14 = pd.read_csv("/home/jw3514/Work/ASD_Circuits_CellType/results/FMRI_GENCIC_Combined_Table.csv")
# TableS14_name = 'Table-S14-Gencic vs Mouse fMRI'

# # Add TableS14 to the dict of new tables to be added to the Excel file
# new_tables[TableS14_name] = TableS14

# %%
# Supplementary Table 15: Human fMRI and GENCIC comparison
TableS14 = pd.read_excel("../notebook_rebuttal/Buch_et_al/claude_mapping_v6_DCG.xlsx")  # Placeholder - update with correct file
TableS14_name = 'Table-S14-Human-Mouse region mapping'

# Add TableS15 to the dict of new tables to be added to the Excel file
new_tables[TableS14_name] = TableS14

# %%
# Save all supplementary tables (existing + new) to the output Excel file
save_supplementary_tables(output_excel, new_tables, existing_tables)

# %% [markdown]
# # Supplementary Table 1: ASD Bias & Qvalue & Other

# %% [markdown]
# #### ASD Bias & Qvalue & Other

# %%
ASD_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv", index_col="STR")

# %%
# ASD Bias Qvalue
import statsmodels.stats as stats

def getBiasesBySTR(STR, dfs):
    biases = []
    for df in dfs:
        bias = df.loc[STR, "EFFECT"]
        biases.append(bias)
    biases = np.array(biases)
    return biases

ASD_Sim_dir = "../dat/Unionize_bias/SubSampleSib/"

cont_dfs = []
for file in os.listdir(ASD_Sim_dir):
    if file.startswith("cont.genes"):
        continue
    df = pd.read_csv(ASD_Sim_dir+file, index_col="STR")
    cont_dfs.append(df)

for STR, row in ASD_Bias.iterrows():
    mat_bias = getBiasesBySTR(STR, cont_dfs)
    Z, P = GetPermutationP(mat_bias, row["EFFECT"])
    ASD_Bias.loc[STR, "Pvalue"] = P
    ASD_Bias.loc[STR, "Z_Match"] = Z

acc, qvalues = stats.multitest.fdrcorrection(ASD_Bias["Pvalue"].values, alpha=0.1,
                                                   method="i")
print(sum(acc))
ASD_Bias["qvalues"] = qvalues
ASD_Bias.to_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.subsib.FDR.csv")

# %%
acc, qvalues = stats.multitest.fdrcorrection(ASD_Bias["Pvalue"].values, alpha=0.05,
                                                   method="i")
print(sum(acc))

# %%
ASD_Bias.head(50)

# %%
# Things to annotate:
# 1. ASC 102 genes bias
# 2. Spark 159 Genes bias
# 3. Neuro Density Normed bias
# 4. Neuro/Glia Normed bias
# 5. Male Bias
# 6. Female Bias
# 7. 46 Circuit membership
# 8. 32 Circuit membership

ASD_ASC = pd.read_csv("../dat/Unionize_bias/ASD.ASC102.Z2.bias.csv",
                                      index_col="STR")
ASD_159 = pd.read_csv("../dat/Unionize_bias/ASD.159.pLI.z2.csv",
                                      index_col="STR")
ASD_Male = pd.read_csv("../dat/Unionize_bias/ASD.Male.ALL.bias.csv",
                                      index_col="STR")
ASD_Female = pd.read_csv("../dat/Unionize_bias/ASD.Female.ALL.bias.csv",
                                      index_col="STR")
ASD_Neuron_den_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuron.density.norm.bias.csv",
                                      index_col="STR")
ASD_Glia_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuro2glia.norm.bias.csv",
                                      index_col="STR")
ASD_CircuitsSet = pd.read_csv(
    "/home/jw3514/Work/ASD_Circuits/notebooks/ASD.SA.Circuits.Size46.csv",
    index_col="idx")
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")

# %%
ASD_32 = "Anterior_cingulate_area_dorsal_part,Anterior_olfactory_nucleus,Anterior_pretectal_nucleus,Anteromedial_visual_area,Bed_nuclei_of_the_stria_terminalis,Caudoputamen,Claustrum,Dentate_gyrus,Endopiriform_nucleus_dorsal_part,Field_CA2,Infralimbic_area,Lateral_amygdalar_nucleus,Lateral_posterior_nucleus_of_the_thalamus,Lateral_visual_area,Mediodorsal_nucleus_of_thalamus,Nucleus_accumbens,Nucleus_of_reuniens,Orbital_area_lateral_part,Orbital_area_ventrolateral_part,Parafascicular_nucleus,Parataenial_nucleus,Posterior_parietal_association_areas,Prelimbic_area,Primary_motor_area,Primary_somatosensory_area_lower_limb,Primary_somatosensory_area_trunk,Primary_visual_area,Retrosplenial_area_lateral_agranular_part,Rhomboid_nucleus,Secondary_motor_area,Subiculum_ventral_part,Submedial_nucleus_of_the_thalamus"
ASD_32 = ASD_32.split(",")

# %%
for STR, row in ASD_Bias.iterrows():
    ASD_Bias.loc[STR, "Bias.Male"] = ASD_Male.loc[STR, "EFFECT"]
    ASD_Bias.loc[STR, "Bias.Famale"] = ASD_Female.loc[STR, "EFFECT"]
    ASD_Bias.loc[STR, "Bias.ASC102"] = ASD_ASC.loc[STR, "EFFECT"]
    ASD_Bias.loc[STR, "Bias.Spark159"] = ASD_159.loc[STR, "EFFECT"]
    ASD_Bias.loc[STR, "Bias.Neuron_density_normalized"] = ASD_Neuron_den_norm_bias.loc[STR, "EFFECT"]
    ASD_Bias.loc[STR, "Bias.Neuron_glia_ratio_normalized"] = ASD_Glia_norm_bias.loc[STR, "EFFECT"]
    if STR in ASD_Circuits:
        ASD_Bias.loc[STR, "Circuits.46"] = 1
    else:
        ASD_Bias.loc[STR, "Circuits.46"] = 0
    if STR in ASD_32:
        ASD_Bias.loc[STR, "Circuits.32"] = 1
    else:
        ASD_Bias.loc[STR, "Circuits.32"] = 0

# %%
ASD_Bias.to_excel("../Manuscript/SupTabs/SupTab1.xlsx")

# %%
ASD_Bias
