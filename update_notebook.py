
import json

notebook_path = 'notebook_rebuttal/QN_or_not.ipynb'

new_cell_source = [
    "# Create comparison dataframe\n",
    "Diff_DF = ASD_STR_Bias_Main[[\"EFFECT\", \"REGION\"]].copy()\n",
    "Diff_DF = Diff_DF.rename(columns={\"EFFECT\": \"EFFECT_Main\"})\n",
    "Diff_DF[\"EFFECT_WO_QN\"] = ASD_STR_Bias_WO_QN[\"EFFECT\"]\n",
    "\n",
    "# Calculate differences\n",
    "Diff_DF[\"Diff\"] = (Diff_DF[\"EFFECT_Main\"] - Diff_DF[\"EFFECT_WO_QN\"]).abs()\n",
    "Diff_DF[\"Diff_Signed\"] = Diff_DF[\"EFFECT_Main\"] - Diff_DF[\"EFFECT_WO_QN\"]\n",
    "\n",
    "# Sort by absolute difference\n",
    "Diff_DF = Diff_DF.sort_values(\"Diff\", ascending=False)\n",
    "\n",
    "print(\"Top 20 structures with largest difference (Main - WO_QN):\")\n",
    "display(Diff_DF.head(20)[[\"REGION\", \"EFFECT_Main\", \"EFFECT_WO_QN\", \"Diff\", \"Diff_Signed\"]])\n",
    "\n",
    "print(\"\\nMean absolute difference by Region:\")\n",
    "Region_Diff = Diff_DF.groupby(\"REGION\")[\"Diff\"].mean().sort_values(ascending=False)\n",
    "display(Region_Diff)"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "comparison_cell",
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Check if last cell is empty and replace it, otherwise append
if nb['cells'] and nb['cells'][-1]['source'] == []:
    nb['cells'][-1] = new_cell
else:
    nb['cells'].append(new_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

