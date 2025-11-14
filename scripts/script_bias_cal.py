import sys
import os
import yaml
import argparse
import pandas as pd
import gc
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests

# Load config to get project directory
def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
ProjDIR = config["ProjDIR"]
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *


def ReadNullBias(Bias_Null):
    # Bias_Null is a DF, index are cell type (cluster_id), columns are bias from a simulation of random genes 
    Bias_Null_DF = pd.read_parquet(Bias_Null)
    return Bias_Null_DF

def AddPvalue_vec(df, Null_Bias_values, method='vectorized', **kwargs):
#def AddPvalue_optimized(df, control_dfs, method='vectorized', **kwargs):
    """Optimized version of AddPvalue with multiple acceleration strategies."""
    # Convert to numpy arrays for faster processing
    observed_vals = df["EFFECT"].values
    cell_type_ids = df.index.tolist()
    
    # Strategy 1: Pure vectorization
    null_matrix = Null_Bias_values.loc[cell_type_ids].values.T
    #print(null_matrix.shape)
    z_scores, p_values, obs_adjs = GetPermutationP_vectorized(null_matrix, observed_vals)
        
    # Add results to dataframe
    df = df.copy()
    df["P-value"] = p_values
    df["Z-score"] = z_scores  
    df["EFFECT_adj"] = obs_adjs

    # Calculate FDR-corrected q-values

    _, q_values = multipletests(df["P-value"].values, alpha=0.05, method="fdr_i")[0:2]
    df["q-value"] = q_values
    df["-logP"] = -np.log10(df["P-value"])
    
    return df


def plot_boxplot_mouseCT(pc_scores_df, Anno, ALL_CTs, PC, geneset_name="", ylabel=None, ax=None):
    """
    Plot boxplot for mouse cell types with neuronal vs non-neuronal coloring.
    
    Parameters:
    -----------
    pc_scores_df : pandas.DataFrame
        DataFrame containing bias scores/p-values
    Anno : pandas.DataFrame
        Annotation dataframe with cell type information
    ALL_CTs : list
        List of all cell types to plot
    PC : str
        Column name to plot (e.g., "EFFECT", "-logP")
    ylabel : str, optional
        Y-axis label
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure
    """
    if ax is None:
        plt.figure(figsize=(12,8), dpi=300)
        ax = plt.gca()

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Define color palette
    palette = {'Neuron': 'red', 'Non-neuron': 'blue'}

    # Collect data and means for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["class_id_label"]==CT].index.values
        valid_CT_idx = [idx for idx in CT_idx if idx in pc_scores_df.index]
        dat = pc_scores_df.loc[valid_CT_idx, PC]
        box_data.append(dat)
        tick_labels.append(CT)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot with custom styling
    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                  markersize=4, alpha=0.5))

    # Color boxes based on neuronal vs non-neuronal using palette
    Non_NEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']
    for i, (patch, label) in enumerate(zip(bp['boxes'], tick_labels)):
        if label not in Non_NEUR:
            patch.set_facecolor(palette['Neuron'])
        else:
            patch.set_facecolor(palette['Non-neuron'])
        patch.set_alpha(0.7)

    # Set tick labels separately
    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set appropriate ylabel based on PC column
    if ylabel is None:
        if PC == "EFFECT":
            ylabel = "Effect Size (Bias Score)"
        elif PC == "-logP":
            ylabel = "-log10(P-value)"
        else:
            ylabel = PC
    
    ax.set_ylabel('{}'.format(ylabel), fontsize=14, fontweight='bold')
    title = f"Mouse Cell Type - {ylabel}"
    if geneset_name:
        title += f" ({geneset_name})"
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Add significance threshold line for -logP plots
    if PC == "-logP":
        sig_threshold = -np.log10(0.05)
        ax.axhline(y=sig_threshold, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'p = 0.05 (-log10 = {sig_threshold:.2f})')

    # Add grid and spines
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Create custom legend patches for cell types
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=palette['Neuron'], alpha=0.7, label='Neuron'),
                      plt.Rectangle((0,0),1,1, facecolor=palette['Non-neuron'], alpha=0.7, label='Non-neuron')]
    
    # Add significance line to legend if applicable
    if PC == "-logP":
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                        linewidth=2, alpha=0.7, label='p = 0.05'))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    return ax

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate bias for a gene set using provided weights and expression matrix.")
    parser.add_argument('--SpecMat', required=True, help='Path to expression matrix (parquet)')
    parser.add_argument('--gw', required=True, help='Path to gene weights file (CSV)')
    parser.add_argument('--mode', required=True, help='Analysis mode: human_ct_bias, mouse_ct_bias, or mouse_str_bias')
    parser.add_argument('--geneset', required=True, help='Name of the gene set')
    parser.add_argument('--Bias_Out', required=True, help='Output file for bias results (TSV)')
    parser.add_argument('--Bias_Null', required=True, help='Output file for bias results (TSV)')
    parser.add_argument('--plot', default=True, action='store_true', help='Plot bias analysis (default: True)')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting')
    return parser.parse_args()

def main():
    args = parse_args()
    SpecMat = load_expression_matrix_cached(args.SpecMat)
    GW = Fil2Dict(args.gw)
    
    # Extract geneset name for plots and file naming
    geneset_name = args.geneset
    
    # Compute bias based on mode
    if args.mode == 'human_ct_bias':
        Bias = HumanCT_AvgZ_Weighted(SpecMat, GW)
        Bias = AnnotateCTDat(Bias, Anno)
        # For human CT bias, just save the basic bias results (no p-values computed)
        Bias_add_P = Bias

    elif args.mode == 'mouse_ct_bias':
        Bias = MouseCT_AvgZ_Weighted(SpecMat, GW)

        # Load annotation from config
        mouse_ct_annotation_path = os.path.join(ProjDIR, config["data_files"]["mouse_ct_annotation"])
        try:
            ClusterAnn = pd.read_csv(mouse_ct_annotation_path, index_col="cluster_id_label")
            print(f"Loaded annotation from: {mouse_ct_annotation_path}")
        except FileNotFoundError:
            print(f"Error: Could not find MouseCT_Cluster_Anno.csv at {mouse_ct_annotation_path}")
            print("Please check if the file exists and update the config.yaml accordingly")
            raise

        # Optimize memory usage
        #ClusterAnn = optimize_dataframe_memory(ClusterAnn)
        Bias = add_class(Bias, ClusterAnn)
        Null_Bias_values = ReadNullBias(args.Bias_Null)
        Bias_add_P = AddPvalue_vec(Bias, Null_Bias_values)
        
        # Get all mouse classes for plotting
        ALL_Mouse_Class = sorted(ClusterAnn["class_id_label"].unique())
        print(f"Found {len(ALL_Mouse_Class)} unique mouse classes")
        
        # Create plots if enabled
        if args.plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=300)
            plot_boxplot_mouseCT(Bias_add_P, ClusterAnn, ALL_Mouse_Class, "EFFECT", 
                               geneset_name=geneset_name, ax=ax1)
            plot_boxplot_mouseCT(Bias_add_P, ClusterAnn, ALL_Mouse_Class, "-logP", 
                               geneset_name=geneset_name, ax=ax2)
            plt.tight_layout()
            
            # Create organized plot directory and filename
            bias_output_dir = os.path.dirname(args.Bias_Out)
            plot_dir = os.path.join(bias_output_dir, 'bias_plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_filename = os.path.join(plot_dir, f'{geneset_name}_MouseCT_bias_plots.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Plots saved to: {plot_filename}")

    elif args.mode == 'mouse_str_bias':
        Anno = STR2Region()
        Bias = MouseSTR_AvgZ_Weighted(SpecMat, GW)
        Bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in Bias.index.values]
        
        # For mouse STR bias, add p-values if null bias data is available
        if args.Bias_Null:
            Null_Bias_values = ReadNullBias(args.Bias_Null)
            Bias_add_P = AddPvalue_vec(Bias, Null_Bias_values)
        else:
            # If no null bias data, just save basic bias results
            Bias_add_P = Bias
            
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Supported modes: human_ct_bias, mouse_ct_bias, mouse_str_bias")

    Bias_add_P.to_csv(args.Bias_Out)
    print(f"Bias results saved to: {args.Bias_Out}")
    
    # Clean up memory
    del SpecMat, GW, Bias, Null_Bias_values, Bias_add_P
    if 'ClusterAnn' in locals():
        del ClusterAnn
    cleanup_memory()
    

if __name__ == "__main__":
    main()
