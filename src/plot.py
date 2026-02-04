"""
Plotting functions for ASD Circuits analysis.

This module contains all plotting and visualization functions used throughout
the project for creating figures and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Helper for pretty p-value formatting everywhere
def pretty_pval_allstyle(pval):
    if np.isnan(pval):
        return "p = nan"
    elif pval < 1e-10:
        return "p < 1e-10"
    else:
        # choose scientific or decimal based on size
        if pval < 1e-3:
            return f"p = {pval:.2e}"
        elif pval < 0.01:
            return f"p = {pval:.3f}"
        elif pval < 0.1:
            return f"p = {pval:.2f}"
        else:
            return f"p = {pval:.2f}"

# Import ScoreCircuit_SI_Joint from ASD_Circuits if available
try:
    from ASD_Circuits import ScoreCircuit_SI_Joint
except ImportError:
    # Define a placeholder if not available
    def ScoreCircuit_SI_Joint(STRs, InfoMat):
        raise ImportError("ScoreCircuit_SI_Joint must be imported from ASD_Circuits")

# Constants for cell type classification
ABC_nonNEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']


# ============================================================================
# Structure Bias Correlation Plots
# ============================================================================

def plot_structure_bias_correlation(df_a, df_b, label_a='Dataset A', label_b='Dataset B', title=None):
    """
    Create comparison plot between two structure bias datasets

    Parameters:
    df_a: DataFrame with EFFECT column for first dataset
    df_b: DataFrame with EFFECT column for second dataset
    label_a: Label for x-axis (first dataset)
    label_b: Label for y-axis (second dataset)
    title: Custom title for the plot (ignored, no title drawn)

    Returns:
    correlation: Pearson correlation coefficient
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(5, 4), facecolor='none')

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Merge the datasets on structure names for comparison
    merged_data = pd.merge(df_a[['EFFECT']], df_b[['EFFECT']], 
                          left_index=True, right_index=True, suffixes=('_A', '_B'))

    # Create scatter plot
    ax.scatter(merged_data['EFFECT_A'], merged_data['EFFECT_B'], 
              alpha=1, s=20, c='#1f77b4', edgecolors='black', linewidth=0.5)

    # Add diagonal line for reference
    min_val = min(merged_data['EFFECT_A'].min(), merged_data['EFFECT_B'].min())
    max_val = max(merged_data['EFFECT_A'].max(), merged_data['EFFECT_B'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='y=x')

    # Calculate correlation and p-value
    correlation, pval = pearsonr(merged_data['EFFECT_A'], merged_data['EFFECT_B'])

    # Set labels (no title)
    ax.set_xlabel(f'{label_a}', fontsize=14)
    ax.set_ylabel(f'{label_b}', fontsize=14)

    # Format p-value display to "p < 1e-10" if p<1e-10 everywhere
    p_disp = pretty_pval_allstyle(pval)

    # Put correlation and p-value inside plot as annotation
    ax.annotate(f'r = {correlation:.3f}\n{p_disp}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=15,
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="gray", alpha=0.8))

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(fontsize=12)

    # Make axes equal for better comparison
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout() 
    plt.show()

    return correlation


def plot_correlation_profile_together(
    TopN_list, 
    DF_list1, 
    DF_list2, 
    reference_df,
    label1="Weighted", 
    label2="Uniform",
    title="Correlation with Reference vs TopN",
    ylabel="Spearman r with Reference"
):
    """
    Plot the correlation (Spearman r) of each DataFrame in the lists with the reference DataFrame over TopN_list.
    Ensures that the structure (index) between each df and reference_df are matched before calculation.
    """
    ref_effect = reference_df['EFFECT']
    ref_index = reference_df.index

    cors1 = []
    pvals1 = []
    for df in DF_list1:
        # Align indices before comparing
        df_matched = df.loc[df.index.intersection(ref_index)].copy()
        ref_matched = ref_effect.loc[df_matched.index]
        corrval, pval = spearmanr(df_matched['EFFECT'], ref_matched)
        cors1.append(corrval)
        pvals1.append(pval)
    cors2 = []
    pvals2 = []
    for df in DF_list2:
        df_matched = df.loc[df.index.intersection(ref_index)].copy()
        ref_matched = ref_effect.loc[df_matched.index]
        corrval, pval = spearmanr(df_matched['EFFECT'], ref_matched)
        cors2.append(corrval)
        pvals2.append(pval)
    
    plt.figure(figsize=(8, 5.5))
    sns.set(style="whitegrid", font_scale=1.25)
    sns.lineplot(x=TopN_list, y=cors1, marker='o', linewidth=2.5, label=label1, color='royalblue')
    sns.lineplot(x=TopN_list, y=cors2, marker='s', linewidth=2.5, label=label2, color='orange')
    plt.xlabel("Number of Top ASD Genes (Excluding Top 61)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_structure_bias_comparison(
    merged_data, 
    metric='Rank', 
    suffixes=('_1', '_2'),
    show_region_legend=True
):
    """
    Plot structure bias comparison between two datasets.

    Parameters:
    -----------
    merged_data : DataFrame
        Merged structure bias data from merge_bias_datasets function
    metric : str
        Which metric to plot ('Rank' or 'EFFECT')
    suffixes : tuple of str
        Suffixes used in the merged data column names
    show_region_legend : bool
        Whether to show the region color legend (default: True)
    """
    REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                    'Hippocampus','Amygdala','Striatum', 
                    "Thalamus", "Hypothalamus", "Midbrain", 
                    "Medulla", "Pallidum", "Pons", 
                    "Cerebellum"]
    REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                        "#2c9d39", "#742eb5", "#ed8921", 
                                        "#e82315", "#E6B0AA", "#f6b26b",  
                                        "#20124d", "#2ECC71", "#D2B4DE", 
                                        "#ffd966", ]))

    plt.style.use('seaborn-v0_8-whitegrid')

    region_color_map = REG_COR_Dic

    # Transparent background: set facecolor and axes alpha explicitly
    fig1, ax1 = plt.subplots(1, 1, dpi=480, figsize=(8, 6), facecolor='none')
    fig1.patch.set_alpha(0)
    ax1.patch.set_alpha(0)

    scatter_handles = []
    scatter_labels = []

    for region in REGIONS_seq:
        region_data = merged_data[merged_data['Region'] == region]
        if len(region_data) > 0:  # Only plot if there's data for this region
            sc = ax1.scatter(
                region_data[f'{metric}{suffixes[1]}'], 
                region_data[f'{metric}{suffixes[0]}'], 
                c=region_color_map[region], 
                s=60, 
                edgecolors='black', 
                linewidth=0.5, 
                alpha=0.8, 
                label=region
            )
            scatter_handles.append(sc)
            scatter_labels.append(region)

    # Add diagonal line for reference for Rank only (not for EFFECT)
    if metric == 'Rank':
        min_val = min(
            merged_data[f'{metric}{suffixes[0]}'].min(), 
            merged_data[f'{metric}{suffixes[1]}'].min()
        )
        max_val = max(
            merged_data[f'{metric}{suffixes[0]}'].max(), 
            merged_data[f'{metric}{suffixes[1]}'].max()
        )
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)

    # Add linear fit if metric is EFFECT
    if metric == 'EFFECT':
        x = merged_data[f'{metric}{suffixes[1]}']
        y = merged_data[f'{metric}{suffixes[0]}']
        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, fit_fn(x_line), 'r-', linewidth=2, label="Linear fit")

    # Calculate correlation and p-value
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(
        merged_data[f'{metric}{suffixes[0]}'], 
        merged_data[f'{metric}{suffixes[1]}']
    )

    # Set labels
    if metric == 'Rank':
        xlabel = f'{suffixes[1][1:]} Structure Bias Rank'
        ylabel = f'{suffixes[0][1:]} Structure Bias Rank'
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    else:  # EFFECT
        xlabel = f'{suffixes[1][1:]} Structure Bias'
        ylabel = f'{suffixes[0][1:]} Structure Bias'

    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Show region legend conditionally
    if show_region_legend:
        # Build legend from the artists and region names (guaranteed unique order)
        handles = []
        used_labels = set()
        for reg, h in zip(scatter_labels, scatter_handles):
            if reg not in used_labels:
                handles.append(h)
                used_labels.add(reg)
        ax1.legend(
            handles, list(used_labels), fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left'
        )
    else:
        # No legend for regions
        ax1.legend().set_visible(False)

    # Add text annotation for correlation and p-value - BIGGER FONT, plain background!
    ax1.text(
        0.02, 0.98,
        f'r = {correlation:.2f}\n{pretty_pval_allstyle(p_value)}',
        transform=ax1.transAxes,
        fontsize=18,
        ha='left',
        va='top'
    )

    plt.tight_layout()
    plt.show()

    return 
# ============================================================================
# Circuit Connectivity Plots
# ============================================================================

def compute_circuit_scores_for_profiles(
    profile_bias_dict,
    topNs,
    info_mats_dict,
    scoring_func=ScoreCircuit_SI_Joint,
):
    """
    Compute circuit scores for multiple profiles and connection types.

    Args:
        profile_bias_dict: dict of {profile_name: STR_Bias DataFrame}
        topNs: list of top N structure ranks to scan
        info_mats_dict: dict of {conn_type: info_mat pandas DataFrame}
        scoring_func: function (top_str_list, info_mat) => score

    Returns:
        results: dict of {profile_name: {conn_type: np.array of scores}}
    """
    results = {profile_name: {conn_type: [] for conn_type in info_mats_dict}
               for profile_name in profile_bias_dict}
    for profile_name, bias_df in profile_bias_dict.items():
        str_ranks = bias_df.sort_values("EFFECT", ascending=False).index.values
        for topN in topNs:
            top_strs = str_ranks[:topN]
            for conn_type, info_mat in info_mats_dict.items():
                score = scoring_func(top_strs, info_mat)
                results[profile_name][conn_type].append(score)
    # Convert lists to np.arrays for easier plotting
    for profile_name in results:
        for conn_type in results[profile_name]:
            results[profile_name][conn_type] = np.array(results[profile_name][conn_type])
    return results


def plot_circuit_connectivity_scores_multi(
    topNs,
    circuit_scores_results,
    cont_distance_dict,
    profile_plot_kwargs=None,
    show_siblings=True,
    profile_labels=None,
    xlim=(0, 121)
):
    """
    Plot circuit connectivity scores for multiple profiles and connection types.
    - circuit_scores_results: output of compute_circuit_scores_for_profiles
    - cont_distance_dict: {conn_type: np.array [n_iter, len(topNs)]}
    - profile_plot_kwargs: {profile_name: {kwargs for plt.plot}}
    - profile_labels: {profile_name: label string}
    """
    conn_types = list(cont_distance_dict.keys())
    n_conn = len(conn_types)
    fig, axes = plt.subplots(n_conn, 1, dpi=480, figsize=(7, 4*n_conn))

    if n_conn == 1:
        axes = [axes]

    BarLen = 34.1

    colors = ["blue", "red", "purple", "orange", "green", "black", "brown"]
    if profile_plot_kwargs is None:
        profile_plot_kwargs = {}
    if profile_labels is None:
        profile_labels = {}

    for i, conn_type in enumerate(conn_types):
        ax = axes[i]
        cont = (np.median if not conn_type.lower().startswith("long") else np.nanmean)(cont_distance_dict[conn_type], axis=0)
        # Plot scores for all profiles
        for idx, (profile_name, prof_scores) in enumerate(circuit_scores_results.items()):
            scores = prof_scores[conn_type]
            label = profile_labels.get(profile_name, profile_name)
            plot_args = profile_plot_kwargs.get(profile_name, {})
            if not plot_args:
                # Generate plot styles dynamically
                plot_args = dict(
                    color=colors[idx % len(colors)],
                    marker=["o", "s", "^", "d", "x", "v"][idx % 6],
                    markersize=5 if idx == 0 else 3,
                    lw=1,
                    ls='dashed' if idx == 0 else '-',
                    label=label
                )
            ax.plot(topNs, scores, **plot_args)

        # Plot Sibling controls
        if show_siblings:
            lower = np.nanpercentile(cont_distance_dict[conn_type], 50-BarLen, axis=0)
            upper = np.nanpercentile(cont_distance_dict[conn_type], 50+BarLen, axis=0)
            ax.errorbar(
                topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
                yerr=(cont - lower, np.abs(upper - cont)),
                ls="dashed", label="Siblings"
            )

        ax.set_xlabel("Structure Rank\n", fontsize=17)
        ax.set_ylabel("Circuit Connectivity Score", fontsize=15)
        ax.legend(fontsize=13)
        ax.set_xlim(*xlim)
        ax.grid(True)
        ax.set_title(f'Connection Type: {conn_type}', fontsize=15)
    plt.tight_layout()
    return fig


# ============================================================================
# Neurotransmitter System Plots
# ============================================================================

def plot_source_target_heatmap(results, top_n=15, save_plot=False, results_dir="./results"):
    """
    Create heatmap comparing source vs target across all systems
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    """
    comparison_data = {}
    all_structures = set()
    
    for system, system_data in results.items():
        for group_type in ['source', 'target', 'combined']:
            if group_type in system_data:
                key = f"{system}_{group_type}"
                comparison_data[key] = system_data[group_type]['EFFECT']
                all_structures.update(system_data[group_type].head(top_n).index)
    
    # Create comparison matrix
    comparison_df = pd.DataFrame(index=list(all_structures), columns=list(comparison_data.keys()))
    
    for key, effects in comparison_data.items():
        comparison_df[key] = effects
    
    # Fill NaN with 0 and sort by mean effect
    comparison_df = comparison_df.fillna(0)
    comparison_df = comparison_df.loc[comparison_df.mean(axis=1).sort_values(ascending=False).index]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison_df) * 0.3)))
    sns.heatmap(comparison_df.head(top_n), 
                annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                yticklabels=[s.replace('_', ' ') for s in comparison_df.head(top_n).index],
                ax=ax)
    ax.set_title('Neurotransmitter Source vs Target Comparison - Top Structures')
    ax.set_xlabel('System_Group')
    ax.set_ylabel('Brain Structure')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        import os
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_heatmap_top{top_n}.svg')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {plot_path}")
    
    plt.show()
    return fig


def plot_source_target_only(
    results, 
    top_n=15, 
    save_plot=False, 
    results_dir="./results",
    pvalue_label="q-value",
    dpi=300
):
    """
    Visualize source vs target neurotransmitter system bias results, 
    with significance stars (* for q < 0.05, ** for q < 0.01, *** for q < 0.001)
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    pvalue_label : str, default "q-value"
        Column/field name used to extract p-value for significance stars (e.g., "q-value" or "p-value")
    """
    
    def get_significance_star(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        return ''
    
    systems = list(results.keys())
    n_systems = len(systems)
    
    # Create subplots
    fig, axes = plt.subplots(n_systems, 3, figsize=(25, 4*n_systems), dpi=480)
    if n_systems == 1:
        axes = axes.reshape(1, -1)
    
    for i, system in enumerate(systems):
        system_data = results[system]
        
        # Plot source
        if 'source' in system_data:
            ax = axes[i, 0] if n_systems > 1 else axes[0]
            top_source = system_data['source'].head(top_n)
            bars = ax.barh(range(len(top_source)), top_source['EFFECT'])
            ax.set_yticks(range(len(top_source)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_source.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Source\n(synthesis, transporter, storage_release)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('orange')
            # Add significance stars
            if pvalue_label in top_source.columns:
                top_source_pval = top_source[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'source'.")
            for j, (eff_val, p_val) in enumerate(zip(top_source['EFFECT'], top_source_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset to right/left of bar depending on sign
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
                # Annotate p-value next to star, using "p < 1e-10" rule
                if not np.isnan(p_val):
                    p_text = pretty_pval_allstyle(p_val)
                    ax.text(
                        eff_val + 0.12 * (1 if eff_val >= 0 else -1),
                        j,
                        p_text,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=11
                    )
        
        # Plot target
        if 'target' in system_data:
            ax = axes[i, 1] if n_systems > 1 else axes[1]
            top_target = system_data['target'].head(top_n)
            bars = ax.barh(range(len(top_target)), top_target['EFFECT'])
            ax.set_yticks(range(len(top_target)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_target.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Target\n(receptor, degradation)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('blue')
            # Add significance stars
            if pvalue_label in top_target.columns:
                top_target_pval = top_target[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'target'.")
            for j, (eff_val, p_val) in enumerate(zip(top_target['EFFECT'], top_target_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
                if not np.isnan(p_val):
                    p_text = pretty_pval_allstyle(p_val)
                    ax.text(
                        eff_val + 0.12 * (1 if eff_val >= 0 else -1),
                        j,
                        p_text,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=11
                    )
        
        # Plot combined
        if 'combined' in system_data:
            ax = axes[i, 2] if n_systems > 1 else axes[2]
            top_combined = system_data['combined'].head(top_n)
            bars = ax.barh(range(len(top_combined)), top_combined['EFFECT'])
            ax.set_yticks(range(len(top_combined)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_combined.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Combined\n(source + target)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('gray')
            # Add significance stars
            if pvalue_label in top_combined.columns:
                top_combined_pval = top_combined[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'combined'.")
            for j, (eff_val, p_val) in enumerate(zip(top_combined['EFFECT'], top_combined_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
                if not np.isnan(p_val):
                    p_text = pretty_pval_allstyle(p_val)
                    ax.text(
                        eff_val + 0.12 * (1 if eff_val >= 0 else -1),
                        j,
                        p_text,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=11
                    )
    
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        import os
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_barplots_top{top_n}.svg')
        fig.savefig(plot_path, dpi=480, bbox_inches='tight')
        print(f"Bar plots saved to: {plot_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# Cell Type Correlation Plots
# ============================================================================

def plot_correlation_scatter_mouseCT(
    df1, 
    df2, 
    name1, 
    name2, 
    effect_col1="EFFECT", 
    effect_col2="EFFECT", 
    class_col="class_id_label", 
    ax=None, 
    dpi=80,
    corr_method="pearson"
):
    """
    Plot scatter and correlation between two mouse CT summary DataFrames.
    Adds a linear regression (fit from sklearn.linear_model.LinearRegression).
    corr_method: "pearson" (default) or "spearman"
    """
    # -- Imports for LR
    from sklearn.linear_model import LinearRegression

    # Find indices that appear in both dataframes
    common_indices = df1.index.intersection(df2.index)

    # Filter both dataframes to only include common indices and sort them
    df1_filtered = df1.loc[common_indices].sort_index()
    df2_filtered = df2.loc[common_indices].sort_index()

    # Ensure the indices are aligned
    assert df1_filtered.index.equals(df2_filtered.index), "Indices are not properly aligned"

    # Extract values
    values1 = df1_filtered[effect_col1].values
    values2 = df2_filtered[effect_col2].values

    # Get class labels (using df1, but they should be the same in both)
    try:
        class_labels = df1_filtered[class_col].values
    except:
        class_labels = df2_filtered[class_col].values

    # Create boolean mask for non-neuronal cell types
    non_neuronal_types = [
        '30 Astro-Epen',
        '31 OPC-Oligo',
        '32 OEC',
        '33 Vascular',
        '34 Immune'
    ]
    non_neur_mask = np.isin(class_labels, non_neuronal_types)

    # Choose correlation function
    if corr_method == "pearson":
        corrfunc = pearsonr
        rho_sign = "r"
    elif corr_method == "spearman":
        corrfunc = spearmanr
        rho_sign = "ρ"
    else:
        raise ValueError("corr_method must be 'pearson' or 'spearman'")

    # Calculate correlations and p-values
    if np.sum(~non_neur_mask) > 1:
        corr_neur, p_neur = corrfunc(values1[~non_neur_mask], values2[~non_neur_mask])
    else:
        corr_neur, p_neur = np.nan, np.nan
    if np.sum(non_neur_mask) > 1:
        corr_nonneur, p_nonneur = corrfunc(values1[non_neur_mask], values2[non_neur_mask])
    else:
        corr_nonneur, p_nonneur = np.nan, np.nan
    if len(values1) > 1:
        corr_all, p_all = corrfunc(values1, values2)
    else:
        corr_all, p_all = np.nan, np.nan

    def pretty_pval(pval):
        # Use style for all pvalue reporting
        return pretty_pval_allstyle(pval)

    print(f"All cells correlation: {corr_all:.2f} ({pretty_pval(p_all)})")
    print(f"Neuronal correlation: {corr_neur:.2f} ({pretty_pval(p_neur)})")
    print(f"Non-neuronal correlation: {corr_nonneur:.2f} ({pretty_pval(p_nonneur)})")

    if ax is None:
        plt.figure(figsize=(6.5,6), dpi=dpi)
        ax = plt.gca()

    # Scatter plot
    # x = values1, y = values2 for both groups
    ax.scatter(values1[~non_neur_mask], values2[~non_neur_mask],
               color="red", alpha=0.6, s=5, label="Neuronal")
    ax.scatter(values1[non_neur_mask], values2[non_neur_mask],
               color="blue", alpha=0.6, s=5, label="Non-neuronal")
    ax.set_xlabel(name1, fontsize=22)
    ax.set_ylabel(name2, fontsize=22)
    ax.legend(fontsize=12, loc="upper left")

    # Add linear regression line using all points (as in the notebook)
    if len(values1) > 1:
        # LinearRegression expects 2D X
        X = values1.reshape(-1, 1)
        y = values2
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        # Sort for plotting line
        sort_ix = np.argsort(values1)
        ax.plot(values1[sort_ix], y_pred[sort_ix], color='black', lw=2, linestyle='--', label='Linear fit')

    # Add correlation information
    ax.text(
        0.95, 0.05,
        f'All: {rho_sign} = {corr_all:.2f} ({pretty_pval(p_all)})\n'
        f'Neuronal: {rho_sign} = {corr_neur:.2f} ({pretty_pval(p_neur)})\n'
        f'Non-neuronal: {rho_sign} = {corr_nonneur:.2f} ({pretty_pval(p_nonneur)})',
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=12
    )

    return ax


def PlotBiasContrast_v2(MergeDF, name1, name2, dataset="Human", title=""):
    """
    Plot bias contrast between two datasets, separating neuronal and non-neuronal cell types.
    
    Parameters:
    -----------
    MergeDF : pandas.DataFrame
        Merged dataframe containing bias effects from both datasets
    name1 : str
        Name/label for first dataset
    name2 : str
        Name/label for second dataset
    dataset : str, default "Human"
        Either "HumanCT" or "MouseCT" to determine cell type classification
    title : str, default ""
        Plot title
    """
    if dataset == "HumanCT":
        # Note: Neurons variable would need to be imported or defined
        # For now, this will raise an error if Neurons is not defined
        try:
            Neurons  # Check if Neurons is defined
        except NameError:
            raise ValueError("For HumanCT dataset, 'Neurons' list must be defined or imported")
        
        NEUR = MergeDF[MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        NonNEUR = MergeDF[~MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7

        pval_neur_disp = pretty_pval_allstyle(p_Neur)
        pval_nonneur_disp = pretty_pval_allstyle(p_nonNeur)

        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f} ({pval_neur_disp})\nR_NonNEUR = {r_nonNeur:.2f} ({pval_nonneur_disp})",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return
    elif dataset == "MouseCT":
        NEUR = MergeDF[~MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        NonNEUR = MergeDF[MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7

        pval_neur_disp = pretty_pval_allstyle(p_Neur)
        pval_nonneur_disp = pretty_pval_allstyle(p_nonNeur)
        
        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f} ({pval_neur_disp})\nR_NonNEUR = {r_nonNeur:.2f} ({pval_nonneur_disp})",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return


# ============================================================================
# Boxplot Functions
# ============================================================================

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
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    return ax

