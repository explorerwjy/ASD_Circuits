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
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

# Helper for pretty p-value formatting everywhere
def format_pval(pval):
    """Format a p-value as a string (number only, no 'p =' prefix)."""
    if np.isnan(pval):
        return "nan"
    if pval < 1e-10:
        return "< 1e-10"
    if pval < 0.001:
        return f"{pval:.1e}"
    if pval < 0.01:
        return f"{pval:.4f}"
    if pval < 0.1:
        return f"{pval:.3f}"
    return f"{pval:.2f}"


def pretty_pval_allstyle(pval):
    """Format a p-value with 'p =' prefix for display in plots."""
    if np.isnan(pval):
        return "p = nan"
    if pval < 1e-10:
        return "p < 1e-10"
    return f"p = {format_pval(pval)}"

# Import ScoreCircuit_SI_Joint and GetPermutationP from ASD_Circuits if available
try:
    from ASD_Circuits import ScoreCircuit_SI_Joint, GetPermutationP
except ImportError:
    # Define placeholders if not available
    def ScoreCircuit_SI_Joint(STRs, InfoMat):
        raise ImportError("ScoreCircuit_SI_Joint must be imported from ASD_Circuits")
    def GetPermutationP(null, obs):
        raise ImportError("GetPermutationP must be imported from ASD_Circuits")

# Constants for cell type classification
ABC_nonNEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

# Shared color palettes for disorders and neurotransmitter systems
DISORDER_COLORS = {
    "ASD": "#1f77b4",
    "T2D": "#ff7f0e",
    "IBD": "#2ca02c",
    "HDL_C": "#d62728",
    "Parkinson": "#9467bd",
    "HBALC": "#8c564b",
    "Alzheimer": "#e377c2",
}

NT_COLORS = {
    "dopamine":      "#e45756",
    "serotonin":     "#4e79a7",
    "oxytocin":      "#76b7b2",
    "acetylcholine": "#f28e2b",
}

def get_system_color(name):
    """Look up color for a disorder or neurotransmitter system name."""
    if name in DISORDER_COLORS:
        return DISORDER_COLORS[name]
    if name.lower() in NT_COLORS:
        return NT_COLORS[name.lower()]
    # case-insensitive fallback for disorder names
    for k, v in DISORDER_COLORS.items():
        if name.lower() == k.lower():
            return v
    return "#555555"


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
    
    fig = plt.figure(figsize=(8, 5.5), facecolor='none')
    fig.patch.set_alpha(0)
    plt.gca().patch.set_alpha(0)
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
    # Increase fontsize of axis labels by 50%: from 14 to 21, and ticks from default to 18
    if metric == 'Rank':
        xlabel = f'{suffixes[1][1:]} Structure Bias Rank'
        ylabel = f'{suffixes[0][1:]} Structure Bias Rank'
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    else:  # EFFECT
        xlabel = f'{suffixes[1][1:]} Structure Bias'
        ylabel = f'{suffixes[0][1:]} Structure Bias'

    ax1.set_xlabel(xlabel, fontsize=21, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=21, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=18)

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
    fig, axes = plt.subplots(n_conn, 1, dpi=480, figsize=(7, 4*n_conn), facecolor='none')
    fig.patch.set_alpha(0)

    if n_conn == 1:
        axes = [axes]
    for _ax in axes:
        _ax.patch.set_alpha(0)

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
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison_df) * 0.3)), facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
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
    fig, axes = plt.subplots(n_systems, 3, figsize=(25, 4*n_systems), dpi=480, facecolor='none')
    fig.patch.set_alpha(0)
    if n_systems == 1:
        axes = axes.reshape(1, -1)
    for _ax in axes.flat:
        _ax.patch.set_alpha(0)
    
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
        fig = plt.figure(figsize=(6.5,6), dpi=dpi, facecolor='none')
        fig.patch.set_alpha(0)
        ax = plt.gca()
        ax.patch.set_alpha(0)

    # Scatter plot
    # x = values1, y = values2 for both groups
    ax.scatter(values1[~non_neur_mask], values2[~non_neur_mask],
               color="red", alpha=0.6, s=5, label="Neuronal")
    ax.scatter(values1[non_neur_mask], values2[non_neur_mask],
               color="blue", alpha=0.6, s=5, label="Non-neuronal")
    ax.set_xlabel(name1, fontsize=22)
    ax.set_ylabel(name2, fontsize=22)
    ax.legend(fontsize=18, loc="upper left")

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
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5), facecolor='none')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

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
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5), facecolor='none')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

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
        fig = plt.figure(figsize=(12,8), dpi=300, facecolor='none')
        fig.patch.set_alpha(0)
        ax = plt.gca()
        ax.patch.set_alpha(0)

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


def plot_top_residual_structures_with_CI(merged_data, residual_ci_df=None, top_n=30, top_threshold=40,
                                         name1="ASD", name2="DD", figsize=(10, 8)):
    """
    Plot brain structures with largest residuals from regression analysis, with optional 95% CI error bars.

    Parameters:
    -----------
    merged_data : DataFrame
        Merged dataset with residual and region information
    residual_ci_df : DataFrame or None
        DataFrame with CI information (columns: ci_lower, ci_upper, median, mean).
        If None, plots without error bars.
    top_n : int
        Number of top structures to display
    top_threshold : int
        Filter to structures in top N of at least one dataset
    name1, name2 : str
        Names of the two datasets being compared
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    top_diff : DataFrame
        Top structures with largest residuals
    """
    # Filter to only structures that appear in top threshold of at least one dataset
    top_structures = merged_data[(merged_data[f"Rank_{name1}"] <= top_threshold) |
                                (merged_data[f"Rank_{name2}"] <= top_threshold)]

    print(f"Total structures in top {top_threshold} of at least one dataset: {len(top_structures)}")

    # Sort by absolute difference for top structures only
    top_structures = top_structures.copy()
    top_structures["ABS_DIFF"] = abs(merged_data[f"residual"])
    top_structures = top_structures.sort_values('ABS_DIFF', ascending=True)

    # Take the top N structures with largest differences from those in top threshold
    top_n = min(top_n, len(top_structures))
    top_diff = top_structures.tail(top_n)

    print(f"Showing top {len(top_diff)} structures with largest differences (from top {top_threshold} filter)")

    # Merge with CI data if provided
    if residual_ci_df is not None:
        top_diff = top_diff.merge(residual_ci_df[['ci_lower', 'ci_upper', 'median']],
                                 left_index=True, right_index=True, how='left')
        top_diff['residual_plot'] = top_diff['residual']
        top_diff['ci_lower_plot'] = top_diff['ci_lower'].fillna(top_diff['residual'])
        top_diff['ci_upper_plot'] = top_diff['ci_upper'].fillna(top_diff['residual'])
    else:
        top_diff['residual_plot'] = top_diff['residual']

    # Define regions and colors
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

    # Create publication-quality plot of residuals for top_diff structures
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Sort top_diff by residuals for better visualization
    top_diff_sorted = top_diff.sort_values('ABS_DIFF', ascending=True)

    # Create colors based on region
    colors = [REG_COR_Dic.get(region, '#808080') for region in top_diff_sorted['Region']]

    # Calculate positions and values
    y_pos = range(len(top_diff_sorted))
    x_vals = top_diff_sorted['residual_plot'].values

    # Create horizontal bar plot
    bars = ax.barh(y_pos, x_vals,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add error bars if CI data is available
    if residual_ci_df is not None:
        xerr_lower = x_vals - top_diff_sorted['ci_lower_plot'].values
        xerr_upper = top_diff_sorted['ci_upper_plot'].values - x_vals
        ax.errorbar(x_vals, y_pos,
                    xerr=[xerr_lower, xerr_upper],
                    fmt='none', ecolor='black', elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.7)

    # Customize the plot with publication-quality styling
    # yticklabels are the main tick labels for this horizontal bar plot,
    # so these are the ones whose fontsize you want to increase
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ') for name in top_diff_sorted.index],
                       fontsize=15, fontweight='normal')
    #xlabel_suffix = " with 95% CI" if residual_ci_df is not None else ""
    xlabel_suffix = ""
    ax.set_xlabel(f'Bias Residuals ({name1} vs {name2}){xlabel_suffix}', fontsize=20, fontweight='bold')

    # Set X-axis tick font size (THIS IS YOUR REQUESTED CHANGE)
    ax.tick_params(axis='x', labelsize=16)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Add subtle grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add vertical line at x=0 with better styling
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)

    # Create legend for regions with better styling
    unique_regions = sorted(list(set(top_diff_sorted['Region'])))
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=REG_COR_Dic.get(region, '#808080'),
                                    alpha=0.8, edgecolor='black', linewidth=0.5)
                       for region in unique_regions if region in REG_COR_Dic]
    legend_labels = [region.replace('_', ' ') for region in unique_regions if region in REG_COR_Dic]

    if legend_elements:
        ax.legend(
            legend_elements, legend_labels,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )

    # Adjust layout: left margin for structure names, right margin for legend
    fig.subplots_adjust(left=0.3, right=0.72)
    plt.show()

    return top_diff


# --- Helpers for cluster_residual_boxplot ---

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = np.sum(np.isfinite(pvals))
    out = pvals.copy()
    if n == 0:
        return out
    idx = np.where(np.isfinite(pvals))[0]
    p = pvals[idx]
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_idx = np.empty_like(adj)
    out_idx[order] = adj
    out[idx] = out_idx
    return out


def p_to_star(p):
    """Convert p-value to significance stars."""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"


def wrap_label(s, max_len=16):
    """Wrap a long label string for tick labels."""
    if len(s) <= max_len:
        return s
    for sep in [" ", "_"]:
        if sep in s:
            parts = s.split(sep)
            line1, line2 = [], []
            cur = 0
            for part in parts:
                add = len(part) + (1 if line1 else 0)
                if cur + add <= max_len:
                    line1.append(part); cur += add
                else:
                    line2.append(part)
            if line2:
                return " ".join(line1) + "\n" + " ".join(line2)
    return s


def cluster_residual_boxplot(
    results_df,
    cluster_dict,
    metric="residual",
    palette=None,
    figsize=(12, 8),
    pairwise_tests=None,
    p_adjust="fdr_bh",
    p_style="stars",
    show_ns=False,
    wrap_xticks=True,
    wrap_len=16,
    point_size=2.2,
    point_alpha=0.16,
    point_color="0.2",
    rasterize_points=True,
    box_width=0.6,
    fontsize=12,
    title=None,
    show=True
):
    """
    Boxplot of residuals (or any metric) grouped by cell-type clusters,
    with optional pairwise Mann-Whitney U tests and FDR correction.

    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with metric column, indexed by cell type IDs
    cluster_dict : dict
        {cluster_name: [cell_type_ids]}
    metric : str
        Column name in results_df to plot
    palette : list or dict or None
        Colors for each cluster
    pairwise_tests : list of tuples
        [(groupA, groupB), ...] for Mann-Whitney U tests
    p_adjust : str or None
        "fdr_bh" for BH correction, None for raw p-values
    p_style : str
        "stars" or "exact" for annotation style
    show_ns : bool
        Whether to show non-significant comparisons

    Returns:
    --------
    plot_df : DataFrame
        Long-form DataFrame used for plotting
    """
    if metric not in results_df.columns:
        raise ValueError(f"metric='{metric}' not in results_df.columns")
    if pairwise_tests is None:
        pairwise_tests = []

    cluster_labels = list(cluster_dict.keys())

    # palette
    if palette is None:
        palette = sns.color_palette("tab10", n_colors=len(cluster_labels))
    elif isinstance(palette, dict):
        palette = [palette[k] for k in cluster_labels]
    else:
        if len(palette) < len(cluster_labels):
            raise ValueError(f"palette has {len(palette)} colors but needs {len(cluster_labels)}.")
        palette = palette[:len(cluster_labels)]

    # build plot_df
    vals_list, n_points = [], []
    for k in cluster_labels:
        v = results_df.loc[cluster_dict[k], metric].dropna().values
        vals_list.append(v)
        n_points.append(len(v))

    plot_df = pd.DataFrame({
        "Cluster": np.repeat(cluster_labels, n_points),
        metric: np.concatenate(vals_list) if len(vals_list) else np.array([])
    })

    # plot
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=240, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    sns.boxplot(
        x="Cluster", y=metric, data=plot_df,
        palette=palette, width=box_width,
        showfliers=False, linewidth=1.0,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black",
                   "markeredgecolor": "black", "markersize": 5},
        ax=ax
    )
    for patch in ax.artists:
        patch.set_alpha(0.88)

    sns.stripplot(
        x="Cluster", y=metric, data=plot_df,
        color=point_color, alpha=point_alpha,
        jitter=0.22, size=point_size,
        ax=ax
    )
    if rasterize_points:
        for coll in ax.collections:
            coll.set_rasterized(True)

    ax.axhline(0, color="black", linewidth=2.0, alpha=0.85, linestyle="--", zorder=2)
    ax.grid(axis="y", color="0.86", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.set_ylabel("Bias Residual", fontsize=fontsize * 1.8)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=fontsize)

    xticklabels = cluster_labels
    if wrap_xticks:
        xticklabels = [wrap_label(s, max_len=wrap_len) for s in xticklabels]
    ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=fontsize*1.3)

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # prepare comparisons
    def get_vals(group):
        group = [group] if isinstance(group, str) else list(group)
        arrs = []
        for k in group:
            if k not in cluster_dict:
                continue
            arrs.append(results_df.loc[cluster_dict[k], metric].dropna().values)
        return np.concatenate(arrs) if len(arrs) else np.array([])

    tests = []
    for gA, gB in pairwise_tests:
        A = get_vals(gA)
        B = get_vals(gB)
        if len(A) == 0 or len(B) == 0:
            continue
        gA_list = [gA] if isinstance(gA, str) else list(gA)
        gB_list = [gB] if isinstance(gB, str) else list(gB)
        x1 = float(np.mean([cluster_labels.index(k) for k in gA_list]))
        x2 = float(np.mean([cluster_labels.index(k) for k in gB_list]))
        _, p = mannwhitneyu(A, B, alternative="two-sided")
        local_top = max(np.max(A), np.max(B))
        tests.append({"x1": x1, "x2": x2, "p": p, "local_top": local_top})

    if len(tests) == 0:
        plt.subplots_adjust(bottom=0.28, top=0.92)
        if show:
            plt.show()
        return plot_df

    # adjust p
    raw_p = np.array([t["p"] for t in tests], dtype=float)
    adj_p = bh_fdr(raw_p) if p_adjust == "fdr_bh" else raw_p
    for t, p_adj in zip(tests, adj_p):
        t["p_adj"] = p_adj

    # annotate brackets
    y_min = float(np.nanmin(plot_df[metric].values))
    y_max = float(np.nanmax(plot_df[metric].values))
    y_range = (y_max - y_min) if y_max != y_min else 1.0
    h = 0.020 * y_range
    clearance = 0.03 * y_range
    y_step = 0.10 * y_range

    tests_sorted = sorted(tests, key=lambda t: (t["local_top"], abs(t["x2"] - t["x1"])))

    placed = []
    for t in tests_sorted:
        p_use = t["p_adj"] if p_adjust else t["p"]
        label = p_to_star(p_use) if p_style == "stars" else f"$p$={p_use:.2e}"
        if (label == "ns") and (not show_ns):
            continue

        x1, x2 = t["x1"], t["x2"]
        xlo, xhi = min(x1, x2), max(x1, x2)

        y = t["local_top"] + clearance
        while True:
            yhi = y + h
            overlap = False
            for pxlo, pxhi, pylo, pyhi in placed:
                if not (xhi < pxlo - 0.3 or xlo > pxhi + 0.3):
                    if not (yhi < pylo - 0.01 or y > pyhi + 0.01):
                        overlap = True
                        break
            if not overlap:
                break
            y += y_step

        placed.append((xlo, xhi, y, y + h))
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="k", alpha=0.9)
        ax.text((x1 + x2) / 2, y + h - 2*h, label, ha="center", va="bottom", fontsize=fontsize*2.0)

    plt.subplots_adjust(bottom=0.28, top=0.92)
    if show:
        plt.show()
    return plot_df


def plot_null_distribution_analysis(structure_name, null_dfs, observed_df, title_prefix="", plot=True):
    """
    Plot null distribution analysis for a given brain structure.

    Parameters:
    -----------
    structure_name : str
        Name of the structure to analyze
    null_dfs : list of DataFrame
        List of dataframes containing null distribution data
    observed_df : DataFrame
        Dataframe containing observed data
    title_prefix : str
        Optional prefix for the plot title
    plot : bool
        Whether to show the plot

    Returns:
    --------
    p_value : float
        One-tailed p-value (observed >= null)
    observed_effect : float
        Observed EFFECT value
    null_effects : ndarray
        Array of null EFFECT values
    """
    # Extract EFFECT values from all null datasets
    null_effects = []
    for df in null_dfs:
        null_effects.append(df.loc[structure_name, "EFFECT"])

    # Get observed value
    observed_effect = observed_df.loc[structure_name, "EFFECT"]

    # Calculate p-value (one-tailed test: observed > null)
    null_effects = np.array(null_effects)
    p_value = (np.sum(null_effects >= observed_effect) + 1) / (len(null_effects) + 1)

    # Plot histogram
    if plot:
        fig = plt.figure(figsize=(10, 6), facecolor='none')
        fig.patch.set_alpha(0)
        plt.gca().patch.set_alpha(0)
        plt.hist(null_effects, bins=50, alpha=0.7, color='lightblue', edgecolor='black',
                 label='Null distribution (Constrained Genes)')
        plt.axvline(observed_effect, color='red', linestyle='--', linewidth=2,
                    label=f'Observed (Spark ASD): {observed_effect:.4f}')
        plt.xlabel('EFFECT')
        plt.ylabel('Frequency')
        plt.title(f'{title_prefix}{structure_name} EFFECT: Null Distribution vs Observed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.text(0.05, 0.95, f'P-value: {p_value:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.show()

        print(f"Observed Spark ASD effect: {observed_effect:.4f}")
        print(f"Null mean: {np.mean(null_effects):.4f}")
        print(f"Null std: {np.std(null_effects):.4f}")
        print(f"P-value: {p_value:.4f}")

    return p_value, observed_effect, null_effects


# ============================================================================
# Neurotransmitter / Disorder CCS Plots
# ============================================================================

def plot_combined_bias_only(
    results,
    top_n=15,
    save_plot=False,
    results_dir="./results",
    pvalue_label="q-value",
    dpi=300
):
    """
    Visualize only the COMBINED neurotransmitter system bias results,
    in a 1xN (one row, N columns) figure: one barplot per system.
    Shows significance stars (* for q < 0.05, ** for q < 0.01, *** for q < 0.001).
    Each structure bar colored by region, using provided color schema.
    """
    import matplotlib.patches as mpatches
    import os

    REGIONS_seq = [
        'Isocortex', 'Olfactory_areas', 'Cortical_subplate',
        'Hippocampus', 'Amygdala', 'Striatum',
        "Thalamus", "Hypothalamus", "Midbrain",
        "Medulla", "Pallidum", "Pons",
        "Cerebellum"
    ]
    REG_COR_Dic = dict(zip(REGIONS_seq, [
        "#268ad5", "#D5DBDB", "#7ac3fa",
        "#2c9d39", "#742eb5", "#ed8921",
        "#e82315", "#E6B0AA", "#f6b26b",
        "#20124d", "#2ECC71", "#D2B4DE",
        "#ffd966",
    ]))

    def get_significance_star(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        return ''

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.5)

    systems = list(results.keys())
    n_systems = len(systems)
    n_panels = min(n_systems, 3)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7), dpi=dpi, constrained_layout=True, facecolor='none')
    fig.patch.set_alpha(0)
    if n_panels == 1:
        axes = [axes]
    for _ax in axes:
        _ax.patch.set_alpha(0)

    # Gather region-color presence across all subplots for the legend
    found_region_colors = dict()
    for i in range(n_panels):
        system = systems[i]
        system_data = results[system]
        if 'combined' not in system_data:
            continue
        top_combined = system_data['combined'].head(top_n)
        if "Region" in top_combined.columns:
            for region in top_combined["Region"]:
                region = str(region)
                col = REG_COR_Dic.get(region, "#888888")
                found_region_colors[region] = col
        else:
            found_region_colors["Unknown"] = "#888888"

    # For legend: order by REGIONS_seq, then add others
    used_regions = [r for r in REGIONS_seq if r in found_region_colors]
    others = sorted([r for r in found_region_colors if r not in REGIONS_seq])
    legend_entries = used_regions + others
    legend_handles = [
        mpatches.Patch(color=found_region_colors[r], label=r.replace('_', ' ')) for r in legend_entries
    ]

    # Plotting each panel
    for i in range(n_panels):
        system = systems[i]
        system_data = results[system]
        ax = axes[i]

        if 'combined' not in system_data:
            ax.axis('off')
            continue

        top_combined = system_data['combined'].head(top_n)

        region_col = []
        if "Region" in top_combined.columns:
            for region in top_combined["Region"]:
                region = str(region)
                col = REG_COR_Dic.get(region, "#888888")
                region_col.append(col)
        else:
            region_col = ["#888888"] * len(top_combined)

        eff_vals = top_combined['EFFECT'].values
        ax.barh(
            y=np.arange(len(top_combined)),
            width=eff_vals,
            color=region_col,
            edgecolor='black',
            alpha=0.94,
            zorder=2
        )

        ax.set_yticks(np.arange(len(top_combined)))
        ax.set_yticklabels([s.replace('_', ' ') for s in top_combined.index], fontsize=35, fontweight='bold')
        ax.set_xlabel('Bias Effect', fontsize=18, fontweight='bold', labelpad=8)
        ax.set_title(f'{system.capitalize()}', fontsize=21, fontweight='bold', pad=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=15)
        ax.invert_yaxis()

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['bottom'].set_linewidth(1.6)

        ax.axvline(0, ls='--', color='grey', lw=1.1, zorder=1)

        if pvalue_label in top_combined.columns:
            top_combined_pval = top_combined[pvalue_label]
        else:
            raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'combined'.")
        for j, (eff_val, p_val) in enumerate(zip(top_combined['EFFECT'], top_combined_pval)):
            star = get_significance_star(p_val)
            if star:
                x_offset = 0.08 * (np.nanmax(np.abs(eff_vals)) or 1)
                x = eff_val + x_offset if eff_val >= 0 else eff_val - x_offset
                ha = 'left' if eff_val >= 0 else 'right'
                ax.text(
                    x, j,
                    star,
                    va='center',
                    ha=ha,
                    color='firebrick',
                    fontsize=23,
                    fontweight='bold',
                    zorder=5,
                )
        bar_absmax = np.nanmax(np.abs(eff_vals))
        ax.set_xlim(0, bar_absmax * 1.18)

    # Place region legend on last panel
    for i, ax in enumerate(axes):
        if i == len(axes) - 1:
            ax.legend(
                handles=legend_handles,
                loc='upper left',
                bbox_to_anchor=(0.7, 0.6),
                borderaxespad=0.6,
                fontsize=16,
                ncol=1,
                title="Region",
                title_fontsize=16,
                frameon=True
            )

    fig.subplots_adjust(wspace=0.23, left=0.13, right=0.98, bottom=0.07, top=0.93)

    if save_plot:
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_COMBINED_barplots_top{top_n}.svg')
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Combined bar plots saved to: {plot_path}")
    plt.show()
    return fig


def plot_CCS_pvalues_at_N(n, neuro_system_scores, topNs, Cont_Distance, colors=None):
    """
    Plot -log10(p-values) of circuit connectivity scores at given N (topN).

    Args:
        n: int
            Rank at which to evaluate CCS p-values (e.g., 40, 50, etc).
        neuro_system_scores: dict
            Dictionary with system names as keys and CCS score arrays as values.
        topNs: array-like
            Array of topN values corresponding to score/Cont_Distance columns.
        Cont_Distance: numpy array
            Null distribution array, shape (n_permutations, len(topNs)).
        colors: list or None
            Color palette for bars. If None, uses default tab10 palette.
    Returns:
        fig, ax: matplotlib Figure and Axes objects
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    topNs = np.asarray(topNs)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    pvalues = {}
    for system_category, scores in neuro_system_scores.items():
        obs = scores[np.where(topNs == n)][0]
        Null = Cont_Distance[:, np.where(topNs == n)].flatten()
        z, p, xx = GetPermutationP(Null, obs)
        pvalues[system_category] = p

    labels = list(neuro_system_scores.keys())
    values = [-np.log10(pvalues[label]) for label in labels]

    colors_bar = colors[:len(labels)]
    if len(colors_bar) < len(labels):
        times = (len(labels) // len(colors)) + 1
        colors_bar = (colors * times)[:len(labels)]

    bars = ax.bar(range(len(labels)), values, color=colors_bar)
    for bar in bars:
        bar.set_linewidth(1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f'CCS Score P-values at N_Str={n} (Neurotransmitter systems)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)

    print(f"CCS score P-values at N_Str={n} (Neurotransmitter systems):")
    for label in labels:
        print(f"{label}: {format_pval(pvalues[label])}")

    plt.tight_layout()
    return fig, ax


def plot_CCS_histogram_with_NT_bars(n, neuro_system_scores, Cont_Distance, topNs):
    """
    Plot histogram of siblings CCS distribution with vertical bars for each system.
    System color is searched in both neurotransmitter and disorder color dicts.
    Labels are placed with arrows to avoid overlap.

    Args:
        n: int
            Rank at which to evaluate CCS (e.g., 20, 40)
        neuro_system_scores: dict
            Dictionary with system names as keys and CCS arrays as values
        Cont_Distance: numpy array
            Array of CCS scores for siblings (null distribution), shape (n_permutations, n_topNs)
        topNs: numpy array
            Array of topN values corresponding to Cont_Distance columns
    Returns:
        fig, ax: matplotlib Figure and Axes objects
    """
    import matplotlib as mpl

    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.edgecolor": "k",
        "axes.linewidth": 1.4,
    })

    n_idx = np.where(topNs == n)[0]
    if len(n_idx) == 0:
        raise ValueError(f"N={n} not found in topNs")
    n_idx = n_idx[0]
    siblings_CCS = Cont_Distance[:, n_idx].flatten()

    # Gather observed values and p-values
    nt_systems = list(neuro_system_scores.keys())
    nt_values = {}
    nt_pvalues = {}
    for system in nt_systems:
        obs = neuro_system_scores[system][n_idx]
        nt_values[system] = obs
        _, p_value, _ = GetPermutationP(siblings_CCS, obs)
        nt_pvalues[system] = p_value

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5), facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    n_bins = 25
    hist_n, bins, patches = ax.hist(
        siblings_CCS, bins=n_bins, histtype="barstacked", align="mid",
        facecolor="gray", alpha=0.75, label="Siblings", edgecolor="black", linewidth=0.7, zorder=2
    )

    systems_with_values = sorted(
        [(system, nt_values[system], nt_pvalues[system]) for system in nt_systems],
        key=lambda x: x[1]
    )

    # Plot vertical lines
    for i, (system, obs_val, p_val) in enumerate(systems_with_values):
        col = get_system_color(system)
        if system == "ASD":
            lw, alpha, ls, zorder = 3.5, 1.0, '-', 4
        else:
            lw, alpha, ls, zorder = 2.2, 0.85, '--', 3
        ax.axvline(
            obs_val, ymin=0, ymax=1, linewidth=lw, color=col,
            linestyle=ls, label=system, alpha=alpha, zorder=zorder
        )

    # Annotate labels near vertical lines, staggered vertically to avoid overlap
    hist_peak = hist_n.max()
    x_range = siblings_CCS.max() - siblings_CCS.min()
    x_offset = 0.04 * x_range  # small offset to the right of the line

    # Stagger y-positions in upper part of histogram
    y_positions = np.linspace(hist_peak * 0.95, hist_peak * 0.55, len(systems_with_values))

    for (system, obs_val, p_val), y_pos in zip(systems_with_values, y_positions):
        col = get_system_color(system)
        ax.annotate(
            f"{system}\np={format_pval(p_val)}",
            xy=(obs_val, y_pos),
            xytext=(obs_val + x_offset, y_pos),
            fontsize=11, fontweight="bold",
            color=col, ha="left", va="center",
            bbox=dict(boxstyle='round,pad=0.25', facecolor="white", alpha=0.85, edgecolor=col, linewidth=1.5),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.2)
        )

    ax.set_xlabel("Circuit Connectivity Score (CCS)", fontsize=18, weight="bold")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.grid(axis='y', linestyle="--", alpha=0.3)

    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    od = OrderedDict()
    for h, l in zip(handles, labels):
        od[l] = h
    ax.legend(
        od.values(), od.keys(),
        loc="upper right", bbox_to_anchor=(1.15, 1.0),
        borderaxespad=0, frameon=False, ncol=1
    )

    plt.tight_layout()
    return fig, ax


# ============================================================================
# Average Positive Mutation Bias Histogram
# ============================================================================

def plot_avg_positive_bias_histogram(
    null_biases,
    observed_bias,
    control_biases=None,
    observed_label="ASD Probands",
    dpi=300,
    figsize=(7, 5),
):
    """Plot histogram of null distribution with observed and control vertical lines.

    Parameters
    ----------
    null_biases : array-like
        Sibling null distribution of mean positive biases.
    observed_bias : float
        Observed ASD mean positive bias value.
    control_biases : dict or None
        Dictionary mapping disorder name to its mean positive bias value,
        e.g. {"T2D": 0.12, "IBD": 0.08}.
    observed_label : str
        Label for the observed (ASD) line.
    dpi : int
        Figure resolution.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects.
    """
    import matplotlib as mpl

    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.edgecolor": "k",
        "axes.linewidth": 1.4,
    })

    if control_biases is None:
        control_biases = {}

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Histogram of null distribution
    n_hist, bins, patches = ax.hist(
        null_biases, bins=20, histtype="barstacked", align="mid",
        facecolor="gray", alpha=0.75, label="Siblings",
        edgecolor="black", linewidth=0.7, zorder=2,
    )

    # ASD observed line
    asd_color = get_system_color("ASD")
    ax.axvline(
        observed_bias, linewidth=3.5, color=asd_color,
        linestyle='-', label=observed_label, zorder=4,
    )

    # Control lines
    for name, val in control_biases.items():
        col = get_system_color(name)
        ax.axvline(
            val, linewidth=2.2, color=col,
            linestyle='--', label=name, alpha=0.85, zorder=3,
        )

    # Compute p-values for all lines
    all_systems = [("ASD", observed_bias)]
    for name, val in control_biases.items():
        all_systems.append((name, val))

    systems_with_pvals = []
    for name, val in all_systems:
        _, p_val, _ = GetPermutationP(null_biases, val)
        systems_with_pvals.append((name, val, p_val))

    # Stagger annotations vertically to avoid overlap
    y_max = n_hist.max()
    n_labels = len(systems_with_pvals)
    y_positions = np.linspace(y_max * 0.95, y_max * 0.35, max(n_labels, 2))[:n_labels]

    x_lo, x_hi = ax.get_xlim()
    x_range = x_hi - x_lo
    x_offset = x_range * 0.02
    x_mid = (x_lo + x_hi) / 2.0

    for (name, val, p_val), y_pos in zip(systems_with_pvals, y_positions):
        col = get_system_color(name)
        # Place label to the left if the line is in the right half
        if val > x_mid:
            txt_x = val - x_offset
            ha = "right"
        else:
            txt_x = val + x_offset
            ha = "left"
        ax.annotate(
            f"{name}\np={format_pval(p_val)}",
            xy=(val, y_pos),
            xytext=(txt_x, y_pos),
            fontsize=11, fontweight="bold",
            color=col, ha=ha, va="center",
            bbox=dict(
                boxstyle='round,pad=0.25', facecolor="white",
                alpha=0.85, edgecolor=col, linewidth=1.5,
            ),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.2),
        )

    # Axes styling
    ax.set_xlabel("Average Positive Mutation Bias", fontsize=18, weight="bold")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.grid(axis='y', linestyle="--", alpha=0.3)

    # Deduplicated legend
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    od = OrderedDict()
    for h, l in zip(handles, labels):
        od[l] = h
    ax.legend(
        od.values(), od.keys(),
        loc="upper left", bbox_to_anchor=(0.45, 1.0),
        borderaxespad=0, frameon=False, ncol=1,
    )

    plt.tight_layout()
    return fig, ax
