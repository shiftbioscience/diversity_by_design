import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.stats import ranksums # Added ranksums
import scienceplots

import sys
sys.path.append(os.path.dirname(os.getcwd())) # For finding the 'analyses' package
from common import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

DATASET_NAME = args.dataset

# Initialize analysis using the common function
(
    adata,
    pert_means, # This is the dictionary from get_pert_means(adata) 
    total_mean_original,
    ctrl_mean_original,
    DATASET_NAME,
    DATASET_CELL_COUNTS,
    DATASET_PERTS_TO_SWEEP,
    dataset_specific_subdir, # e.g. "norman19" or "replogle22"
    DATA_CACHE_DIR, # Base cache dir, e.g., "../../../data/"
    original_np_random_state,
    ANALYSIS_DIR,
    pert_normalized_abs_scores_vsrest,
    pert_counts,
    scores_df_vsrest,
    names_df_vsrest,
) = initialize_analysis(DATASET_NAME, 'sensitivity_to_niche_signals')


#### Comparison of MSE and WMSE for all perturbations via augmentation of total_mean_original
#### The goal of this step is to take total mean, agument it to set the 25 most differentially expressed genes to the mean expression of the perturbation, 
#### and then compare the MSE/WMSE of the augmented total mean to the total mean.
#### We expect WMSE to be more sensitive to the niche signals than MSE
PLOT_DIR = f'{ANALYSIS_DIR}/plots'
os.makedirs(PLOT_DIR, exist_ok=True)
N_TOP_DEGS = 25 # Define the number of top DEGs to select

results = []
# Ensure we are using perturbations for which we have all necessary data
valid_perts_for = [
    p for p in pert_means.keys() 
    if p != 'control' and 
    p in scores_df_vsrest.columns and 
    p in names_df_vsrest.columns and 
    p in pert_normalized_abs_scores_vsrest
]

for pert in tqdm(valid_perts_for, desc="Processing Perturbations"):
    current_pert_mean = pert_means[pert]
    # Ensure weights are correctly aligned with adata.var_names for wmse function
    current_pert_weights = pert_normalized_abs_scores_vsrest[pert].reindex(adata.var_names).values

    ## 1. Identify top 25 DEGs based on scores_df_vsrest for the current perturbation
    pert_scores_series = scores_df_vsrest[pert]
    pert_gene_names_series = names_df_vsrest[pert] # These are aligned by index

    # Combine scores and gene names, then calculate absolute scores
    df_scores_genes = pd.DataFrame({
        'gene': pert_gene_names_series, 
        'score': pert_scores_series
    }).dropna() # Drop rows if either gene name or score is NaN
    df_scores_genes['abs_score'] = df_scores_genes['score'].abs()
    
    # Sort by absolute score to find top DEGs
    df_sorted_degs = df_scores_genes.sort_values(by='abs_score', ascending=False)
    top_degs_gene_names = df_sorted_degs['gene'].iloc[:N_TOP_DEGS].tolist()

    ## 2. Create augmented_total_mean
    augmented_total_mean = total_mean_original.copy() # total_mean_original is aligned with adata.var_names
    
    # Get indices in adata.var_names for these top genes
    valid_top_genes_in_adata = [g for g in top_degs_gene_names if g in adata.var_names]
    top_genes_indices_in_adata = adata.var_names.get_indexer(valid_top_genes_in_adata)

    # Augment the total_mean_original for the identified top 5% DEGs
    augmented_total_mean[top_genes_indices_in_adata] = current_pert_mean[top_genes_indices_in_adata]

    ## 3. Calculate metrics for MSE/WMSE
    # Comparison 1: pert_mean vs total_mean_original
    mse_vs_total = mse(current_pert_mean, total_mean_original)
    wmse_vs_total = wmse(current_pert_mean, total_mean_original, current_pert_weights)
    
    results.append({
        'Perturbation': pert,
        'Metric': 'MSE',
        'Comparison': 'vs Original Total Mean',
        'Value': mse_vs_total
    })
    results.append({
        'Perturbation': pert,
        'Metric': 'WMSE',
        'Comparison': 'vs Original Total Mean',
        'Value': wmse_vs_total
    })

    # Comparison 2: pert_mean vs augmented_total_mean
    mse_vs_augmented = mse(current_pert_mean, augmented_total_mean)
    wmse_vs_augmented = wmse(current_pert_mean, augmented_total_mean, current_pert_weights)

    results.append({
        'Perturbation': pert,
        'Metric': 'MSE',
        'Comparison': 'vs Augmented Total Mean (Top 25 DEGs)',
        'Value': mse_vs_augmented
    })
    results.append({
        'Perturbation': pert,
        'Metric': 'WMSE',
        'Comparison': 'vs Augmented Total Mean (Top 25 DEGs)',
        'Value': wmse_vs_augmented
    })



df_results = pd.DataFrame(results)

# 4. Plotting function and calls
def plot_metric_comparison_subplots(df_full_results, plot_dir, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 6.5), sharey=False) # Increased figsize
    
    metrics_to_plot = ['MSE', 'WMSE']
    # Subplot titles will be set dynamically based on metric_type
    
    comparison_order = ['vs Original Total Mean', 'vs Augmented Total Mean (Top 25 DEGs)']
    palette = {
        'vs Original Total Mean': 'skyblue',
        'vs Augmented Total Mean (Top 25 DEGs)': 'lightcoral'
    }

    for i, metric_type in enumerate(metrics_to_plot):
        ax = axes[i]
        metric_data = df_full_results[df_full_results['Metric'] == metric_type]

        sns.violinplot(
            x='Comparison', 
            y='Value', 
            data=metric_data, 
            order=comparison_order,
            palette=palette,
            ax=ax,
            inner='quartile',
            cut=0,
            linewidth=1.5
        )
        sns.stripplot(
            x='Comparison', 
            y='Value', 
            data=metric_data, 
            order=comparison_order,
            color='black', 
            alpha=0.4, 
            size=4, 
            jitter=0.15, 
            ax=ax
        )

        ax.set_ylabel(f'{metric_type}' + '($\mu^{pert}$,$\mu^{all}$)', fontsize=16)
        ax.set_xlabel('') # X-label will be common or below the subplots
        ax.set_xticklabels(['$\mu^{all}$ \n(original)', '$\mu^{all*}$\n(Top ' + str(N_TOP_DEGS) + ' DEGs)'], rotation=0, ha="center", fontsize=14)
        ax.set_title(f'{metric_type} ({dataset_name})', fontsize=16, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Extend y-axis limit by 5% to make space for text
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.05)
        y_lim_top_new = ax.get_ylim()[1] # Get the new top limit

        # Statistical comparison
        data_orig = metric_data[metric_data['Comparison'] == comparison_order[0]]['Value'].dropna()
        data_aug = metric_data[metric_data['Comparison'] == comparison_order[1]]['Value'].dropna()

        # Position for stats text in top-left corner
        # Using 0.02 for x (left) and 0.98 for y (top) in axes coordinates
        x_text_pos_stats = 0.02 
        y_text_pos_stats_axes = 0.98

        if len(data_orig) >= 2 and len(data_aug) >= 2:
            statistic, p_val = ranksums(data_orig, data_aug, nan_policy='omit')
            
            mean_orig = data_orig.mean()
            mean_aug = data_aug.mean()
            
            fold_change_val_str = "N/A"
            if not (np.isnan(mean_orig) or np.isnan(mean_aug)):
                if mean_aug != 0:
                    fold_change_val = mean_orig / mean_aug # Fold reduction in error
                    fold_change_val_str = f"{fold_change_val:.2f}x"
                elif mean_orig == 0 and mean_aug == 0: # Both zero
                     fold_change_val_str = "No Change (Both 0)"
                else: # mean_aug is 0, mean_orig is not
                     fold_change_val_str = "Infinite (Error -> 0)"
            
            stats_text = f"P-val (Wilcoxon): {p_val:.2e}\nFold Reduction: {fold_change_val_str}"
            
            ax.text(x_text_pos_stats, y_text_pos_stats_axes, stats_text, horizontalalignment='left', 
                    verticalalignment='top', fontsize=12, transform=ax.transAxes)
                    # Removed bbox argument
        else:
            ax.text(x_text_pos_stats, y_text_pos_stats_axes, "Not enough data for Wilcoxon test", horizontalalignment='left', 
                    verticalalignment='top', fontsize=12, transform=ax.transAxes)
                    # Removed bbox argument

        # Add median values as text for each group
        for j, comp_type in enumerate(comparison_order):
            condition_data = metric_data[metric_data['Comparison'] == comp_type]['Value']
            if not condition_data.empty:
                median_val = condition_data.mean()
                if pd.notna(median_val):
                    # Position text directly at the median value, with a slight point offset to the right
                    ax.annotate(f' μ: {median_val:.4f}', # Text to display
                                xy=(j, median_val), # Point to annotate (category index, median value)
                                xytext=(20, 10),     # Offset from xy (10 points right, 0 points vertical)
                                textcoords='offset points',
                                color='black',
                                fontweight='bold',
                                ha='left', 
                                va='center', # Vertically center the text relative to median_val
                                fontsize=10)

    fig.supxlabel('Baseline compared against perturbation mean', fontsize=16, y=0.05) 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    filename = "mse_wmse_subplots_sensitivity_comparison.pdf"
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path, dpi=300)
    filename = "mse_wmse_subplots_sensitivity_comparison.png"
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path, dpi=300)
    print(f"Saved MSE vs WMSE subplots sensitivity plot to {plot_path}")
    plt.close(fig)

plot_metric_comparison_subplots(df_results, PLOT_DIR, DATASET_NAME)



# Create histogram of the perturbation weights for 4 random perturbations
# One subplot for each perturbation, with the histogram of the weights
# The weights are the normalized absolute scores of the perturbation vs the rest

if valid_perts_for:
    num_perts_to_plot = min(4, len(valid_perts_for))
    if num_perts_to_plot > 0:
        selected_perts_for_hist = np.random.choice(valid_perts_for, size=num_perts_to_plot, replace=False)
        
        # Determine subplot grid
        if num_perts_to_plot == 1:
            fig_hist, axs_hist = plt.subplots(1, 1, figsize=(6, 5))
            axs_hist = [axs_hist] # Make it iterable
        elif num_perts_to_plot == 2:
            fig_hist, axs_hist = plt.subplots(1, 2, figsize=(12, 5))
            axs_hist = axs_hist.flatten()
        elif num_perts_to_plot == 3:
            fig_hist, axs_hist = plt.subplots(1, 3, figsize=(18, 5))
            axs_hist = axs_hist.flatten()
        else: # num_perts_to_plot == 4
            fig_hist, axs_hist = plt.subplots(2, 2, figsize=(12, 10))
            axs_hist = axs_hist.flatten()

        for i, pert_name in enumerate(selected_perts_for_hist):
            ax_curr = axs_hist[i]
            if pert_name in pert_normalized_abs_scores_vsrest:
                weights = pert_normalized_abs_scores_vsrest[pert_name]
                if weights is not None and not weights.empty:
                    sns.histplot(weights, ax=ax_curr, bins=100, kde=False)
                    ax_curr.set_title(f'Weights for pert:{pert_name}', fontsize=12)
                    ax_curr.set_xlabel('Weight Value', fontsize=10)
                    ax_curr.set_ylabel('Frequency', fontsize=10)
                else:
                    ax_curr.text(0.5, 0.5, 'No weights data', ha='center', va='center', transform=ax_curr.transAxes)
                    ax_curr.set_title(f'Weights for pert:{pert_name}', fontsize=12)
            else:
                ax_curr.text(0.5, 0.5, 'Perturbation not in weights dict', ha='center', va='center', transform=ax_curr.transAxes)
                ax_curr.set_title(f'Weights for pert:{pert_name}', fontsize=12)
        
        # Remove any unused subplots if fewer than 4 were plotted in a 2x2 grid
        for j in range(num_perts_to_plot, len(axs_hist)):
            fig_hist.delaxes(axs_hist[j])

        fig_hist.suptitle(f"Distribution of DEG Weights ({DATASET_NAME})", fontsize=16, fontweight='bold')
        fig_hist.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        hist_plot_filename_pdf = "perturbation_weights_histograms.pdf"
        hist_plot_path_pdf = os.path.join(PLOT_DIR, hist_plot_filename_pdf)
        plt.savefig(hist_plot_path_pdf, dpi=300)
        
        hist_plot_filename_png = "perturbation_weights_histograms.png"
        hist_plot_path_png = os.path.join(PLOT_DIR, hist_plot_filename_png)
        plt.savefig(hist_plot_path_png, dpi=300)
        print(f"Saved perturbation weights histograms to {hist_plot_path_png} and {hist_plot_path_pdf}")
        plt.close(fig_hist)
    else:
        print("No perturbations selected for histogram plotting.")
else:
    print("No valid perturbations available to plot weight histograms.")




