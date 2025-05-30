# %%
# Disable all warnings
import warnings
warnings.filterwarnings('ignore')

# %%
import os

# Configure dataset
DATASET_NAME = 'replogle22'  # Change to 'replogle22' to use that dataset


if DATASET_NAME == 'replogle22':
    if not os.path.exists('../../data/replogle22/replogle22_processed.h5ad'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/replogle22/replogle22_processed.h5ad ../../data/replogle22/replogle22_processed.h5ad')

    if not os.path.exists('../../data/replogle22/replogle22_names_df_vsrest.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/replogle22/replogle22_names_df_vsrest.pkl ../../data/replogle22/replogle22_names_df_vsrest.pkl')

    if not os.path.exists('../../data/replogle22/replogle22_scores_df_vsrest.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/replogle22/replogle22_scores_df_vsrest.pkl ../../data/replogle22/replogle22_scores_df_vsrest.pkl')

    if not os.path.exists('../../data/gears_predictions.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/lucas/icml/gears_predictions.pkl ../../data/gears_predictions.pkl')



    if not os.path.exists('../../data/gears_predictions_default_loss_unweighted.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/lucas/icml/weight_test/gears_predictions_default_loss_unweighted.pkl ../../data/gears_predictions_default_loss_unweighted.pkl')

    if not os.path.exists('../../data/gears_predictions_default_loss_weighted.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/lucas/icml/weight_test/gears_predictions_default_loss_weighted.pkl ../../data/gears_predictions_default_loss_weighted.pkl')

    if not os.path.exists('../../data/gears_predictions_mse_unweighted.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/lucas/icml/weight_test/gears_predictions_mse_unweighted.pkl ../../data/gears_predictions_mse_unweighted.pkl')

    if not os.path.exists('../../data/gears_predictions_mse_weighted.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/lucas/icml/weight_test/gears_predictions_mse_weighted.pkl ../../data/gears_predictions_mse_weighted.pkl')

elif DATASET_NAME == 'norman19':
    if not os.path.exists('../../data/norman19/norman19_processed.h5ad'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/norman19/norman19_processed.h5ad ../../data/norman19/norman19_processed.h5ad')

    if not os.path.exists('../../data/norman19/norman19_names_df_vsrest.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/norman19/norman19_names_df_vsrest.pkl ../../data/norman19/norman19_names_df_vsrest.pkl')

    if not os.path.exists('../../data/norman19/norman19_scores_df_vsrest.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/norman19/norman19_scores_df_vsrest.pkl ../../data/norman19/norman19_scores_df_vsrest.pkl')

    if not os.path.exists('../../data/gears_predictions.norman19.pkl'):
        os.system('aws s3 cp s3://shift-personal-dev/henry/icml_data/gears_predictions.norman19.pkl ../../data/gears_predictions.norman19.pkl')


# %%
import numpy as np
import pandas as pd

# Read the numpy files based on dataset
try:
    names_df_vsrest = np.load(f'../../data/{DATASET_NAME}/{DATASET_NAME}_names_df_vsrest.pkl', allow_pickle=True)
    print(f"Successfully loaded names_df_vsrest for {DATASET_NAME}")
except Exception as e:
    print(f"Error loading names_df_vsrest: {e}")

try:
    scores_df_vsrest = np.load(f'../../data/{DATASET_NAME}/{DATASET_NAME}_scores_df_vsrest.pkl', allow_pickle=True)
    print(f"Successfully loaded scores_df_vsrest for {DATASET_NAME}")
except Exception as e:
    print(f"Error loading scores_df_vsrest: {e}")


# %%
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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import os
from scipy.stats import ranksums # Added ranksums
import scienceplots


# DATASET_NAME is already defined at the top of the file

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
) = initialize_analysis(DATASET_NAME, 'modeling_with_gears')

# %%
import pickle

# Load predictions only for replogle22
if DATASET_NAME == 'replogle22':
    # Load the gears predictions
    with open('../../data/gears_predictions.pkl', 'rb') as f:
        gears_predictions = pickle.load(f)

    with open('../../data/gears_predictions_default_loss_unweighted.pkl', 'rb') as f:
        gears_predictions_default_loss_unweighted = pickle.load(f)

    with open('../../data/gears_predictions_default_loss_weighted.pkl', 'rb') as f:
        gears_predictions_default_loss_weighted = pickle.load(f)

    with open('../../data/gears_predictions_mse_unweighted.pkl', 'rb') as f:
        gears_predictions_mse_unweighted = pickle.load(f)

    with open('../../data/gears_predictions_mse_weighted.pkl', 'rb') as f:
        gears_predictions_mse_weighted = pickle.load(f)


else:

    with open('../../data/gears_predictions.norman19.pkl', 'rb') as f:
        gears_predictions = pickle.load(f)

    # Set empty dict for GEARS predictions
    gears_predictions_default_loss_unweighted = {}
    gears_predictions_default_loss_weighted = {}
    gears_predictions_mse_unweighted = {}
    gears_predictions_mse_weighted = {}

    # Sub + for _
    gears_predictions = {k.replace('_', '+'): v for k, v in gears_predictions.items()}


first_half_cells = []
second_half_cells = []
for pert in tqdm(pert_means.keys(), desc="Processing perturbations"):
    # Get all cells for this perturbation
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    
    # Randomly shuffle the cells and split into two halves
    np.random.shuffle(pert_cells)
    split_idx = len(pert_cells) // 2
    first_half_cells.extend(pert_cells[:split_idx])
    second_half_cells.extend(pert_cells[split_idx:])

adata_first_half = adata[first_half_cells].copy()
adata_second_half = adata[second_half_cells].copy()

# %%
# Get means for first half and second half
pert_means_first_half = get_pert_means(adata_first_half)
total_mean_first_half = np.mean(list(pert_means_first_half.values()), axis=0)
pert_means_second_half = get_pert_means(adata_second_half)
total_mean_second_half = np.mean(list(pert_means_second_half.values()), axis=0)

# %%
# Create dictionaries to store the metrics for each perturbation
pearson_delta_dict_predictive = {}
pearson_delta_degs_dict_predictive = {}
mse_dict_predictive = {}
wmse_dict_predictive = {}
r2_delta_dict_predictive = {}
wr2_delta_dict_predictive = {}


# %%
# Skip 'control' perturbation and focus only on actual perturbations
if DATASET_NAME == 'replogle22':
    # For replogle22, filter by GEARS predictions availability
    all_perts_for_predictive = [pert for pert in adata.obs['condition'].unique() if pert != 'control' and pert in list(gears_predictions.keys())]
else:
    # For other datasets, use all non-control perturbations
    all_perts_for_predictive = [pert for pert in adata.obs['condition'].unique() if pert != 'control' and pert in list(gears_predictions.keys())]

MIN_DEGS_FOR_METRIC = 0

# Helper function to calculate metrics for a given prediction
def calculate_metrics_for_condition(pred_mean, second_half_mean, key_suffix, pert, 
                                   current_pert_weights, pert_degs_vsrest_idx,
                                   delta_control_mean=total_mean_first_half,
                                   special_pearson_handling=False):
    """Calculate all metrics for a given condition/prediction"""
    key = f"{pert}_{key_suffix}" if key_suffix else pert
    
    # Basic metrics
    mse_dict_predictive[key] = mse(pred_mean, second_half_mean)
    if pert_degs_vsrest_idx.sum() > MIN_DEGS_FOR_METRIC:
        wmse_dict_predictive[key] = wmse(pred_mean, second_half_mean, current_pert_weights)
    else:
        wmse_dict_predictive[key] = np.nan
    
    # Delta metrics
    delta_pred = pred_mean - delta_control_mean
    delta_second_half = second_half_mean - total_mean_first_half
    
    # Special handling for data mean (pearson = NaN)
    if special_pearson_handling:
        pearson_delta_dict_predictive[key] = np.nan
        pearson_delta_degs_dict_predictive[key] = np.nan
    else:
        pearson_delta_dict_predictive[key] = pearson(delta_pred, delta_second_half)
        if pert_degs_vsrest_idx.sum() > MIN_DEGS_FOR_METRIC:
            pearson_delta_degs_dict_predictive[key] = pearson(delta_pred[pert_degs_vsrest_idx], 
                                                             delta_second_half[pert_degs_vsrest_idx])
        else:
            pearson_delta_degs_dict_predictive[key] = np.nan
    
    # R2 metrics
    r2_delta_dict_predictive[key] = r2_score_on_deltas(delta_second_half, delta_pred)
    if pert_degs_vsrest_idx.sum() > MIN_DEGS_FOR_METRIC:
        wr2_delta_dict_predictive[key] = r2_score_on_deltas(delta_second_half, delta_pred, current_pert_weights)
    else:
        wr2_delta_dict_predictive[key] = np.nan

# %%

for pert in tqdm(all_perts_for_predictive, desc="Processing perturbations"):
    # Get mean expressions
    first_half_mean = adata_first_half[adata_first_half.obs['condition'] == pert].X.mean(axis=0).A1
    second_half_mean = adata_second_half[adata_second_half.obs['condition'] == pert].X.mean(axis=0).A1
    
    # Get DEG info (only calculate once)
    current_pert_weights = pert_normalized_abs_scores_vsrest.get(pert)
    pert_degs_vsrest = list(set(adata.uns['deg_dict_vsrest'][pert]['up']) | set(adata.uns['deg_dict_vsrest'][pert]['down']))
    pert_degs_vsrest_idx = adata.var_names.isin(pert_degs_vsrest)
    
    # Calculate metrics for each condition
    # Tech Duplicate
    calculate_metrics_for_condition(first_half_mean, second_half_mean, None, pert, 
                                  current_pert_weights, pert_degs_vsrest_idx)
    
    # Data mean baseline
    calculate_metrics_for_condition(total_mean_original, second_half_mean, "datamean", pert,
                                  current_pert_weights, pert_degs_vsrest_idx,
                                  special_pearson_handling=True)
    
    # Control baseline
    calculate_metrics_for_condition(ctrl_mean_original, second_half_mean, "control", pert,
                                  current_pert_weights, pert_degs_vsrest_idx)
    
    # Additive baseline (only for Norman19 and double perturbations)
    if DATASET_NAME == 'norman19' and '+' in pert:
        # Split the double perturbation to get individual genes
        genes = pert.split('+')
        if len(genes) == 2:
            gene1, gene2 = genes
            # Check if we have data for both single perturbations
            if gene1 in pert_means_first_half and gene2 in pert_means_first_half:
                # Calculate effects of single perturbations
                effect1 = pert_means_first_half[gene1] - total_mean_first_half
                effect2 = pert_means_first_half[gene2] - total_mean_first_half
                # Predict double perturbation as additive combination
                additive_pred = total_mean_first_half + effect1 + effect2
                calculate_metrics_for_condition(additive_pred, second_half_mean, "additive", pert,
                                              current_pert_weights, pert_degs_vsrest_idx)
    
    # GEARS (only if available)
    if pert in gears_predictions:
        gears_mean = gears_predictions.get(pert)
        calculate_metrics_for_condition(gears_mean, second_half_mean, "gears", pert,
                                      current_pert_weights, pert_degs_vsrest_idx)

    if pert in gears_predictions_default_loss_unweighted:
        gears_mean = gears_predictions_default_loss_unweighted.get(pert)
        calculate_metrics_for_condition(gears_mean, second_half_mean, "gears_default_loss_unweighted", pert,
                                      current_pert_weights, pert_degs_vsrest_idx)

    if pert in gears_predictions_default_loss_weighted:
        gears_mean = gears_predictions_default_loss_weighted.get(pert)
        calculate_metrics_for_condition(gears_mean, second_half_mean, "gears_default_loss_weighted", pert,
                                        current_pert_weights, pert_degs_vsrest_idx)

    if pert in gears_predictions_mse_unweighted:
        gears_mean = gears_predictions_mse_unweighted.get(pert)
        calculate_metrics_for_condition(gears_mean, second_half_mean, "gears_mse_unweighted", pert,
                                        current_pert_weights, pert_degs_vsrest_idx)

    if pert in gears_predictions_mse_weighted:
        gears_mean = gears_predictions_mse_weighted.get(pert)
        calculate_metrics_for_condition(gears_mean, second_half_mean, "gears_mse_weighted", pert,
                                        current_pert_weights, pert_degs_vsrest_idx)





# %%
# Create plots for the predictive baseline metrics
PLOT_DIR = f'{ANALYSIS_DIR}/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)

# Process data for plotting
# Split keys into three groups based on suffix - regular, _control, and _datamean
regular_keys = [key for key in mse_dict_predictive.keys() if '_control' not in key and '_datamean' not in key and '_gears' not in key and '_additive' not in key]
control_keys = [key for key in mse_dict_predictive.keys() if '_control' in key]
datamean_keys = [key for key in mse_dict_predictive.keys() if '_datamean' in key]
gears_keys = [key for key in mse_dict_predictive.keys() if '_gears' in key]
gears_default_loss_unweighted_keys = [key for key in mse_dict_predictive.keys() if '_gears_default_loss_unweighted' in key]
gears_default_loss_weighted_keys = [key for key in mse_dict_predictive.keys() if '_gears_default_loss_weighted' in key]
gears_mse_unweighted_keys = [key for key in mse_dict_predictive.keys() if '_gears_mse_unweighted' in key]
gears_mse_weighted_keys = [key for key in mse_dict_predictive.keys() if '_gears_mse_weighted' in key]

# Create restructured dataframes for side-by-side condition comparison
data_for_plotting = []

# Define metrics and their corresponding dictionaries
metrics_config = [
    ('MSE', mse_dict_predictive),
    ('WMSE', wmse_dict_predictive),
    ('Pearson Delta', pearson_delta_dict_predictive),
    ('Pearson Delta DEGs', pearson_delta_degs_dict_predictive),
    ('R-Squared Delta', r2_delta_dict_predictive),
    ('Weighted R-Squared Delta', wr2_delta_dict_predictive)
]

# Define conditions and their display names
conditions_config = [
    (None, 'Tech Duplicate'),  # None suffix for base perturbation
    ('_control', '$\mu^c$\n(ctrl mean)'),
    ('_datamean', '$\mu^{all}$\n(perts mean)'),
]

# Add GEARS only if available
if DATASET_NAME == 'replogle22':
    conditions_config.extend([
        ('_gears', 'GEARS'),
        ('_gears_default_loss_unweighted', 'GEARS\n(Default, Unweighted)'),
        ('_gears_default_loss_weighted', 'GEARS\n(Default, Weighted)'),
        ('_gears_mse_unweighted', 'GEARS\n(MSE, Unweighted)'),
        ('_gears_mse_weighted', 'GEARS\n(MSE, Weighted)')
    ])

if DATASET_NAME == 'norman19':
    conditions_config.extend([
        ('_additive', 'Additive'),
        ('_gears', 'GEARS'),
    ])

# Process all metrics
for metric_name, metric_dict in metrics_config:
    for base_pert in regular_keys:
        # Check if all required keys exist (base 3 conditions: tech duplicate, control, datamean)
        base_required_keys = [base_pert, f"{base_pert}_control", f"{base_pert}_datamean"]
        required_keys_exist = all(key in metric_dict for key in base_required_keys)
        
        if required_keys_exist:
            # Add data for each condition
            for suffix, condition_name in conditions_config:
                key = f"{base_pert}{suffix}" if suffix else base_pert
                if key in metric_dict:
                    data_for_plotting.append({
                        'Perturbation': base_pert,
                        'Metric': metric_name,
                        'Condition': condition_name,
                        'Value': metric_dict[key]
                    })

# Create main DataFrame for plotting
df_for_plotting = pd.DataFrame(data_for_plotting)

# %%

# Function to create comparison violin plots for the three conditions
def plot_predictive_conditions_boxplot(df, metric_name, y_label, plot_title, plot_dir, dataset_name, plot_suffix=''):
    # Filter for just this metric
    df_metric = df[df['Metric'] == metric_name].copy()
    
    # For Pearson Delta metrics, remove data for μ^all condition
    if metric_name in ['Pearson Delta', 'Pearson Delta DEGs']:
        df_metric_plot = df_metric[df_metric['Condition'] != '$\mu^{all}$\n(perts mean)'].copy()
    else:
        df_metric_plot = df_metric.copy()
    
    # Adjust figure size based on dataset and plot type
    if DATASET_NAME == 'norman19':
        x_inches = 8
    elif plot_suffix == 'basic':
        x_inches = 10  # Smaller width for basic replogle22 plots
    else:
        x_inches = 22
    plt.figure(figsize=(x_inches, 7))
    ax = plt.gca()
    
    # Define colors for conditions
    condition_colors = {
        'Tech Duplicate': 'steelblue',
        '$\mu^c$\n(ctrl mean)': 'forestgreen',
        '$\mu^{all}$\n(perts mean)': 'indianred',
        'Additive': 'darkorange',
        'GEARS': 'purple',
        'GEARS\n(Default, Unweighted)': 'purple',
        'GEARS\n(Default, Weighted)': 'purple',
        'GEARS\n(MSE, Unweighted)': 'purple',
        'GEARS\n(MSE, Weighted)': 'purple'
    }
    
    # Define the condition order based on what's actually available in the data
    base_condition_order = ['$\mu^c$\n(ctrl mean)', '$\mu^{all}$\n(perts mean)']

    if 'GEARS' in df_metric['Condition'].values:
        base_condition_order.append('GEARS')
    if 'GEARS\n(Default, Unweighted)' in df_metric['Condition'].values:
        base_condition_order.append('GEARS\n(Default, Unweighted)')
    if 'GEARS\n(Default, Weighted)' in df_metric['Condition'].values:
        base_condition_order.append('GEARS\n(Default, Weighted)')
    if 'GEARS\n(MSE, Unweighted)' in df_metric['Condition'].values:
        base_condition_order.append('GEARS\n(MSE, Unweighted)')
    if 'GEARS\n(MSE, Weighted)' in df_metric['Condition'].values:
        base_condition_order.append('GEARS\n(MSE, Weighted)')
    if 'Additive' in df_metric['Condition'].values:
        base_condition_order.append('Additive')
    base_condition_order.append('Tech Duplicate')
    
    # For Pearson Delta metrics, keep μ^all in condition_order even if no data
    if metric_name in ['Pearson Delta', 'Pearson Delta DEGs']:
        condition_order = base_condition_order  # Keep all conditions including μ^all
    else:
        # Filter condition_order to only include conditions present in the data
        available_conditions = df_metric_plot['Condition'].unique()
        condition_order = [c for c in base_condition_order if c in available_conditions]
    
    # Create violin plots with conditions side by side
    violinplot = sns.violinplot(
        x='Condition', 
        y='Value', 
        data=df_metric_plot,
        palette=condition_colors,
        ax=ax,
        order=condition_order,
        inner='quartile',  # Show quartiles inside the violins
        cut=0              # Don't extend beyond observed data
    )
    
    # Add individual points
    sns.stripplot(
        x='Condition', 
        y='Value', 
        data=df_metric_plot,
        color='black', 
        size=3, 
        alpha=0.3,
        ax=ax,
        dodge=True,
        order=condition_order
    )
    
    # Add mean values for each condition in black, using Greek μ (mu) symbol
    for i, condition in enumerate(condition_order):
        # Special handling for Pearson Delta metrics and μ^all condition
        if metric_name in ['Pearson Delta', 'Pearson Delta DEGs'] and condition == '$\mu^{all}$\n(perts mean)':
            # Add "NaN" text at y=0.02 for μ^all condition
            ax.text(
                i, 0.02, 
                'NaN', 
                color='black',
                fontweight='bold',
                ha='center', 
                va='bottom',
                fontsize=12  # Slightly larger font for NaN
            )
        else:
            condition_data = df_metric[df_metric['Condition'] == condition]['Value']
            if not condition_data.empty:
                median_val = condition_data.median()
                mean_val = condition_data.mean()
                # if not np.isnan(mean_val):
                #     yloc = mean_val * 1.02 if mean_val > -1 else -.94
                #     ax.text(
                #         i + 0.15, yloc, 
                #         f'μ: {mean_val:.3f}', 
                #         color='black',
                #         fontweight='bold',
                #         ha='left', 
                #         va='bottom'
                #     )
                if not np.isnan(median_val):
                    if metric_name in ['MSE', 'WMSE']:
                        yloc = median_val * 1.15 if median_val > -1 else -.94
                    else:
                        yloc = median_val + 0.02 if median_val > -1 else -.94 # Pearson Delta, Pearson Delta DEGs, R-Squared Delta, Weighted R-Squared Delta
                    ax.text(
                        i + 0.1, yloc, 
                        f'Med: {median_val:.3f}', 
                        color='black',
                        fontweight='bold',
                        ha='left', 
                        va='bottom',
                        fontsize=10  # Increased font size
                    )
    
    # Add a horizontal line at y=0 for R-squared and Pearson delta plots
    if metric_name in ['R-Squared', 'Pearson Delta', "Pearson Delta DEGs", 'R-Squared Delta', 'Weighted R-Squared Delta']:
        ax.axhline(y=0, color='firebrick', linestyle='--', linewidth=0.8, zorder=20, alpha=0.7)
        ax.set_ylim(-1.1, 1.15) # Set Y-axis from -1 to 1, with a little padding

        # Count and annotate points below -1 for each condition
        for i, condition in enumerate(condition_order):
            # Skip outlier count for μ^all in Pearson Delta metrics
            if metric_name in ['Pearson Delta', 'Pearson Delta DEGs'] and condition == '$\mu^{all}$\n(perts mean)':
                continue
            condition_data = df_metric[df_metric['Condition'] == condition]
            num_outliers = (condition_data['Value'] < -1).sum()
            if num_outliers > 0:
                ax.text(
                    i + 0.1, -1, 
                    f'N < -1: {num_outliers}', 
                    color='black',
                    fontweight='bold',
                    ha='left', 
                    va='bottom',
                    fontsize=10  # Increased font size
                )
    
    # Format plot
    plt.title(f'{plot_title} ({dataset_name})', fontsize=18, fontweight='bold')
    plt.ylabel(y_label, fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=14)
    
    # Add count of perturbations
    unique_perts = df_metric['Perturbation'].nunique()
    
    # Incorporate the optional suffix in the filename to differentiate between regular and DEG plots
    filename = f"condition_comparison_{metric_name.lower().replace(' ', '_')}"
    # Check if DEGs name
    
    if plot_suffix:
        filename += f"_{plot_suffix}"
    plot_path = f"{plot_dir}/{filename}.png"
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()

# %%
# For replogle22, create two different sets of plots
if DATASET_NAME == 'replogle22':
    # First set: Only Tech Duplicate, $\mu^c$, $\mu^{all}$, and GEARS
    df_for_plotting_basic = df_for_plotting[df_for_plotting['Condition'].isin([
        'Tech Duplicate', '$\mu^c$\n(ctrl mean)', '$\mu^{all}$\n(perts mean)', 'GEARS'
    ])].copy()
    
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'MSE', 'MSE (vs Second Half)', 'MSE (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'WMSE', 'WMSE (vs Second Half)', 'WMSE (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'R-Squared Delta', r'$R^2(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'Weighted R-Squared Delta', r'$R^2_w(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2_w(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'Pearson Delta', r'Pearson($\Delta$) ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    plot_predictive_conditions_boxplot(df_for_plotting_basic, 'Pearson Delta DEGs', r'Pearson($\Delta$) DEGs ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) DEGs (vs Second Half)', PLOT_DIR, DATASET_NAME, 'basic')
    
    # Second set: Tech Duplicate, $\mu^c$, $\mu^{all}$, GEARS (MSE, Unweighted), and GEARS (MSE, Weighted)
    df_for_plotting_mse = df_for_plotting[df_for_plotting['Condition'].isin([
        'Tech Duplicate', '$\mu^c$\n(ctrl mean)', '$\mu^{all}$\n(perts mean)', 
        'GEARS\n(MSE, Unweighted)', 'GEARS\n(MSE, Weighted)'
    ])].copy()
    
    # Create a modified plotting function for the MSE comparison
    from scipy.stats import ttest_rel
    
    def plot_predictive_conditions_boxplot_with_ttest(df, metric_name, y_label, plot_title, plot_dir, dataset_name, plot_suffix=''):
        # Filter for just this metric
        df_metric = df[df['Metric'] == metric_name].copy()
        
        # For Pearson Delta metrics, remove data for μ^all condition
        if metric_name in ['Pearson Delta', 'Pearson Delta DEGs']:
            df_metric_plot = df_metric[df_metric['Condition'] != '$\mu^{all}$\n(perts mean)'].copy()
        else:
            df_metric_plot = df_metric.copy()
        
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        
        # Define colors for conditions
        condition_colors = {
            'Tech Duplicate': 'steelblue',
            '$\mu^c$\n(ctrl mean)': 'forestgreen',
            '$\mu^{all}$\n(perts mean)': 'indianred',
            'GEARS\n(MSE, Unweighted)': 'mediumpurple',
            'GEARS\n(MSE, Weighted)': 'darkviolet'
        }
        
        # Define the condition order
        base_condition_order = ['$\mu^c$\n(ctrl mean)', '$\mu^{all}$\n(perts mean)', 
                                'GEARS\n(MSE, Unweighted)', 'GEARS\n(MSE, Weighted)', 
                                'Tech Duplicate']
        
        # For Pearson Delta metrics, keep μ^all in condition_order even if no data
        if metric_name in ['Pearson Delta', 'Pearson Delta DEGs']:
            condition_order = base_condition_order
        else:
            available_conditions = df_metric_plot['Condition'].unique()
            condition_order = [c for c in base_condition_order if c in available_conditions]
        
        # Create violin plots
        violinplot = sns.violinplot(
            x='Condition', 
            y='Value', 
            data=df_metric_plot,
            palette=condition_colors,
            ax=ax,
            order=condition_order,
            inner='quartile',
            cut=0
        )
        
        # Add individual points
        sns.stripplot(
            x='Condition', 
            y='Value', 
            data=df_metric_plot,
            color='black', 
            size=3, 
            alpha=0.3,
            ax=ax,
            dodge=True,
            order=condition_order
        )
        
        # Calculate t-test between GEARS (MSE, Unweighted) and GEARS (MSE, Weighted)
        unweighted_data = df_metric[df_metric['Condition'] == 'GEARS\n(MSE, Unweighted)']['Value'].dropna()
        weighted_data = df_metric[df_metric['Condition'] == 'GEARS\n(MSE, Weighted)']['Value'].dropna()
        
        if len(unweighted_data) > 0 and len(weighted_data) > 0:
            # Get paired data
            unweighted_perts = df_metric[df_metric['Condition'] == 'GEARS\n(MSE, Unweighted)']['Perturbation'].values
            weighted_perts = df_metric[df_metric['Condition'] == 'GEARS\n(MSE, Weighted)']['Perturbation'].values
            common_perts = set(unweighted_perts) & set(weighted_perts)
            
            paired_unweighted = []
            paired_weighted = []
            for pert in common_perts:
                unw_val = df_metric[(df_metric['Condition'] == 'GEARS\n(MSE, Unweighted)') & 
                                    (df_metric['Perturbation'] == pert)]['Value'].values[0]
                w_val = df_metric[(df_metric['Condition'] == 'GEARS\n(MSE, Weighted)') & 
                                  (df_metric['Perturbation'] == pert)]['Value'].values[0]
                if not np.isnan(unw_val) and not np.isnan(w_val):
                    paired_unweighted.append(unw_val)
                    paired_weighted.append(w_val)
            
            if len(paired_unweighted) > 1:
                t_stat, p_value = ttest_rel(paired_unweighted, paired_weighted)
                
                # Add t-test result to plot
                y_max = ax.get_ylim()[1]
                y_pos = y_max * 0.95
                
                # Draw significance bracket
                unw_idx = condition_order.index('GEARS\n(MSE, Unweighted)')
                w_idx = condition_order.index('GEARS\n(MSE, Weighted)')
                
                ax.plot([unw_idx, unw_idx, w_idx, w_idx], 
                        [y_pos*0.92, y_pos*0.94, y_pos*0.94, y_pos*0.92], 
                        'k-', linewidth=1)
                
                # Add significance text
                sig_text = 'n.s.'
                if p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                elif p_value < 0.05:
                    sig_text = '*'
                
                ax.text((unw_idx + w_idx) / 2, y_pos*0.95, 
                        f't={t_stat:.2f}, p={p_value:.3f}\n{sig_text}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add median values for each condition
        for i, condition in enumerate(condition_order):
            if metric_name in ['Pearson Delta', 'Pearson Delta DEGs'] and condition == '$\mu^{all}$\n(perts mean)':
                ax.text(i, 0.02, 'NaN', color='black', fontweight='bold',
                        ha='center', va='bottom', fontsize=12)
            else:
                condition_data = df_metric[df_metric['Condition'] == condition]['Value']
                if not condition_data.empty:
                    median_val = condition_data.median()
                    if not np.isnan(median_val):
                        if metric_name in ['MSE', 'WMSE']:
                            yloc = median_val * 1.15 if median_val > -1 else -.94
                        else:
                            yloc = median_val + 0.02 if median_val > -1 else -.94
                        ax.text(i + 0.1, yloc, f'Med: {median_val:.3f}', 
                                color='black', fontweight='bold',
                                ha='left', va='bottom', fontsize=10)
        
        # Add horizontal line for certain metrics
        if metric_name in ['R-Squared', 'Pearson Delta', "Pearson Delta DEGs", 'R-Squared Delta', 'Weighted R-Squared Delta']:
            ax.axhline(y=0, color='firebrick', linestyle='--', linewidth=0.8, zorder=20, alpha=0.7)
            ax.set_ylim(-1.1, 1.15)
            
            # Count outliers
            for i, condition in enumerate(condition_order):
                if metric_name in ['Pearson Delta', 'Pearson Delta DEGs'] and condition == '$\mu^{all}$\n(perts mean)':
                    continue
                condition_data = df_metric[df_metric['Condition'] == condition]
                num_outliers = (condition_data['Value'] < -1).sum()
                if num_outliers > 0:
                    ax.text(i + 0.1, -1, f'N < -1: {num_outliers}', 
                            color='black', fontweight='bold',
                            ha='left', va='bottom', fontsize=10)
        
        # Format plot
        plt.title(f'{plot_title} ({dataset_name})', fontsize=18, fontweight='bold')
        plt.ylabel(y_label, fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', labelsize=14)
        
        filename = f"condition_comparison_{metric_name.lower().replace(' ', '_')}"
        if plot_suffix:
            filename += f"_{plot_suffix}"
        plot_path = f"{plot_dir}/{filename}.png"
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")
        plt.show()
        plt.close()
    
    # Use the modified function for MSE comparison plots
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'MSE', 'MSE (vs Second Half)', 'MSE (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'WMSE', 'WMSE (vs Second Half)', 'WMSE (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'R-Squared Delta', r'$R^2(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'Weighted R-Squared Delta', r'$R^2_w(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2_w(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'Pearson Delta', r'Pearson($\Delta$) ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    plot_predictive_conditions_boxplot_with_ttest(df_for_plotting_mse, 'Pearson Delta DEGs', r'Pearson($\Delta$) DEGs ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) DEGs (vs Second Half)', PLOT_DIR, DATASET_NAME, 'mse_comparison')
    
else:
    # For other datasets, use the original plotting
    plot_predictive_conditions_boxplot(df_for_plotting, 'MSE', 'MSE (vs Second Half)', 'MSE (vs Second Half)', PLOT_DIR, DATASET_NAME)
    plot_predictive_conditions_boxplot(df_for_plotting, 'WMSE', 'WMSE (vs Second Half)', 'WMSE (vs Second Half)', PLOT_DIR, DATASET_NAME)
    plot_predictive_conditions_boxplot(df_for_plotting, 'R-Squared Delta', r'$R^2(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME)
    plot_predictive_conditions_boxplot(df_for_plotting, 'Weighted R-Squared Delta', r'$R^2_w(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', r'$R^2_w(\Delta)$ (vs Second Half)', PLOT_DIR, DATASET_NAME)
    plot_predictive_conditions_boxplot(df_for_plotting, 'Pearson Delta', r'Pearson($\Delta$) ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) (vs Second Half)', PLOT_DIR, DATASET_NAME)
    plot_predictive_conditions_boxplot(df_for_plotting, 'Pearson Delta DEGs', r'Pearson($\Delta$) DEGs ($\mu^{all}$ as $\Delta_{ctrl}$)', r'Pearson($\Delta$) DEGs (vs Second Half)', PLOT_DIR, DATASET_NAME)

# %%
# Check whether MSE, weighted vs MSE, Unweighted is significantly different with a Wilcoxon test
# Check this for all metrics
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

# Perform Wilcoxon rank sum tests comparing weighted vs unweighted versions
print("\n=== Statistical Comparison: Weighted vs Unweighted GEARS Models ===\n")

# Define the comparison pairs
comparison_pairs = [
    ('Default Loss', 'gears_default_loss_unweighted', 'gears_default_loss_weighted'),
    ('MSE Loss', 'gears_mse_unweighted', 'gears_mse_weighted')
]

# Define metrics to compare
metrics_to_compare = [
    ('MSE', mse_dict_predictive),
    ('WMSE', wmse_dict_predictive),
    ('Pearson Delta', pearson_delta_dict_predictive),
    ('Pearson Delta DEGs', pearson_delta_degs_dict_predictive),
    ('R-Squared Delta', r2_delta_dict_predictive),
    ('Weighted R-Squared Delta', wr2_delta_dict_predictive)
]

# Perform comparisons
for loss_type, unweighted_suffix, weighted_suffix in comparison_pairs:
    print(f"\n{'='*60}")
    print(f"Comparison: {loss_type} - Unweighted vs Weighted")
    print(f"{'='*60}\n")
    
    for metric_name, metric_dict in metrics_to_compare:
        # Collect paired values
        unweighted_values = []
        weighted_values = []
        perturbations = []
        
        # Get all perturbations that have both weighted and unweighted predictions
        for pert in all_perts_for_predictive:
            unweighted_key = f"{pert}_{unweighted_suffix}"
            weighted_key = f"{pert}_{weighted_suffix}"
            
            if unweighted_key in metric_dict and weighted_key in metric_dict:
                unweighted_val = metric_dict[unweighted_key]
                weighted_val = metric_dict[weighted_key]
                
                # Only include if both values are not NaN
                if not np.isnan(unweighted_val) and not np.isnan(weighted_val):
                    unweighted_values.append(unweighted_val)
                    weighted_values.append(weighted_val)
                    perturbations.append(pert)
        
        if len(unweighted_values) > 0:
            # Convert to numpy arrays
            unweighted_values = np.array(unweighted_values)
            weighted_values = np.array(weighted_values)
            
            # Perform Wilcoxon rank sum test
            stat, p_value = ranksums(unweighted_values, weighted_values)
            
            # Calculate statistics
            unweighted_mean = np.mean(unweighted_values)
            unweighted_median = np.median(unweighted_values)
            unweighted_positive = np.sum(unweighted_values > 0)
            
            weighted_mean = np.mean(weighted_values)
            weighted_median = np.median(weighted_values)
            weighted_positive = np.sum(weighted_values > 0)
            
            # Calculate mean difference
            mean_diff = weighted_mean - unweighted_mean
            median_diff = weighted_median - unweighted_median
            
            print(f"{metric_name}:")
            print(f"  Sample size: {len(unweighted_values)} perturbations")
            print(f"  Unweighted - Mean: {unweighted_mean:.4f}, Median: {unweighted_median:.4f}, N>0: {unweighted_positive}")
            print(f"  Weighted   - Mean: {weighted_mean:.4f}, Median: {weighted_median:.4f}, N>0: {weighted_positive}")
            print(f"  Difference - Mean: {mean_diff:.4f}, Median: {median_diff:.4f}")
            print(f"  Wilcoxon rank sum test: stat={stat:.4f}, p={p_value:.4e}")
            
            # Significance interpretation
            if p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            elif p_value < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            print(f"  Significance: {sig_text}")
            
            # For metrics where higher is better (correlation/R2 metrics)
            if metric_name in ['Pearson Delta', 'Pearson Delta DEGs', 'R-Squared Delta', 'Weighted R-Squared Delta']:
                if mean_diff > 0 and p_value < 0.05:
                    print(f"  → Weighted is significantly BETTER (higher)")
                elif mean_diff < 0 and p_value < 0.05:
                    print(f"  → Unweighted is significantly BETTER (higher)")
            # For metrics where lower is better (MSE/WMSE)
            else:
                if mean_diff < 0 and p_value < 0.05:
                    print(f"  → Weighted is significantly BETTER (lower)")
                elif mean_diff > 0 and p_value < 0.05:
                    print(f"  → Unweighted is significantly BETTER (lower)")
            
            print()
        else:
            print(f"{metric_name}: No valid paired data available\n")



# %%
# Stop here for datasets without GEARS/scGPT predictions
if DATASET_NAME != 'replogle22':
    print("\n=== Analysis complete for", DATASET_NAME, "===")
    print("Skipping GEARS/scGPT-specific analyses as they are not available for this dataset.")
    import sys
    sys.exit(0)

# %%
# Find the perturbation which has the highest pearson delta DEGs in GEARS predictions
# Find the perturbation which has the highest pearson delta DEGs in GEARS predictions
gears_pearson_delta_degs = {pert.replace('_gears', ''): value 
                            for pert, value in pearson_delta_degs_dict_predictive.items() 
                            if '_gears' in pert and not pd.isna(value)}

if gears_pearson_delta_degs:
    max_pert_gears = max(gears_pearson_delta_degs, key=gears_pearson_delta_degs.get)
    max_value_gears = gears_pearson_delta_degs[max_pert_gears]
    print(f"Perturbation with highest Pearson delta DEGs (GEARS): {max_pert_gears}, Value: {max_value_gears}")
else:
    print("No GEARS predictions found or all values are NaN.")


# %%
# Show the corelation plot between the GEARS prediction and ground truth for the selected perturbation
if gears_pearson_delta_degs:
    # Get the GEARS prediction and ground truth for the best perturbation
    selected_pert = max_pert_gears
    
    # Get GEARS prediction
    gears_pred = gears_predictions[selected_pert]
    
    # Get ground truth (second half mean)
    ground_truth = adata_second_half[adata_second_half.obs['condition'] == selected_pert].X.mean(axis=0).A1
    
    # Get DEGs for this perturbation
    pert_degs = list(set(adata.uns['deg_dict_vsrest'][selected_pert]['up']) | 
                     set(adata.uns['deg_dict_vsrest'][selected_pert]['down']))
    pert_degs_idx = adata.var_names.isin(pert_degs)
    
    # Get weights for this perturbation
    current_pert_weights = pert_normalized_abs_scores_vsrest.get(selected_pert)
    
    # Calculate deltas
    delta_gears = gears_pred - total_mean_first_half
    delta_ground_truth = ground_truth - total_mean_first_half
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: All genes
    ax1.scatter(delta_ground_truth, delta_gears, alpha=0.3, s=1, color='gray', label='All genes')
    
    # Add diagonal line
    lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
            max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    # Calculate and display correlation and R2
    corr_all = pearson(delta_ground_truth, delta_gears)
    r2_all = r2_score_on_deltas(delta_ground_truth, delta_gears, current_pert_weights)
    ax1.set_xlabel('Ground Truth ($\Delta$ Expression)', fontsize=12)
    ax1.set_ylabel('GEARS Prediction ($\Delta$ Expression)', fontsize=12)
    ax1.set_title(f'{selected_pert} - All Genes', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add metrics as text in top left corner
    ax1.text(0.05, 0.95, f'Pearson r = {corr_all:.3f}\nR² = {r2_all:.3f}',
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Plot 2: DEGs only
    ax2.scatter(delta_ground_truth[pert_degs_idx], delta_gears[pert_degs_idx], 
                alpha=0.6, s=10, color='darkred', label='DEGs')
    
    # Add diagonal line
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    # Calculate and display correlation and R2 for DEGs
    corr_degs = pearson(delta_ground_truth[pert_degs_idx], delta_gears[pert_degs_idx])
    r2_degs = r2_score_on_deltas(delta_ground_truth[pert_degs_idx], delta_gears[pert_degs_idx], 
                       current_pert_weights[pert_degs_idx])
    ax2.set_xlabel('Ground Truth ($\Delta$ Expression)', fontsize=12)
    ax2.set_ylabel('GEARS Prediction ($\Delta$ Expression)', fontsize=12)
    ax2.set_title(f'{selected_pert} - DEGs Only ({pert_degs_idx.sum()} genes)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add metrics as text in top left corner
    ax2.text(0.05, 0.95, f'Pearson r = {corr_degs:.3f}\nR² = {r2_degs:.3f}',
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{PLOT_DIR}/gears_correlation_best_pert_{selected_pert}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{PLOT_DIR}/gears_correlation_best_pert_{selected_pert}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print additional statistics
    print(f"\nStatistics for {selected_pert}:")
    print(f"Number of cells in ground truth: {(adata_second_half.obs['condition'] == selected_pert).sum()}")
    print(f"Number of DEGs: {pert_degs_idx.sum()}")
    print(f"MSE (all genes): {mse_dict_predictive[f'{selected_pert}_gears']:.6f}")
    print(f"WMSE (weighted by DEGs): {wmse_dict_predictive[f'{selected_pert}_gears']:.6f}")
    print(f"R² (all genes): {r2_all:.6f}")
    print(f"R² (DEGs only): {r2_degs:.6f}")


# %%
selected_pert = np.random.choice(adata_second_half.obs['condition'].unique())
print(f"Selected perturbation: {selected_pert}")
second_half_mean = adata_second_half[adata_second_half.obs['condition'] == selected_pert].X.mean(axis=0).A1
first_half_mean = adata_first_half[adata_first_half.obs['condition'] == selected_pert].X.mean(axis=0).A1


# Get DEGs for this perturbation
pert_degs = list(set(adata.uns['deg_dict_vsrest'][selected_pert]['up']) | 
                    set(adata.uns['deg_dict_vsrest'][selected_pert]['down']))
pert_degs_idx = adata.var_names.isin(pert_degs)

# Get weights for this perturbation
current_pert_weights = pert_normalized_abs_scores_vsrest.get(selected_pert)

# Calculate deltas
delta_first_half = first_half_mean - total_mean_first_half
delta_second_half = second_half_mean - total_mean_first_half

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: All genes
ax1.scatter(delta_first_half, delta_second_half, alpha=0.3, s=1, color='gray', label='All genes')

# Add diagonal line
lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
        max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
ax1.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

# Calculate and display correlation and R2
corr_all = pearson(delta_second_half, delta_first_half)
r2_all = r2_score_on_deltas(delta_second_half, delta_first_half)
r2_all_weighted = r2_score_on_deltas(delta_second_half, delta_first_half, current_pert_weights)
ax1.set_ylabel('Second Half ($\Delta$ Expression)', fontsize=12)
ax1.set_xlabel('First Half ($\Delta$ Expression)', fontsize=12)
ax1.set_title(f'{selected_pert} - All Genes', fontsize=14)
ax1.grid(True, alpha=0.3)

# Add metrics as text in top left corner
ax1.text(0.05, 0.95, f'Pearson r = {corr_all:.3f}\nR² = {r2_all:.3f}\nR² weighted = {r2_all_weighted:.3f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10)

# Plot 2: DEGs only
ax2.scatter(delta_first_half[pert_degs_idx], delta_second_half[pert_degs_idx], 
            alpha=0.6, s=10, color='darkred', label='DEGs')

# Add diagonal line
lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
        max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
ax2.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

# Calculate and display correlation and R2 for DEGs
corr_degs = pearson(delta_second_half[pert_degs_idx], delta_first_half[pert_degs_idx])
r2_degs = r2_score_on_deltas(delta_second_half[pert_degs_idx], delta_first_half[pert_degs_idx])
r2_degs_weighted = r2_score_on_deltas(delta_second_half[pert_degs_idx], delta_first_half[pert_degs_idx], 
                    current_pert_weights[pert_degs_idx])
ax2.set_ylabel('Second Half ($\Delta$ Expression)', fontsize=12)
ax2.set_xlabel('First Half ($\Delta$ Expression)', fontsize=12)
ax2.set_title(f'{selected_pert} - DEGs Only ({pert_degs_idx.sum()} genes)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add metrics as text in top left corner
ax2.text(0.05, 0.95, f'Pearson r = {corr_degs:.3f}\nR² = {r2_degs:.3f}\nR² weighted = {r2_degs_weighted:.3f}',
         transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10)

plt.tight_layout()

plt.show()

# Print additional statistics
print(f"\nStatistics for {selected_pert}:")
print(f"Number of cells in ground truth: {(adata_second_half.obs['condition'] == selected_pert).sum()}")
print(f"Number of DEGs: {pert_degs_idx.sum()}")
if f'{selected_pert}_gears' in mse_dict_predictive:
    print(f"MSE (all genes): {mse_dict_predictive[f'{selected_pert}_gears']:.6f}")
    print(f"WMSE (weighted by DEGs): {wmse_dict_predictive[f'{selected_pert}_gears']:.6f}")
print(f"R² (all genes): {r2_all:.6f}")
print(f"R² (DEGs only): {r2_degs:.6f}")

# %%
# NEW SECTION: Analysis with different DEG quantile ranges
# Create grouped boxplots showing impact across DEG quantile ranges

print("\n=== Running analysis with DEG quantile ranges ===")

# First, calculate the number of DEGs for each perturbation
deg_counts = {}
for pert in all_perts_for_predictive:
    pert_degs_vsrest = list(set(adata.uns['deg_dict_vsrest'][pert]['up']) | set(adata.uns['deg_dict_vsrest'][pert]['down']))
    deg_counts[pert] = len(pert_degs_vsrest)

# Sort perturbations by DEG count and divide into quantiles
sorted_perts = sorted(deg_counts.keys(), key=lambda x: deg_counts[x])
n_perts = len(sorted_perts)

# Define custom quantile ranges
quantile_labels = ['0-50%', '50-75%', '75-88%', '88-95%', '95-100%']
quantile_boundaries = [0.0, 0.50, 0.75, 0.88, 0.95, 1.0]

# Assign perturbations to quantiles based on percentile boundaries
pert_to_quantile = {}
for i in range(len(quantile_labels)):
    start_idx = int(quantile_boundaries[i] * n_perts)
    end_idx = int(quantile_boundaries[i + 1] * n_perts) if i < len(quantile_labels) - 1 else n_perts
    
    for j in range(start_idx, end_idx):
        pert_to_quantile[sorted_perts[j]] = i

print(f"Quantile distribution:")
for i, label in enumerate(quantile_labels):
    perts_in_quantile = [p for p, q in pert_to_quantile.items() if q == i]
    deg_range = f"{min(deg_counts[p] for p in perts_in_quantile)}-{max(deg_counts[p] for p in perts_in_quantile)}"
    print(f"  {label}: {len(perts_in_quantile)} perturbations, DEG range: {deg_range}")

# Store results for each quantile
quantile_results = []

# Process each quantile
for quantile_idx, quantile_label in enumerate(quantile_labels):
    print(f"\nProcessing quantile: {quantile_label}")
    
    # Get perturbations in this quantile
    perts_in_quantile = [p for p, q in pert_to_quantile.items() if q == quantile_idx]
    
    # Create dictionaries to store the metrics for this quantile
    wmse_dict_quantile = {}
    pearson_delta_degs_dict_quantile = {}
    wr2_delta_dict_quantile = {}
    
    # Process each perturbation in this quantile
    for pert in tqdm(perts_in_quantile, desc=f"Processing {quantile_label}"):
        # Get means
        first_half_mean = adata_first_half[adata_first_half.obs['condition'] == pert].X.mean(axis=0).A1
        second_half_mean = adata_second_half[adata_second_half.obs['condition'] == pert].X.mean(axis=0).A1
        
        # Get DEG info
        current_pert_weights = pert_normalized_abs_scores_vsrest.get(pert)
        pert_degs_vsrest = list(set(adata.uns['deg_dict_vsrest'][pert]['up']) | set(adata.uns['deg_dict_vsrest'][pert]['down']))
        pert_degs_vsrest_idx = adata.var_names.isin(pert_degs_vsrest)
        
        # Tech Duplicate metrics
        if pert_degs_vsrest_idx.sum() > 0:  # As long as there are some DEGs
            wmse_dict_quantile[pert] = wmse(first_half_mean, second_half_mean, current_pert_weights)
            delta_first_half = first_half_mean - total_mean_first_half
            delta_second_half = second_half_mean - total_mean_first_half
            pearson_delta_degs_dict_quantile[pert] = pearson(delta_first_half[pert_degs_vsrest_idx], delta_second_half[pert_degs_vsrest_idx])
            wr2_delta_dict_quantile[pert] = r2_score_on_deltas(delta_second_half, delta_first_half, current_pert_weights)
        else:
            wmse_dict_quantile[pert] = np.nan
            pearson_delta_degs_dict_quantile[pert] = np.nan
            wr2_delta_dict_quantile[pert] = np.nan
        
        # Control baseline
        control_key = f"{pert}_control"
        if pert_degs_vsrest_idx.sum() > 0:
            wmse_dict_quantile[control_key] = wmse(ctrl_mean_original, second_half_mean, current_pert_weights)
            delta_control = ctrl_mean_original - total_mean_first_half
            delta_second_half = second_half_mean - total_mean_first_half
            pearson_delta_degs_dict_quantile[control_key] = pearson(delta_control[pert_degs_vsrest_idx], delta_second_half[pert_degs_vsrest_idx])
            wr2_delta_dict_quantile[control_key] = r2_score_on_deltas(delta_second_half, delta_control, current_pert_weights)
        else:
            wmse_dict_quantile[control_key] = np.nan
            pearson_delta_degs_dict_quantile[control_key] = np.nan
            wr2_delta_dict_quantile[control_key] = np.nan
        
        # Data mean baseline
        datamean_key = f"{pert}_datamean"
        if pert_degs_vsrest_idx.sum() > 0:
            wmse_dict_quantile[datamean_key] = wmse(total_mean_original, second_half_mean, current_pert_weights)
            pearson_delta_degs_dict_quantile[datamean_key] = np.nan  # Changed from 0.0 to np.nan
            delta_data_mean = total_mean_first_half - total_mean_first_half
            delta_second_half = second_half_mean - total_mean_first_half
            wr2_delta_dict_quantile[datamean_key] = r2_score_on_deltas(delta_second_half, delta_data_mean, current_pert_weights)
        else:
            wmse_dict_quantile[datamean_key] = np.nan
            pearson_delta_degs_dict_quantile[datamean_key] = np.nan
            wr2_delta_dict_quantile[datamean_key] = np.nan
        
        # GEARS
        gears_key = f"{pert}_gears"
        gears_mean = gears_predictions.get(pert)
        if pert_degs_vsrest_idx.sum() > 0:
            wmse_dict_quantile[gears_key] = wmse(gears_mean, second_half_mean, current_pert_weights)
            delta_gears = gears_mean - total_mean_first_half
            delta_second_half = second_half_mean - total_mean_first_half
            pearson_delta_degs_dict_quantile[gears_key] = pearson(delta_gears[pert_degs_vsrest_idx], delta_second_half[pert_degs_vsrest_idx])
            wr2_delta_dict_quantile[gears_key] = r2_score_on_deltas(delta_second_half, delta_gears, current_pert_weights)
        else:
            wmse_dict_quantile[gears_key] = np.nan
            pearson_delta_degs_dict_quantile[gears_key] = np.nan
            wr2_delta_dict_quantile[gears_key] = np.nan
        
        # scGPT
        scgpt_key = f"{pert}_scgpt"
        scgpt_mean = scgpt_predictions.get(pert)
        if pert_degs_vsrest_idx.sum() > 0:
            wmse_dict_quantile[scgpt_key] = wmse(scgpt_mean, second_half_mean, current_pert_weights)
            delta_control = second_half_mean - total_mean_original
            delta_scgpt = scgpt_mean - total_mean_original
            delta_second_half = second_half_mean - total_mean_first_half
            pearson_delta_degs_dict_quantile[scgpt_key] = pearson(delta_scgpt[pert_degs_vsrest_idx], delta_second_half[pert_degs_vsrest_idx])
            wr2_delta_dict_quantile[scgpt_key] = r2_score_on_deltas(delta_second_half, delta_scgpt, current_pert_weights)
        else:
            wmse_dict_quantile[scgpt_key] = np.nan
            pearson_delta_degs_dict_quantile[scgpt_key] = np.nan
            wr2_delta_dict_quantile[scgpt_key] = np.nan
    
    # Store results for this quantile
    for pert in perts_in_quantile:
        base_pert = pert
        control_key = f"{base_pert}_control"
        datamean_key = f"{base_pert}_datamean"
        gears_key = f"{base_pert}_gears"
        scgpt_key = f"{base_pert}_scgpt"
        
        # Only include perturbations that have all the necessary keys (same as original analysis)
        if control_key in wmse_dict_quantile and datamean_key in wmse_dict_quantile and gears_key in wmse_dict_quantile:
            # WMSE
            for condition, key_name in [
                ('Tech Duplicate', base_pert),
                ('$\mu^c$ (ctrl mean)', control_key),
                ('$\mu^{all}$ (perts mean)', datamean_key),
                ('GEARS', gears_key),
                ('scGPT', scgpt_key)
            ]:
                if key_name in wmse_dict_quantile:
                    quantile_results.append({
                        'Perturbation': base_pert,
                        'Metric': 'WMSE',
                        'Condition': condition,
                        'Quantile': quantile_label,
                        'QuantileIdx': quantile_idx,
                        'Value': wmse_dict_quantile[key_name]
                    })
            
            # Pearson Delta DEGs
            for condition, key_name in [
                ('Tech Duplicate', base_pert),
                ('$\mu^c$ (ctrl mean)', control_key),
                ('$\mu^{all}$ (perts mean)', datamean_key),
                ('GEARS', gears_key),
                ('scGPT', scgpt_key)
            ]:
                if key_name in pearson_delta_degs_dict_quantile:
                    quantile_results.append({
                        'Perturbation': base_pert,
                        'Metric': 'Pearson Delta DEGs',
                        'Condition': condition,
                        'Quantile': quantile_label,
                        'QuantileIdx': quantile_idx,
                        'Value': pearson_delta_degs_dict_quantile[key_name]
                    })
            
            # Weighted R-Squared Delta
            for condition, key_name in [
                ('Tech Duplicate', base_pert),
                ('$\mu^c$ (ctrl mean)', control_key),
                ('$\mu^{all}$ (perts mean)', datamean_key),
                ('GEARS', gears_key),
                ('scGPT', scgpt_key)
            ]:
                if key_name in wr2_delta_dict_quantile:
                    quantile_results.append({
                        'Perturbation': base_pert,
                        'Metric': 'Weighted R-Squared Delta',
                        'Condition': condition,
                        'Quantile': quantile_label,
                        'QuantileIdx': quantile_idx,
                        'Value': wr2_delta_dict_quantile[key_name]
                    })

# Create DataFrame for quantile results
df_quantile_results = pd.DataFrame(quantile_results)

# %%
# Function to create grouped boxplots showing impact of DEG quantiles
def plot_deg_quantile_impact(df, metric_name, y_label, plot_title, plot_dir, dataset_name):
    # Filter for just this metric
    df_metric = df[df['Metric'] == metric_name].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define colors for conditions - same as in plot_predictive_conditions_boxplot
    condition_colors = {
        'Tech Duplicate': 'steelblue',
        '$\mu^c$ (ctrl mean)': 'forestgreen',
        '$\mu^{all}$ (perts mean)': 'indianred',
        'GEARS': 'purple',
        # 'scGPT': 'orange'
    }
    
    # Define the condition order
    condition_order = [
        '$\mu^c$ (ctrl mean)', 
        '$\mu^{all}$ (perts mean)', 
        'GEARS', 
        # 'scGPT', 
        'Tech Duplicate'
    ]
    
    # Get quantile labels
    quantile_labels = sorted(df_metric['Quantile'].unique(), key=lambda x: df_metric[df_metric['Quantile'] == x]['QuantileIdx'].iloc[0])
    n_quantiles = len(quantile_labels)
    n_conditions = len(condition_order)
    width = 0.15  # Width for boxplots
    
    # Create a new column for positioning
    df_metric['x_position'] = 0
    
    # Assign x positions
    for i, quantile in enumerate(quantile_labels):
        for j, condition in enumerate(condition_order):
            mask = (df_metric['Quantile'] == quantile) & (df_metric['Condition'] == condition)
            df_metric.loc[mask, 'x_position'] = i + (j - n_conditions/2 + 0.5) * width * 1.2
    
    # Create narrow boxplots
    for i, quantile in enumerate(quantile_labels):
        for j, condition in enumerate(condition_order):
            data = df_metric[(df_metric['Quantile'] == quantile) & (df_metric['Condition'] == condition)]
            if len(data) > 0:
                x_pos = i + (j - n_conditions/2 + 0.5) * width * 1.2
                # Create boxplot for this subset
                box_parts = ax.boxplot(
                    data['Value'].dropna(), 
                    positions=[x_pos], 
                    widths=width * 0.8,  # Make boxplots narrower
                    patch_artist=True,  # Enable filled boxes
                    showfliers=False,  # Don't show outliers as we'll add all points
                    notch=False,
                    showmeans=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=condition_colors[condition], alpha=0.6, edgecolor='black', linewidth=0.5),
                    whiskerprops=dict(color='black', linewidth=1.0),
                    capprops=dict(color='black', linewidth=1.0)
                )
    
    # Add strip plot for individual points
    for i, quantile in enumerate(quantile_labels):
        for j, condition in enumerate(condition_order):
            data = df_metric[(df_metric['Quantile'] == quantile) & (df_metric['Condition'] == condition)]
            if len(data) > 0:
                x_pos = i + (j - n_conditions/2 + 0.5) * width * 1.2
                # Add jittered points
                jitter = (np.random.random(len(data)) - 0.5) * width * 0.5
                ax.scatter(x_pos + jitter, data['Value'], 
                          color='black', s=3, alpha=0.06, zorder=10)
    
    # Get the y-axis range for positioning median labels
    all_values = df_metric['Value'].dropna()
    y_min_data = all_values.min()
    y_max_data = all_values.max()
    y_range = y_max_data - y_min_data
    
    # Add median values for each condition-quantile combination
    INCREASE_MEDIAN_LABEL_BY = 0.15 if metric_name == 'WMSE' else 0.02
    for i, quantile in enumerate(quantile_labels):
        for j, condition in enumerate(condition_order):
            data = df_metric[(df_metric['Quantile'] == quantile) & (df_metric['Condition'] == condition)]['Value'].dropna()
            if len(data) > 0:
                median_val = data.median()
                if not np.isnan(median_val):
                    x_pos = i + (j - n_conditions/2 + 0.5) * width * 1.2
                    # Position text higher up in the range of y data values
                    yloc = median_val + INCREASE_MEDIAN_LABEL_BY * y_range
                    # Use abbreviated condition names for space
                    condition_abbrev = {
                        'Tech Duplicate': 'TD',
                        '$\mu^c$ (ctrl mean)': 'μᶜ',
                        '$\mu^{all}$ (perts mean)': 'μᵃˡˡ',
                        'GEARS': 'GE',
                        # 'scGPT': 'sG'
                    } 
                    ax.text(x_pos, yloc, f'{median_val:.2f}',
                           color='black', fontweight='bold', ha='center', va='center',
                           fontsize=8, rotation=0)
    
    # Add a horizontal line at y=0 for certain metrics
    if metric_name in ['Pearson Delta DEGs', 'Weighted R-Squared Delta']:
        ax.axhline(y=0, color='firebrick', linestyle='--', linewidth=0.8, zorder=20, alpha=0.7)
        ax.set_ylim(-1.05, 1.05)
    
    # Customize the plot
    ax.set_xticks(range(n_quantiles))
    ax.set_xticklabels(quantile_labels, rotation=0, fontsize=14)
    ax.set_xlabel('DEG Count Quantile', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(f'{plot_title} - By DEG Count Quantile ({dataset_name})', fontsize=18, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=14)
    
    # Create custom legend for conditions
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=condition_colors[c], alpha=0.6, label=c) 
                      for c in condition_order]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              title='Condition', frameon=True, fancybox=True, fontsize=12, title_fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    filename = f"deg_quantile_impact_{metric_name.lower().replace(' ', '_')}.pdf"
    plot_path = f"{plot_dir}/{filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()
    plt.close()

# %%
# Create the grouped boxplots for each metric affected by DEG quantiles
print("\n=== Creating DEG quantile impact plots ===")

plot_deg_quantile_impact(df_quantile_results, 'WMSE', 
                         r'WMSE (vs Second Half)', 
                         r'WMSE', PLOT_DIR, DATASET_NAME)

plot_deg_quantile_impact(df_quantile_results, 'Pearson Delta DEGs', 
                         r'Pearson($\Delta$) DEGs ($\mu^{all}$ as $\Delta_{ctrl}$)', 
                         'Pearson($\Delta$) DEGs', PLOT_DIR, DATASET_NAME)

plot_deg_quantile_impact(df_quantile_results, 'Weighted R-Squared Delta', 
                         r'$R^2_w(\Delta)$ ($\mu^{all}$ as $\Delta_{ctrl}$)', 
                         r'$R^2_w(\Delta)$', PLOT_DIR, DATASET_NAME)





