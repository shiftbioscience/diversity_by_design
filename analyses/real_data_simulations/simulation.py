#### 0. Preliminary: Set up the metrics MAE, MSE, Pearson Delta.
#### 1. Np: Cells per perturbation. Downsample data to get a range of cells per perturbation and then calculate the metrics for each.
#### 2. N0: Control cells number. Downsample the control cells to get a distribution of numbers of cells and then calculate the metrics for each.
#### 3. k: Number of perturbations. Calculate the metrics for a range of numbers of perturbations and then calculate the metrics for each.
#### 4. d: Number of DEGs. Calculate metrics for perturbations with a small number of DEGs and a large number of DEGs.
#### 5. E: Effect size of DEGs. Calculate metrics for perturbations with a small effect size and a large effect size (Based on Log2FC)
#### 6. B: Control bias. We will interpolate between the total mean and the control mean and then outward past the control mean in steps of a fixed size to get a sense of the bias of the control mean.
#### 7. g: Number of genes. Calculate metrics for greater or fewer genes included in the analysis.
#### 8. mu_l: Effect of library size. Calculate library size for all cells and then test with quantiles of the library size.

import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
import pickle
import sys
sys.path.append(os.path.dirname(os.getcwd())) # For finding the 'analyses' package
from common import *

   

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['norman19', 'replogle22'])
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
    names_df_vsrest
) = initialize_analysis(DATASET_NAME, 'real_data_simulations_results')

SCORE_TYPE = 'scores' # or 'logfoldchanges'
names_df_vsctrl = pd.read_pickle(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_names_df_vsctrl.pkl')
scores_df_vsctrl = pd.read_pickle(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_scores_df_vsctrl.pkl')

#### 1. n_p: Cells per perturbation. Downsample data to get a range of cells per perturbation and then calculate the metrics for each.

# Get the selected cells and precalculated weights
import pickle
with open(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells.pkl', 'rb') as f:
    pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells = pickle.load(f)
with open(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_pert_normalized_abs_scores_vsrest_cells_per_pert.pkl', 'rb') as f:
    pert_normalized_abs_scores_vsrest_cells_per_pert = pickle.load(f)

# Get the perts with at least N cells
max_cells_per_pert = max(DATASET_CELL_COUNTS)
pert_counts = adata.obs['condition'].value_counts()
pert_counts = pert_counts[(pert_counts >= max_cells_per_pert) & (pert_counts.index != 'control')]

cell_counts = DATASET_CELL_COUNTS

# Define metric names
metric_names = ['corr_delta', 'mae', 'mse']

# Initialize all metric dictionaries in a structured way
all_metric_dicts = {}
for name in metric_names:
    all_metric_dicts[f'{name}_dict'] = {}

# Get the total mean for each number of cells per perturbation
total_mean_per_pert_cellcount = {}
for cell_count in tqdm(cell_counts):
    adata_n_cells = adata[pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells[cell_count]]
    total_mean_per_pert_cellcount[cell_count] = {}
    for pert in pert_counts.index:
        pert_mean = adata_n_cells[adata_n_cells.obs['condition'] == pert].X.mean(axis=0).A1
        total_mean_per_pert_cellcount[cell_count][pert] = pert_mean
    # Get the mean of the total means
    total_mean_per_pert_cellcount[cell_count]['total_mean'] = np.mean(list(total_mean_per_pert_cellcount[cell_count].values()), axis=0)

for pert in tqdm(pert_counts.index):
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    # Initialize per-perturbation dictionaries
    for dict_key_init in all_metric_dicts:
        all_metric_dicts[dict_key_init][pert] = {}

    for cell_count in cell_counts:
        pert_mean = total_mean_per_pert_cellcount[cell_count][pert]
        total_mean = total_mean_per_pert_cellcount[cell_count]['total_mean']
        
        delta_all_vs_control = total_mean - ctrl_mean_original
        delta_pert_vs_control = pert_mean - ctrl_mean_original

        # Gene index is always all genes now
        gene_idx = slice(None)

        # Apply slicing for the current variant (all genes, DEGs vs ctrl, or DEGs vs rest)
        pert_mean_s = pert_mean[gene_idx]
        total_mean_s = total_mean[gene_idx]
        delta_pert_vs_control_s = delta_pert_vs_control[gene_idx]
        delta_all_vs_control_s = delta_all_vs_control[gene_idx]

            # Metrics
        all_metric_dicts['mae_dict'][pert][cell_count] = mae(pert_mean_s, total_mean_s)
        all_metric_dicts['mse_dict'][pert][cell_count] = mse(pert_mean_s, total_mean_s)
        all_metric_dicts['corr_delta_dict'][pert][cell_count] = pearson(delta_all_vs_control_s, delta_pert_vs_control_s)


# Plot metrics
PLOT_DIR = f'{ANALYSIS_DIR}/np_effect_of_pert_cell_number'

metrics_to_plot = [
    {'name': 'mae', 'title_name': 'MAE', 'ylabel': 'MAE($\mu_p$,$\mu_{all}$)'},
    {'name': 'mse', 'title_name': 'MSE', 'ylabel': 'MSE($\mu_p$,$\mu_{all}$)'},
    {'name': 'corr_delta', 'title_name': 'Pearson($\Delta$)', 'ylabel': 'Pearson($\Delta_{pert}$,$\Delta_{all}$)'},
]

np_aggregate_vals = {}
for metric_info in metrics_to_plot:
    metric_dict_key = f"{metric_info['name']}_dict"
    # Ensure the dictionary key exists and the dictionary is not empty before plotting
    if metric_dict_key in all_metric_dicts and all_metric_dicts[metric_dict_key]:
        title = f"{metric_info['title_name']} by $n_p$"
        np_aggregate_vals[metric_dict_key] = get_aggregate_correlation_from_dict(all_metric_dicts[metric_dict_key], log_x=True, log_x_base=2)

        plot_metrics_as_scatter_trend(
            all_metric_dicts[metric_dict_key],
            PLOT_DIR,
            title,
            DATASET_NAME,
            use_log_x=True,
            log_x_base=2,
            xlabel='$n_p$ (# cells per perturbation)',
            ylabel=metric_info['ylabel']
        )

# Save the aggregate values to pickle in the ANALYSIS_DIR
with open(f'{ANALYSIS_DIR}/np_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(np_aggregate_vals, f)


#### 2. n_0: Control cells number. Downsample the control cells to get a distribution of numbers of cells and then calculate the metrics for each.
cell_counts_n0 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
ctrl_cells = adata.obs[adata.obs['condition'] == 'control'].index.tolist()

# Initialize a new dictionary for n0 simulation results
n0_all_metric_dicts = {}
metric_names_for_n0 = ['corr_delta'] 

for name in metric_names_for_n0:
    n0_all_metric_dicts[f'{name}_dict'] = {}

for cell_count_val in tqdm(cell_counts_n0):
    num_to_sample = min(cell_count_val, len(ctrl_cells))
    if num_to_sample == 0:
        if len(ctrl_cells) == 0:
             ctrl_mean_ds = np.zeros_like(total_mean_original) # Match shape of total_mean_original
        else: # num_to_sample must be > 0 if len(ctrl_cells) > 0
             sampled_control_cells = np.random.choice(ctrl_cells, size=num_to_sample, replace=False)
             ctrl_mean_ds = adata[sampled_control_cells].X.mean(axis=0).A1
    else: # num_to_sample > 0
        sampled_control_cells = np.random.choice(ctrl_cells, size=num_to_sample, replace=False)
        ctrl_mean_ds = adata[sampled_control_cells].X.mean(axis=0).A1


    for pert in pert_counts.index:
        pert_mean = pert_means[pert] # Full mean for the specific perturbation

        # Initialize per-perturbation sub-dictionaries if not already present
        for metric_key_base in metric_names_for_n0:
            dict_key = f'{metric_key_base}_dict'
            if pert not in n0_all_metric_dicts[dict_key]:
                n0_all_metric_dicts[dict_key][pert] = {}
        
        delta_pert_vs_control_ds = pert_mean - ctrl_mean_ds
        delta_total_original_vs_control_ds = total_mean_original - ctrl_mean_ds 

        # Gene index is always all genes now
        gene_idx = slice(None)

        delta_total_original_vs_control_ds_s = delta_total_original_vs_control_ds[gene_idx]
        delta_pert_vs_control_ds_s = delta_pert_vs_control_ds[gene_idx]
        
        current_metric_dict_key = 'corr_delta_dict' # Only one metric type now
        n0_all_metric_dicts[current_metric_dict_key][pert][cell_count_val] = pearson(
            delta_total_original_vs_control_ds_s, delta_pert_vs_control_ds_s
        )

PLOT_DIR_N0 = f'{ANALYSIS_DIR}/n_0_effect_of_control_cell_number' # As per user's new code

n0_plot_configs = [
    {'name': 'corr_delta', 'title_name': 'Pearson($\Delta$)', 'ylabel': 'Pearson($\Delta_{pert}$,$\Delta_{all}$)'},
]

n0_aggregate_vals= {}
for metric_info in n0_plot_configs:
    metric_dict_key = f"{metric_info['name']}_dict"
    # Ensure the dictionary key exists and the dictionary is not empty before plotting
    if metric_dict_key in n0_all_metric_dicts and n0_all_metric_dicts[metric_dict_key]:
        title = f"{metric_info['title_name']} by $n_0$"
        n0_aggregate_vals[metric_dict_key] = get_aggregate_correlation_from_dict(
            n0_all_metric_dicts[metric_dict_key], log_x=True, log_x_base=2
        )
        
        plot_metrics_as_scatter_trend(
            n0_all_metric_dicts[metric_dict_key],
            PLOT_DIR_N0,
            title,
            DATASET_NAME,
            use_log_x=True,
            log_x_base=2,
            xlabel='$n_0$ (# control cells)', 
            ylabel=metric_info['ylabel']
        )

with open(f'{ANALYSIS_DIR}/n0_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(n0_aggregate_vals, f)



#### 3. k: Number of perturbations

# Define 10 random seeds for this step
random_seeds_step3 = range(10)
perts_to_sweep = DATASET_PERTS_TO_SWEEP # Original sweep range

# Lists to store metrics from all seeds
collected_metrics_p_all = []
collected_metrics_p_vsrest = []

# Outer loop for seeds
for seed_val in tqdm(random_seeds_step3, desc="Step 3 Seeds"):
    np.random.seed(seed_val) # Set seed for this iteration

    # Inner loop for number of perturbations
    for n_perts in tqdm(perts_to_sweep, desc=f"  N Perts (Seed {seed_val})", leave=False):
        # Ensure we don't try to sample more perts than available
        if n_perts > len(pert_counts.index):
            print(f"Warning: n_perts ({n_perts}) > available perts ({len(pert_counts.index)}). Skipping for seed {seed_val}, n_perts {n_perts}.")
            continue
            
        # Randomly sample n_perts from the eligible perturbations (pert_counts.index)
        sampled_perts_in_iteration = np.random.choice(pert_counts.index, size=n_perts, replace=False)
        
        # Get the total mean expression vector using all cells from ONLY the sampled_perts_in_iteration
        total_mean_for_n_perts_sample = adata[adata.obs['condition'].isin(sampled_perts_in_iteration)].X.mean(axis=0).A1
        # Calculate the delta of this sampled total mean vs the original control mean
        delta_all_vs_control_for_n_perts_sample = total_mean_for_n_perts_sample - ctrl_mean_original

        # For each specific perturbation (pert_j) WITHIN the current sample of n_perts:
        for pert_j in sampled_perts_in_iteration:
            # Get the pre-calculated mean expression for this specific pert_j (using all its cells)
            pert_j_mean = pert_means[pert_j] 
            # Calculate the delta of this specific pert_j's mean vs the original control mean
            delta_pert_j_vs_control = pert_j_mean - ctrl_mean_original
            
            # Metric 1: Original (vs. ctrl_mean_original)
            pearson_val_all_vs_ctrl = pearson(delta_all_vs_control_for_n_perts_sample, delta_pert_j_vs_control)
            collected_metrics_p_all.append({
                'seed': seed_val,
                'n_perts': n_perts,
                'pert': pert_j, 
                'metric_value': pearson_val_all_vs_ctrl 
            })

            # DEG-specific metrics for pert_j
            pert_degs_vsctrl_list = list(set(adata.uns['deg_dict_vscontrol'][pert_j]['up']) | set(adata.uns['deg_dict_vscontrol'][pert_j]['down']))
            pert_degs_vsctrl_idx = adata.var_names.isin(pert_degs_vsctrl_list)
            pert_degs_vsrest_list = list(set(adata.uns['deg_dict_vsrest'][pert_j]['up']) | set(adata.uns['deg_dict_vsrest'][pert_j]['down']))
            pert_degs_vsrest_idx = adata.var_names.isin(pert_degs_vsrest_list)

            val_vsrest_orig = pearson(delta_all_vs_control_for_n_perts_sample[pert_degs_vsrest_idx], delta_pert_j_vs_control[pert_degs_vsrest_idx])
            collected_metrics_p_vsrest.append({'seed': seed_val, 'n_perts': n_perts, 'pert': pert_j, 'metric_value': val_vsrest_orig})


# Convert lists of dictionaries to DataFrames
df_p_all = pd.DataFrame(collected_metrics_p_all)
df_p_vsrest = pd.DataFrame(collected_metrics_p_vsrest)

# Get the aggregate values for the metrics
df_p_all['n_perts_log2'] = np.log2(df_p_all['n_perts'])
P_aggregate_vals = {'corr_delta_dict': pearsonr(df_p_all['n_perts_log2'], df_p_all['metric_value'])[0]}

# Define plot directory for this step
PLOT_DIR_STEP3 = f'{ANALYSIS_DIR}/k_effect_of_n_perts'
os.makedirs(PLOT_DIR_STEP3, exist_ok=True) # Ensure directory exists

# # Call the new multi-seed plotting function for each metric type
# plot_n_perts_categorical_scatter_multiseed(
#     df_p_all, 
#     PLOT_DIR_STEP3, 
#     'Pearson delta by # perts (in all perts vs other perts)', 
#     DATASET_NAME, 
#     xlabel='# Perturbations', 
#     ylabel='Pearson R'
# )
plot_n_perts_categorical_scatter_multiseed(
    df_p_vsrest, 
    PLOT_DIR_STEP3, 
    'Pearson($\Delta$) by $k$ (in DEGs vs other perts)', 
    DATASET_NAME, 
    xlabel='$k$ (# of Perturbations)', 
    ylabel='Pearson($\Delta^{all}$,$\Delta^p$)'
)


# Reset to the original random seed state after Step 3
np.random.set_state(original_np_random_state)

# Save the aggregate values to pickle in the ANALYSIS_DIR
with open(f'{ANALYSIS_DIR}/k_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(P_aggregate_vals, f)


#### 4. d: Number of DEGs

SCORE_TYPE = 'scores' # or 'logfoldchanges'
names_df_vsctrl = pd.read_pickle(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_names_df_vsctrl.pkl')
scores_df_vsctrl = pd.read_pickle(f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_scores_df_vsctrl.pkl')


# Get the number of DEGs for each pert. Get this as a dataframe
degs_per_pert_vsctrl = {pert: len(adata.uns[f'deg_dict_vscontrol'][pert]['down']) + len(adata.uns[f'deg_dict_vscontrol'][pert]['up']) for pert in pert_counts.index}
degs_per_pert_df_vsctrl = pd.DataFrame(list(degs_per_pert_vsctrl.items()), columns=['pert', 'n_degs'])

# Remove the control from the dataframe
degs_per_pert_df_vsctrl = degs_per_pert_df_vsctrl[degs_per_pert_df_vsctrl['pert'] != 'control']

# Sweep across 20% quantiles tiled so that half overlap each time. E.g., 0-20%, 10-30%, 20-40%, etc.
degs_per_pert_df_vsctrl['n_degs_ranked'] = degs_per_pert_df_vsctrl['n_degs'].rank(pct=True)

corr_delta_dict = {}
mae_dict = {}
mse_dict = {}

ctrl_mean_original_center = total_mean_original

for i in tqdm(range(9)):
    lower_bound = i * .1
    upper_bound = lower_bound + .20
    # Round upper to  one decimal place
    upper_bound = round(upper_bound, 1)
    # Round lower to  one decimal place
    lower_bound = round(lower_bound, 1)
    pert_subset = degs_per_pert_df_vsctrl[degs_per_pert_df_vsctrl['n_degs_ranked'] >= lower_bound]
    pert_subset = pert_subset[pert_subset['n_degs_ranked'] < upper_bound]

    quantile_range_key = f'{lower_bound}-{upper_bound}'
    # Initialize the dicts for these pert subsets
    corr_delta_dict[quantile_range_key] = {}
    mae_dict[quantile_range_key] = {}
    mse_dict[quantile_range_key] = {}


    total_mean_subset = np.array([pert_means[pert] for pert in pert_subset['pert']]).mean(axis=0)
    delta_all_vs_control = total_mean_subset - ctrl_mean_original_center

    # For each pert in the subset, calculate the metrics
    for pert in pert_subset['pert']:
        pert_mean = pert_means[pert]
        delta_pert_vs_control = pert_mean - ctrl_mean_original_center
            
        current_pearson_result = pearson(delta_all_vs_control, delta_pert_vs_control)
        corr_delta_dict[quantile_range_key][pert] = current_pearson_result

        mae_dict[quantile_range_key][pert] = mae(pert_mean, total_mean_subset)
        mse_dict[quantile_range_key][pert] = mse(pert_mean, total_mean_subset)
metric_dicts = {
    'corr_delta_dict': corr_delta_dict,
    'mae_dict': mae_dict,
    'mse_dict': mse_dict
}

# Plot the metrics as a scatter plot
PLOT_DIR = f'{ANALYSIS_DIR}/d_effect_of_n_degs'
plot_pert_strength_scatter(corr_delta_dict, PLOT_DIR, 'Pearson delta perturbation vs all', DATASET_NAME, xlabel='DEG quantile range', ylabel='Pearson R')
plot_pert_strength_scatter(mae_dict, PLOT_DIR, 'MAE', DATASET_NAME, xlabel='DEG quantile range', ylabel='MAE')
plot_pert_strength_scatter(mse_dict, PLOT_DIR, 'MSE', DATASET_NAME, xlabel='DEG quantile range', ylabel='MSE') # Replaced by combined plot

# Get the aggregate values for the metrics
d_aggregate_vals = {}
for metric_name in ['corr_delta', 'mae', 'mse']:
    metric_dict_key = f'{metric_name}_dict'
    d_aggregate_vals[metric_dict_key] = get_aggregate_correlation_for_categorical_levels(metric_dicts[metric_dict_key])
    
with open(f'{ANALYSIS_DIR}/d_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(d_aggregate_vals, f)


#### 5. E: Effect size of DEGs

# Get the scores for each of the DEGs for each perturbation
deg_by_quantile_dict_vsctrl = {}
for pert in tqdm(adata.obs['condition'].unique()):
    if pert == 'control':
        continue
    # Get the gene names
    pert_degs_vsrest = names_df_vsrest[pert]
    pert_scores_vsrest = np.abs(scores_df_vsrest[pert])
    pert_degs_vsctrl = names_df_vsctrl[pert]
    pert_scores_vsctrl = np.abs(scores_df_vsctrl[pert])
    # Rank the scores
    pert_scores_ranked_vsrest = pert_scores_vsrest.rank(pct=True)
    pert_scores_ranked_vsctrl = pert_scores_vsctrl.rank(pct=True)
    quantiles = np.arange(0, 1, 0.1)
    deg_by_quantile_dict_vsctrl[pert] = {}
    # For each quantile, get the DEGs
    for i, quantile in enumerate(quantiles):
        # For the last quantile, set the next quantile to 1
        if i == len(quantiles) - 1:
            next_quantile = 1
        else:
            next_quantile = quantiles[i+1]
        pert_degs_quantile_vsrest = pert_degs_vsrest[(pert_scores_ranked_vsrest >= quantile) & (pert_scores_ranked_vsrest < next_quantile)]
        pert_degs_quantile_vsctrl = pert_degs_vsctrl[(pert_scores_ranked_vsctrl >= quantile) & (pert_scores_ranked_vsctrl < next_quantile)]
        quantile = round(quantile, 1)
        next_quantile = round(next_quantile, 1)
        deg_by_quantile_dict_vsctrl[pert][f"{quantile}-{next_quantile}"] = pert_degs_quantile_vsctrl.tolist()

# Sweep across the quantiles
# Get the delta for each pert
corr_delta_dict_vsctrl = {}

for pert, _ in tqdm(deg_by_quantile_dict_vsctrl.items()):
    corr_delta_dict_vsctrl[pert] = {}

    quantiles_vsctrl = deg_by_quantile_dict_vsctrl[pert]
    for quantile_range, _ in quantiles_vsctrl.items():
        degs_vsctrl = quantiles_vsctrl[quantile_range]
        degs_idx_vsctrl = adata.var_names.isin(degs_vsctrl)
        # Get calculate the pert mean wrt these degs
        pert_mean_degs_vsctrl = pert_means[pert][degs_idx_vsctrl]
        # Calculate the control mean wrt these degs
        ctrl_mean_degs_vsctrl = ctrl_mean_original[degs_idx_vsctrl]
        # total_mean_degs
        total_mean_degs_vsctrl = total_mean_original[degs_idx_vsctrl]
        # Calculate the delta
        delta_pert_vs_control_vsctrl = pert_mean_degs_vsctrl - ctrl_mean_degs_vsctrl
        # Calculate the delta_all_vs_control
        delta_all_vs_control_vsctrl = total_mean_degs_vsctrl - ctrl_mean_degs_vsctrl
        # Calculate the pearson correlation
        pearson_vsctrl = pearson(delta_all_vs_control_vsctrl, delta_pert_vs_control_vsctrl)
        corr_delta_dict_vsctrl[pert][quantile_range] = pearson_vsctrl

# Plot the metrics as a scatter plot
PLOT_DIR = f'{ANALYSIS_DIR}/E_effect_of_degs_effect_size'
plot_categorical_scatter_trend(corr_delta_dict_vsctrl, PLOT_DIR, 'Pearson delta perturbation vs all (DEGs vs ctrl)', DATASET_NAME, xlabel='DEG quantile range', ylabel='Pearson R')

# Get the aggregate values for the metrics
E_aggregate_vals = {}
E_aggregate_vals['corr_delta_dict'] = get_aggregate_correlation_from_dict_categorical(corr_delta_dict_vsctrl)
with open(f'{ANALYSIS_DIR}/E_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(E_aggregate_vals, f)


#### 6. B: Control bias
# We have the total and control mean already
delta_all_vs_control = ctrl_mean_original - total_mean_original

# Do interpolation in steps of 0.2
steps = np.arange(0, 2.1, 0.1)
# Get the interpolated means by multiplying the delta_all_vs_control by the steps and adding to the total mean
interpolated_means_ctrls = []
for step in steps:
    interpolated_means_ctrls.append(total_mean_original + step * delta_all_vs_control)


# Now we will calculate the pearson deltas with and without DEGs for these different values of interpolated controls
corr_delta_dict = {}
all_perts = adata.obs['condition'].unique()
all_perts = [pert for pert in all_perts if pert != 'control']

for pert in all_perts:

    # Get the mean of the pert
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    pert_mean = pert_means[pert]

    # Initialize the dicts for this pert
    corr_delta_dict[pert] = {}

    for i, step in enumerate(steps):
        current_interpolated_baseline = interpolated_means_ctrls[i]

        # --- Original Deltas (vs. Interpolated Control Baseline) ---
        delta_pert_vs_interp_ctrl = pert_mean - current_interpolated_baseline
        delta_total_original_vs_interp_ctrl = total_mean_original - current_interpolated_baseline
        step_key = round(step, 1) # Use rounded step as key consistently

        if i == 0: # current_interpolated_baseline == total_mean_original, so delta_interp_baseline_vs_total_mean is zero vector
            # Original metrics
            # delta_pert_vs_interp_ctrl is pert_mean - total_mean_original
            # delta_total_original_vs_interp_ctrl is zero vector
            # Pearson of (anything, zero_vector) is undefined/nan. R2 score would also be problematic or uninformative.
            corr_delta_dict[pert][step_key] = np.nan 

        else:
            # Original metrics
            corr_delta_dict[pert][step_key] = pearson(delta_total_original_vs_interp_ctrl, delta_pert_vs_interp_ctrl)

# Plot the metrics as a scatter plot
PLOT_DIR = f'{ANALYSIS_DIR}/B_effect_of_control_bias'
plot_metrics_as_scatter_trend(
    corr_delta_dict, PLOT_DIR, 
    'Pearson($\Delta$) by $\\beta$', DATASET_NAME, xlabel='Control Bias ($\\beta$)', 
    ylabel='Pearson($\Delta^p$,$\Delta^{all}$)', yaxis_limits=(-1.1, 1.1)
)

# Get the aggregate values for the metrics
B_aggregate_vals = {}
B_aggregate_vals['corr_delta_dict'] = get_aggregate_correlation_from_dict(corr_delta_dict)
with open(f'{ANALYSIS_DIR}/B_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(B_aggregate_vals, f)


# Select the top N HVGs
adata_hvg = adata[:, adata.var['dispersions_norm'].sort_values(ascending=False).index[:8192]]

# For every columns in scores_df and names_df, reorder the values based on the order of the values in adata.var_names
cols = []
for col in tqdm(scores_df_vsctrl.columns):
    scores_with_names = pd.Series(scores_df_vsctrl[col].values, index=names_df_vsctrl[col].values)
    reordered_series = scores_with_names.reindex(adata_hvg.var_names)
    # Add the reordered series to the list
    cols.append(reordered_series)
# Compile the list into a dataframe
scores_df_reordered = pd.concat(cols, axis=1)
scores_df_reordered.columns = scores_df_vsctrl.columns

PLOT_DIR = f'{ANALYSIS_DIR}/B_control_deg_bias_analysis'
os.makedirs(PLOT_DIR, exist_ok=True)

## 1. Get the number of DEGs between each single perturbation and all other perturbations
deg_dict_vscontrol = adata.uns['deg_dict_vscontrol']
# Get the distribution of the number of DEGs
deg_counts_up = [len(deg_dict_vscontrol[pert]['up']) for pert in deg_dict_vscontrol.keys()]
deg_counts_down = [len(deg_dict_vscontrol[pert]['down']) for pert in deg_dict_vscontrol.keys()]

## 2. Do a heatmap of all the DEGs vs control. It should be a binary matrix with 1 if the gene is a DEG and 0 otherwise
pert_names = list(deg_dict_vscontrol.keys())
gene_names = list(adata.var_names)
pert_to_idx = {pert: i for i, pert in enumerate(pert_names)}
gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}

num_perts = len(pert_names)
num_genes = len(gene_names)
deg_matrix_np = np.zeros((num_perts, num_genes), dtype=np.int8) # Use int8 for memory

for pert, deg_info in tqdm(deg_dict_vscontrol.items()):
    pert_idx = pert_to_idx[pert]
    up_genes = deg_info.get('up', [])
    down_genes = deg_info.get('down', [])

    # Get column indices for valid genes that exist in adata.var_names
    up_gene_indices = [gene_to_idx[g] for g in up_genes if g in gene_to_idx]
    down_gene_indices = [gene_to_idx[g] for g in down_genes if g in gene_to_idx]

    if up_gene_indices:
        deg_matrix_np[pert_idx, up_gene_indices] = 1
    if down_gene_indices:
        deg_matrix_np[pert_idx, down_gene_indices] = -1

# Convert back to DataFrame
deg_matrix_vscontrol_df = pd.DataFrame(deg_matrix_np, index=pert_names, columns=gene_names)
# Remove columns with all zeros
deg_matrix_vscontrol_df = deg_matrix_vscontrol_df.loc[:, (deg_matrix_vscontrol_df != 0).any(axis=0)]
# Remove rows with all zeros
deg_matrix_vscontrol_df = deg_matrix_vscontrol_df.loc[(deg_matrix_vscontrol_df != 0).any(axis=1), :]

if DATASET_NAME == 'replogle22':
    # Downsample perturbations randomly to 256
    print(deg_matrix_vscontrol_df.shape)
    deg_matrix_vscontrol_df = deg_matrix_vscontrol_df.sample(n=256, axis=0)
    print(deg_matrix_vscontrol_df.shape)
    # Downsample genes randomly to 2048
    deg_matrix_vscontrol_df = deg_matrix_vscontrol_df.sample(n=2048, axis=1)
    print(deg_matrix_vscontrol_df.shape)

# Plot the heatmap
plt.clf()
label_fontsize = 6 # Smaller font size

g = sns.clustermap(
    deg_matrix_vscontrol_df.T,
    cmap='bwr', 
    center=0,      # Center color scale at 0
    figsize=(25, 6), 
    linewidths=0.0, 
    row_cluster=True,
    xticklabels=DATASET_NAME == 'norman19',
    yticklabels=False,
    cbar=True,
)

# Turn off the dendrogram axes
if hasattr(g, 'ax_row_dendrogram'):
    g.ax_row_dendrogram.set_visible(False)

if hasattr(g, 'ax_col_dendrogram'):
    g.ax_col_dendrogram.set_visible(False)

# Adjust label properties
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=label_fontsize)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize)
# Add x and y axis labels to the heatmap
g.ax_heatmap.set_ylabel("Genes", fontsize=12)
g.ax_heatmap.set_xlabel("Perturbations", fontsize=12)

# Rotate the x-axis labels 45 degrees
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Set title on the figure with bold font weight and size 14
g.fig.suptitle('DEGs (vs control) - Clustered Heatmap', y=1.02, fontsize=14, fontweight='bold')

g.fig.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_clustermap_vscontrol.png', dpi=300, bbox_inches='tight')
print(f"Saved clustermap to {PLOT_DIR}/{DATASET_NAME}_deg_clustermap_vscontrol.png")
plt.close(g.fig)


## For each DEG, find the number of perturbations that have it as a DEG
deg_matrix_vscontrol_sum_df = deg_matrix_vscontrol_df.astype(int)
# Convert to absolute value
deg_matrix_vscontrol_sum_df = np.abs(deg_matrix_vscontrol_sum_df)
# Sum over the rows to get the number of perturbations that have it as a DEG
deg_matrix_vscontrol_sum_series = deg_matrix_vscontrol_sum_df.sum(axis=0)
# Plot the distribution of the number of perturbations that have it as a DEG
plt.figure() # Use plt.figure() to create a new figure
sns.histplot(deg_matrix_vscontrol_sum_series, bins=100)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_dist_vscontrol.png', dpi=300)
print(f"Saved distribution to {PLOT_DIR}/{DATASET_NAME}_deg_dist_vscontrol.png")
plt.close()

## 1. Get the number of DEGs between each single perturbation and all other perturbations
deg_dict_vsctrl = adata.uns['deg_dict_vscontrol']
# Get the distribution of the number of DEGs
deg_counts_vsctrl = [len(deg_dict_vsctrl[pert]['up']) + len(deg_dict_vsctrl[pert]['down']) for pert in deg_dict_vsctrl.keys()]
deg_dict_vsrest = adata.uns['deg_dict_vsrest']
deg_counts_vsrest = [len(deg_dict_vsrest[pert]['up']) + len(deg_dict_vsrest[pert]['down']) for pert in deg_dict_vsrest.keys()]

# Plot the distribution of the number of DEGs
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(deg_counts_vsctrl, bins=100, color='red', ax=ax[0])
ax[0].set_title(f'DEGs (vs ctrl) ({DATASET_NAME})', fontsize=16, fontweight='bold')
ax[0].set_xlabel('Number of DEGs', fontsize=14)
ax[0].set_ylabel('Count', fontsize=14)

sns.histplot(deg_counts_vsrest, bins=100, color='blue', ax=ax[1])
ax[1].set_title(f'Downregulated DEGs (vs rest) ({DATASET_NAME})', fontsize=16, fontweight='bold')
ax[1].set_xlabel('Number of DEGs', fontsize=14)
ax[1].set_ylabel('Count', fontsize=14)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_counts_subplots.png', dpi=300)
print(f"Saved subplots to {PLOT_DIR}/{DATASET_NAME}_deg_counts_subplots.png")

### For each DEG, find the number of perturbations that have it as a DEG when compared to the rest
# First, get the number of DEGs between each single perturbation and all other perturbations
deg_dict_vsrest = adata.uns['deg_dict_vsrest']

pert_names_rest = list(deg_dict_vsrest.keys())
# gene_names are same as above
pert_to_idx_rest = {pert: i for i, pert in enumerate(pert_names_rest)}
# gene_to_idx is same as above

num_perts_rest = len(pert_names_rest)
# num_genes is same as above
deg_matrix_np_rest = np.zeros((num_perts_rest, num_genes), dtype=np.int8) # Use int8 for memory

for pert, deg_info in tqdm(deg_dict_vsrest.items()):
    pert_idx = pert_to_idx_rest[pert]
    up_genes = deg_info.get('up', [])
    down_genes = deg_info.get('down', [])

    # Get column indices for valid genes that exist in adata.var_names
    up_gene_indices = [gene_to_idx[g] for g in up_genes if g in gene_to_idx]
    down_gene_indices = [gene_to_idx[g] for g in down_genes if g in gene_to_idx]

    if up_gene_indices:
        deg_matrix_np_rest[pert_idx, up_gene_indices] = 1
    if down_gene_indices:
        deg_matrix_np_rest[pert_idx, down_gene_indices] = -1

# Convert back to DataFrame
deg_matrix_vsrest_df = pd.DataFrame(deg_matrix_np_rest, index=pert_names_rest, columns=gene_names)
# Remove columns with all zeros
deg_matrix_vsrest_df = deg_matrix_vsrest_df.loc[:, (deg_matrix_vsrest_df != 0).any(axis=0)]
# Remove rows with all zeros
deg_matrix_vsrest_df = deg_matrix_vsrest_df.loc[(deg_matrix_vsrest_df != 0).any(axis=1), :]

deg_matrix_vsrest_sum_df = deg_matrix_vsrest_df.astype(int)
# Convert to absolute value
deg_matrix_vsrest_sum_df = np.abs(deg_matrix_vsrest_sum_df)
# Sum over the rows to get the number of perturbations that have it as a DEG
deg_matrix_vsrest_sum_series = deg_matrix_vsrest_sum_df.sum(axis=0)


## Plot the relationship between the number of DEGs in vscontrol and vsrest
# Get the union of the DEGs in vscontrol and vsrest
degs_vsrest_list = deg_matrix_vsrest_sum_series.index.tolist()
degs_vscontrol_list = deg_matrix_vscontrol_sum_series.index.tolist()
degs_union = list(set(degs_vsrest_list) | set(degs_vscontrol_list))

# Create a dataframe with the union of the DEGs in vscontrol and vsrest
deg_matrix_vscontrol_sum_union_df = pd.DataFrame(index=degs_union, columns=['vscontrol', 'vsrest'])
for deg in tqdm(degs_union):
    if deg in deg_matrix_vscontrol_sum_series.index:
        deg_matrix_vscontrol_sum_union_df.loc[deg, 'vscontrol'] = deg_matrix_vscontrol_sum_series.loc[deg]
    if deg in deg_matrix_vsrest_sum_series.index:
        deg_matrix_vscontrol_sum_union_df.loc[deg, 'vsrest'] = deg_matrix_vsrest_sum_series.loc[deg]
# Fill na with 0
deg_matrix_vscontrol_sum_union_df.fillna(0, inplace=True)

# Plot the relationship between the number of DEGs in vscontrol and vsrest
fig, ax = plt.subplots(figsize=(11, 9))

x = deg_matrix_vscontrol_sum_union_df['vscontrol'].astype(float) # Ensure float for gaussian_kde
y = deg_matrix_vscontrol_sum_union_df['vsrest'].astype(float)  # Ensure float for gaussian_kde

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x_sorted, y_sorted, z_sorted = x.iloc[idx], y.iloc[idx], z[idx]


sct = ax.scatter(x=x_sorted, y=y_sorted,
                c=z_sorted,  # Use density for color
                cmap="viridis", # Choose a colormap
                norm=colors.LogNorm(), # Apply log scale to color
                s=50, # Adjust point size if needed
                edgecolor=None # Optional: remove point edges
                )

# Add a color bar
cbar = fig.colorbar(sct, ax=ax)
cbar.set_label('Log Point Density')

# Plot xy line
max_val = max(deg_matrix_vscontrol_sum_union_df['vscontrol'].max(), deg_matrix_vscontrol_sum_union_df['vsrest'].max())
ax.plot([0, max_val], [0, max_val], color='red', linewidth=1, linestyle='--')

ax.set_xlabel('# Perts with DEG vs control')
ax.set_ylabel('# Perts with DEG vs all other perts')
ax.set_title('DEG prevalence in perts vs control or vs all other perts')

fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_vsrest_density.png', dpi=300)
print(f"Saved density plot to {PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_vsrest_density.png")
plt.close(fig)


# Get the percent of perturbations each DEG is detected in
deg_matrix_vscontrol_sum_union_df['vscontrol_percent'] = 100 * deg_matrix_vscontrol_sum_union_df['vscontrol'] / num_perts
deg_matrix_vscontrol_sum_union_df['vsrest_percent'] = 100 * deg_matrix_vscontrol_sum_union_df['vsrest'] / num_perts
deg_matrix_vscontrol_sum_union_df['vscontrol_percent_rank'] = deg_matrix_vscontrol_sum_union_df['vscontrol_percent'].rank(pct=True)
deg_matrix_vscontrol_sum_union_df['vsrest_percent_rank'] = deg_matrix_vscontrol_sum_union_df['vsrest_percent'].rank(pct=True)

# Plot scatter of vscontrol_percent vs vscontrol
fig, ax = plt.subplots(figsize=(7.75, 6))

# Plot the scatter
ax.scatter(deg_matrix_vscontrol_sum_union_df['vscontrol_percent_rank'], deg_matrix_vscontrol_sum_union_df['vscontrol_percent'], alpha=0.5)

# Draw a horizontal line at 50%
ax.axhline(y=50, color='r', linestyle='--', linewidth=1)

# Annotate top 5 genes with lines and attempted non-overlap
# Get the top 5 genes, nlargest sorts by 'vscontrol_percent' descending by default
top_5_genes_by_value = deg_matrix_vscontrol_sum_union_df.nlargest(5, 'vscontrol_percent')

# Define offsets for text labels (in points) and corresponding alignments
# These are applied to the top_5_genes_by_value in their order (highest percentage first)
label_configs = [
    (20, 20, 'left', 'bottom'),        # For the gene with the highest percentage
    (30, 10, 'left', 'bottom'),       # For the 2nd highest
    (25, -20, 'left', 'top'),         # For the 3rd highest
    (-30, 15, 'right', 'bottom'),      # For the 4th highest - attempting to place it higher
    (-25, -15, 'right', 'top'),     # For the 5th highest - attempting to place it lower
]

for i, (index, row) in enumerate(top_5_genes_by_value.iterrows()):
    gene_name = index
    x_coord = row['vscontrol_percent_rank']
    y_coord = row['vscontrol_percent']
    
    # Ensure we don't go out of bounds if there are fewer than 5 genes for some reason (though nlargest should handle this)
    if i < len(label_configs):
        offset_x, offset_y, ha, va = label_configs[i]
    else: # Default fallback if more than 5 (should not happen with nlargest(5))
        offset_x, offset_y, ha, va = (20, 20, 'left', 'bottom')

    ax.annotate(
        f'{gene_name} ({y_coord:.2f}%)',
        xy=(x_coord, y_coord),
        xytext=(offset_x, offset_y),
        textcoords='offset points',
        fontsize=8, 
        ha=ha,
        va=va,
        arrowprops=dict(arrowstyle='-', lw=0.75, color='dimgray', connectionstyle="arc3,rad=0.15")
    )

ax.set_xlabel('Rank')
ax.set_ylabel('% perts w/ DEG (vs control)')
ax.set_title(f'DEG (vs control) prevalence across perts ({DATASET_NAME})', fontsize=14, fontweight='bold')

# Increase y-axis upper limit to give more space for title/annotations
current_ylim = ax.get_ylim()
new_ylim_top = current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.1 # Increase top by 10% of current range
ax.set_ylim(current_ylim[0], new_ylim_top)

# Add text above the 50% line showing the number of DEGs in vscontrol > 50%
ax.text(
    0.75,               # 85% of the axis width → further right
    51,                 # still the same data y-value
    f'# DEGs > 50%: {(deg_matrix_vscontrol_sum_union_df["vscontrol_percent"] > 50).sum()}',
    transform=ax.get_yaxis_transform(),   # x→axes fraction, y→data
    fontsize=8, ha='center', va='center'
)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_percent.png', dpi=300)
print(f"Saved percent plot to {PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_percent.png")
plt.close(fig)

# Print the number of DEGs in vscontrol > 50%
print(f"# DEGs > 50%: {len(deg_matrix_vscontrol_sum_union_df[deg_matrix_vscontrol_sum_union_df['vscontrol_percent'] > 50])}")


## Plot the relationship between the percent of DEGs in vscontrol and vsrest
fig, ax = plt.subplots(figsize=(7.75, 6))

x_pct = deg_matrix_vscontrol_sum_union_df['vscontrol_percent'].astype(float)
y_pct = deg_matrix_vscontrol_sum_union_df['vsrest_percent'].astype(float)

# Calculate the point density
xy_pct = np.vstack([x_pct, y_pct])
z_pct = gaussian_kde(xy_pct)(xy_pct)

# Sort the points by density, so that the densest points are plotted last
idx_pct = z_pct.argsort()
x_pct_sorted, y_pct_sorted, z_pct_sorted = x_pct.iloc[idx_pct], y_pct.iloc[idx_pct], z_pct[idx_pct]

sct_pct = ax.scatter(x=x_pct_sorted, y=y_pct_sorted,
                    c=z_pct_sorted,  # Use density for color
                    cmap="viridis", # Choose a colormap
                    norm=colors.LogNorm(), # Apply log scale to color
                    s=50, # Adjust point size if needed
                    edgecolor=None # Optional: remove point edges
                    )

# Add a color bar
cbar_pct = fig.colorbar(sct_pct, ax=ax)
cbar_pct.set_label('Log Point Density')

# Plot xy line (from 0 to 1 for percentages)
ax.plot([0, 100], [0, 100], color='red', linewidth=1, linestyle='--')

ax.set_xlabel('% perts w/ DEG (vs control)')
ax.set_ylabel('% perts w/ DEG (vs all other perts)')
ax.set_title(f'DEG prevalence (control vs other perts) ({DATASET_NAME})', fontsize=14, fontweight='bold')

# Set axis limits to be 0-1 for percentages
ax.set_xlim(0, 100)
ax.set_ylim(0, 110)

fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_vsrest_density_percent.png', dpi=300)
print(f"Saved density plot (percentage) to {PLOT_DIR}/{DATASET_NAME}_deg_vscontrol_vsrest_density_percent.png")
plt.close(fig)





#### 7. g: Number of genes

mae_dict_n_genes = {}
mse_dict_n_genes = {}
corr_delta_dict_n_genes = {}

n_genes_to_test = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

for n_genes in tqdm(n_genes_to_test):
    # Randomly select n_genes from the adata object
    random_genes = np.random.choice(adata.var_names, size=n_genes, replace=False)
    random_genes_idx = adata.var_names.isin(random_genes)

    # Get total mean with the random genes
    total_mean_random_genes = total_mean_original[random_genes_idx]
    # Get control mean with the random genes
    ctrl_mean_random_genes = ctrl_mean_original[random_genes_idx]

    # For each pert, calculate metrics
    for pert in all_perts:
        if not pert in mae_dict_n_genes.keys():
            mae_dict_n_genes[pert] = {}
            mse_dict_n_genes[pert] = {}
            corr_delta_dict_n_genes[pert] = {}
        # Get the mean of the pert
        pert_mean_random_genes = pert_means[pert][random_genes_idx]
        
        # Calculate metrics
        mae_dict_n_genes[pert][n_genes] = mae(total_mean_random_genes, pert_mean_random_genes)
        mse_dict_n_genes[pert][n_genes] = mse(total_mean_random_genes, pert_mean_random_genes)
        current_pearson_delta = pearson(total_mean_random_genes - ctrl_mean_random_genes, pert_mean_random_genes - ctrl_mean_random_genes)
        corr_delta_dict_n_genes[pert][n_genes] = current_pearson_delta

# Plot the metrics as a scatter plot
PLOT_DIR = f'{ANALYSIS_DIR}/g_effect_of_n_genes'
plot_metrics_as_scatter_trend(mae_dict_n_genes, PLOT_DIR, 'MAE perturbation vs all', DATASET_NAME, xlabel='Number of genes', ylabel='MAE')
plot_metrics_as_scatter_trend(mse_dict_n_genes, PLOT_DIR, 'MSE perturbation vs all', DATASET_NAME, xlabel='Number of genes', ylabel='MSE')
plot_metrics_as_scatter_trend(corr_delta_dict_n_genes, PLOT_DIR, 'Pearson delta perturbation vs all', DATASET_NAME, xlabel='Number of genes', ylabel='Pearson R')

# Get the aggregate values for the metrics
metric_dicts = {
    'mae_dict': mae_dict_n_genes,
    'mse_dict': mse_dict_n_genes,
    'corr_delta_dict': corr_delta_dict_n_genes
}
aggregate_vals = {}
for metric_name in ['mae', 'mse', 'corr_delta']:
    metric_dict_key = f'{metric_name}_dict'
    # Use get_aggregate_correlation_from_dict()
    aggregate_vals[metric_name] = get_aggregate_correlation_from_dict(metric_dicts[metric_dict_key], log_x=True, log_x_base=2)

with open(f'{ANALYSIS_DIR}/g_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(aggregate_vals, f)

#### 8. mu_l: effect of library size

n_counts = adata.obs['ncounts'].values

# Get the quantiles for the perturbed and the control cells
quantiles = np.arange(0, 1, 0.1)

# For each perturbation get the quantiles of the library size and the associated cells
# Then get the mean of the quantile-specific pert cells
# Then we aggregate the means to get the total_mean for each quantile
pert_mean_quantile_dict = {}
total_mean_quantile_dict = {}
ctrl_mean_quantile_dict = {}
for i, quantile in tqdm(enumerate(quantiles)):
    quantile_next = quantiles[i+1] if i < len(quantiles) - 1 else 1
    quantile_string = f"{quantile:.1f}-{quantile_next:.1f}"
    pert_mean_quantile_dict[quantile_string] = {}
    total_mean_quantile_dict[quantile_string] = {}
    ctrl_mean_quantile_dict[quantile_string] = {}
    for pert in all_perts:
        pert_cells_idx = adata.obs['condition'] == pert
        pert_n_counts = n_counts[pert_cells_idx]
        cell_ids = adata.obs_names[pert_cells_idx]
        # Rank the pert_n_counts
        pert_n_counts_ranked = pd.Series(pert_n_counts).rank(pct=True).values
        # Get the cells that are in the quantile
        quantile_cells_idx = (pert_n_counts_ranked >= quantile) & (pert_n_counts_ranked < quantile_next)
        # Get the mean of the pert cells
        pert_mean_quantile_dict[quantile_string][pert] = adata[cell_ids[quantile_cells_idx]].X.mean(axis=0).A1
    
    # Get the total mean of the quantile by averaging all the pert means in that quantile
    total_mean_quantile_dict[quantile_string] = np.mean(list(pert_mean_quantile_dict[quantile_string].values()), axis=0)
    # Get the control mean of the quantile by finding the cells that are in the quantile and then averaging their means
    ctrl_cells_idx = adata.obs['condition'] == 'control'
    cell_ids = adata.obs_names[ctrl_cells_idx]
    ctrl_n_counts = n_counts[ctrl_cells_idx]
    ctrl_n_counts_ranked = pd.Series(ctrl_n_counts).rank(pct=True).values
    ctrl_quantile_cells_idx = (ctrl_n_counts_ranked >= quantile) & (ctrl_n_counts_ranked < quantile_next)
    ctrl_mean_quantile_dict[quantile_string] = adata[cell_ids[ctrl_quantile_cells_idx]].X.mean(axis=0).A1

# Then calculate the metrics
mae_dict_theta = {}
mse_dict_theta = {}
corr_delta_dict_theta = {}

for pert in tqdm(all_perts):
    mae_dict_theta[pert] = {}
    mse_dict_theta[pert] = {}
    corr_delta_dict_theta[pert] = {}

    for quantile_string in pert_mean_quantile_dict.keys():
        pert_mean_qt = pert_mean_quantile_dict[quantile_string][pert]
        total_mean_qt = total_mean_quantile_dict[quantile_string]

        mae_dict_theta[pert][quantile_string] = mae(total_mean_qt, pert_mean_qt)
        mse_dict_theta[pert][quantile_string] = mse(total_mean_qt, pert_mean_qt)
        
        delta_pert_vs_control = pert_mean_qt - ctrl_mean_quantile_dict[quantile_string]
        delta_all_vs_control = total_mean_quantile_dict[quantile_string] - ctrl_mean_quantile_dict[quantile_string] 
        current_pearson_delta = pearson(delta_all_vs_control, delta_pert_vs_control)
        corr_delta_dict_theta[pert][quantile_string] = current_pearson_delta

# Plot the metrics as a scatter plot
PLOT_DIR = f'{ANALYSIS_DIR}/mu_l_effect_of_library_size'
plot_categorical_scatter_trend(mae_dict_theta, PLOT_DIR, 'MAE perturbation vs all', DATASET_NAME, xlabel='Quantile of library size', ylabel='MAE')
plot_categorical_scatter_trend(mse_dict_theta, PLOT_DIR, 'MSE perturbation vs all', DATASET_NAME, xlabel='Quantile of library size', ylabel='MSE')
plot_categorical_scatter_trend(corr_delta_dict_theta, PLOT_DIR, 'Pearson delta perturbation vs all', DATASET_NAME, xlabel='Quantile of library size', ylabel='Pearson R')

# Get the aggregate values for the metrics
metric_dicts = {
    'mae_dict': mae_dict_theta,
    'mse_dict': mse_dict_theta,
    'corr_delta_dict': corr_delta_dict_theta
}
aggregate_vals = {}
for metric_name in ['mae', 'mse', 'corr_delta']:
    metric_dict_key = f'{metric_name}_dict'
    # Use get_aggregate_correlation_from_dict()
    aggregate_vals[metric_name] = get_aggregate_correlation_from_dict_categorical(metric_dicts[metric_dict_key])

with open(f'{ANALYSIS_DIR}/mu_l_aggregate_vals.pkl', 'wb') as f:
    pickle.dump(aggregate_vals, f)






