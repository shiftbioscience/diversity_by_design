import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import spearmanr, linregress, pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scanpy as sc
from tqdm import tqdm
import argparse
from sklearn.metrics import r2_score
from typing import Callable, Tuple, Any

ANALYSIS_DIR = './'

np.random.seed(42)


def mae(x1, x2):
    return np.mean(np.abs(x1 - x2))

def mse(x1, x2):
    return np.mean((x1 - x2) ** 2)

def wmse(x1, x2, weights):
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))

def pearson(x1, x2):
    return np.corrcoef(x1, x2)[0, 1]

def r2_score_on_deltas(delta_true, delta_pred, weights=None):
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan
    if weights is not None:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)

def get_pert_means(adata):
    perturbations = adata.obs['condition'].unique()
    pert_means = {}
    for pert in tqdm(perturbations, desc="Calculating perturbation means"):
        pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
        pert_counts = adata[pert_cells].X.toarray()
        pert_means[pert] = np.mean(pert_counts, axis=0)
    return pert_means


def initialize_analysis(dataset_name, analysis_name):
    """
    Initializes analysis by setting up paths, parsing arguments, loading data,
    and calculating initial means.
    """
    # Set matplotlib parameters to create professional plots similar to R's cowplot package
    plt.rcParams.update({
        # Figure aesthetics
        'figure.facecolor': 'white',
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        
        # Text properties
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Axes properties
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Tick properties
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend properties
        'legend.frameon': False,
        'legend.fontsize': 10,
        
        # Saving properties
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

    # Store the original random seed state
    original_np_random_state = np.random.get_state()
    np.random.seed(42) # Initial seed


    # Define paths and dataset-specific variables
    DATA_CACHE_DIR = '../../data/' # Relative to the script calling this function    
    dataset_specific_subdir = dataset_name # e.g., "norman19"

    if dataset_name == 'norman19':
        data_path = os.path.join(DATA_CACHE_DIR, 'norman19/norman19_processed.h5ad')
        DATASET_NAME = 'norman19'
        ANALYSIS_DIR = f'norman19'
        DATASET_CELL_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256]
        DATASET_PERTS_TO_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 175]
    elif dataset_name == 'replogle22':
        data_path = os.path.join(DATA_CACHE_DIR, 'replogle22/replogle22_processed.h5ad')
        DATASET_NAME = 'replogle22'
        ANALYSIS_DIR = f'replogle22'
        DATASET_CELL_COUNTS = [2, 4, 8, 16, 32, 64]
        DATASET_PERTS_TO_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1334]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create analysis directory if it doesn't exist
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Clear the analysis directory
    os.system('find ' + ANALYSIS_DIR + ' -type f -name "*.png" -exec rm -f {} \;')

    # Load data
    adata = sc.read_h5ad(data_path)

    # Get means
    pert_means_dict = get_pert_means(adata)
    total_mean_original = np.mean(list(pert_means_dict.values()), axis=0)
    ctrl_mean_original = adata[adata.obs['condition'] == 'control'].X.mean(axis=0).A1

    
    # Pre-calculate normalized absolute scores for weighting WMSE
    # Using scores_df_vsrest as requested
    scores_df_vsrest_path = f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_scores_df_vsrest.pkl'
    names_df_vsrest_path = f'{DATA_CACHE_DIR}/{DATASET_NAME}/{DATASET_NAME}_names_df_vsrest.pkl'
    if os.path.exists(scores_df_vsrest_path) and os.path.exists(names_df_vsrest_path):
        scores_df_vsrest = pd.read_pickle(scores_df_vsrest_path)
        names_df_vsrest = pd.read_pickle(names_df_vsrest_path)
        pert_normalized_abs_scores_vsrest = {}
        for pert in tqdm(scores_df_vsrest.columns, desc="Calculating WMSE Weights"):
            if pert == 'control': # Typically no scores for control in vsrest, but good to check
                continue

            abs_scores = np.abs(scores_df_vsrest[pert].values) # Ensure it's a numpy array
            min_val = np.min(abs_scores)
            max_val = np.max(abs_scores)
            
            if max_val == min_val:
                if max_val == 0: # All scores are 0
                    normalized_weights = np.zeros_like(abs_scores)
                else: # All scores are the same non-zero value
                    # Squaring ones will still be ones, which is fine.
                    normalized_weights = np.ones_like(abs_scores) 
            else:
                normalized_weights = (abs_scores - min_val) / (max_val - min_val)
            
            # Ensure no NaNs in weights, replace with 0 if any (e.g. if a gene had NaN score originally)
            normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
            
            # Make weighting stronger by squaring the normalized weights
            normalized_weights = np.square(normalized_weights)
            
            weights = pd.Series(normalized_weights, index=names_df_vsrest[pert].values, name=pert)
            # Order by the var_names
            weights = weights.reindex(adata.var_names)
            pert_normalized_abs_scores_vsrest[pert] = weights
    else:
        print(f"WMSE Warning: scores_df_vsrest.pkl not found at {scores_df_vsrest_path}. WMSE will not be calculated or will use fallback.")
        scores_df_vsrest = None # Ensure it's defined for later checks
        pert_normalized_abs_scores_vsrest = {} # Empty dict so lookups can fail gracefully or use defaults


    # Get the perts with at least N cells
    max_cells_per_pert = max(DATASET_CELL_COUNTS)
    pert_counts = adata.obs['condition'].value_counts()
    pert_counts = pert_counts[(pert_counts >= max_cells_per_pert) & (pert_counts.index != 'control')]


    
    return (
        adata,
        pert_means_dict,
        total_mean_original,
        ctrl_mean_original,
        DATASET_NAME,
        DATASET_CELL_COUNTS,
        DATASET_PERTS_TO_SWEEP,
        dataset_specific_subdir, # e.g. "norman19"
        DATA_CACHE_DIR,
        original_np_random_state,
        ANALYSIS_DIR,
        pert_normalized_abs_scores_vsrest,
        pert_counts,
        scores_df_vsrest,
        names_df_vsrest
    )

# Function to calculate aggregate correlation from a structured dictionary for categorical x-axis (e.g. Step 4)
# Takes a dict where keys are categories (e.g. '0.0-0.1') and values are dicts of {pert: metric_value}
# Returns a single Pearson R value representing the correlation between the rank of the categories and the metric values.
def sort_key_for_levels(level_input):
    try:
        # Attempt to convert to string, split, and take the first part as float
        return float(str(level_input).split('-')[0])
    except (ValueError, IndexError, AttributeError):
        # Fallback for keys not matching "number-number" string format
        # or if conversion to string/float fails.
        # This ensures all keys are processed using a consistent string representation for sorting.
        return str(level_input) 

def get_aggregate_correlation_from_dict(data_dict: dict, log_x: bool = False, log_x_base: int = 2) -> float:
    all_independent_vars = []
    all_dependent_vars = []
    for outer_key, inner_dict in data_dict.items():
        for independent_var, dependent_var in inner_dict.items():
            all_independent_vars.append(independent_var)
            all_dependent_vars.append(dependent_var)
    
    independent_vars_arr = np.array(all_independent_vars, dtype=float)
    dependent_vars_arr = np.array(all_dependent_vars, dtype=float)
    # If log_x, take the log of the independent variable
    if log_x:
        independent_vars_arr = np.log(independent_vars_arr, where=independent_vars_arr > 0) / np.log(log_x_base)
    
    # Combine arrays
    combined_arr = np.column_stack((independent_vars_arr, dependent_vars_arr))
    # Remove any rows where the dependent variable is nan
    combined_arr = combined_arr[~np.isnan(combined_arr[:, 1])]
    
    corr, _ = pearsonr(combined_arr[:, 0], combined_arr[:, 1])
    return corr


def get_aggregate_correlation_from_dict_categorical(data_dict: dict) -> float:
    all_independent_vars_categorical = []
    all_dependent_vars = []
    
    for outer_key, inner_dict in data_dict.items():
        for independent_var_cat, dependent_var in inner_dict.items():
            all_independent_vars_categorical.append(independent_var_cat)
            all_dependent_vars.append(dependent_var)
            
    dependent_vars_arr = np.array(all_dependent_vars, dtype=float)
    
    # Using a DataFrame simplifies handling of mixed types and NaNs
    df = pd.DataFrame({
        'independent_cat': all_independent_vars_categorical,
        'dependent_num': dependent_vars_arr
    })
    
    # Remove rows where the dependent variable is NaN, or if independent_cat is NaN
    df_cleaned = df.dropna(subset=['dependent_num', 'independent_cat'])
    
    if df_cleaned.shape[0] < 2: # Need at least 2 valid data points for correlation
        return np.nan

    unique_categories_from_df = df_cleaned['independent_cat'].unique()
    
    # Attempt to sort categories to respect ordinal nature
    try:
        # Handles cases like "0.0-0.1" or numeric types (converted to str then float)
        sorted_unique_categories = sorted(unique_categories_from_df, 
                                          key=lambda x: float(str(x).split('-')[0]))
    except (ValueError, IndexError, TypeError):
        # Fallback to simple string sort if the above sophisticated sort fails
        try:
            sorted_unique_categories = sorted(unique_categories_from_df, key=str)
        except TypeError: 
            # Ultimate fallback if items are not consistently stringifiable or sortable by string representation
            sorted_unique_categories = sorted(unique_categories_from_df) # Relies on default sortability

    if len(sorted_unique_categories) < 2: # Not enough distinct categories to define ranks with variance
        return np.nan

    category_to_rank = {category: rank for rank, category in enumerate(sorted_unique_categories)}
    
    independent_vars_numerical_ranks = df_cleaned['independent_cat'].map(category_to_rank).values
    dependent_vars_cleaned_values = df_cleaned['dependent_num'].values
    
    # Check for conditions where pearsonr would return NaN or error, to ensure robust handling.
    if len(independent_vars_numerical_ranks) < 2: # Should be caught by df_cleaned.shape[0] < 2 already
         return np.nan
    # If standard deviation is zero, pearson correlation is not defined.
    if np.std(independent_vars_numerical_ranks) == 0.0 or np.std(dependent_vars_cleaned_values) == 0.0:
         return np.nan

    corr, _ = pearsonr(independent_vars_numerical_ranks, dependent_vars_cleaned_values)
    return corr



def get_aggregate_correlation_for_categorical_levels(metric_dict):
    all_level_ranks_collected = []
    all_metric_values_collected = []

    raw_levels = list(metric_dict.keys())
    try:
        # Sort unique levels:
        # - Numerically if they follow "num-num" pattern (e.g., "0.0-0.1")
        # - Alphabetically as strings otherwise
        unique_sorted_levels = sorted(raw_levels, key=sort_key_for_levels)
    except TypeError: 
        # Broad exception for unorderable types if sort_key_for_levels produced mixed types 
        # (e.g. if some keys are numbers and others are complex strings not handled by key func)
        # Default to simple string sort of original keys.
        unique_sorted_levels = sorted(raw_levels, key=str)

    level_to_rank_map = {level: i for i, level in enumerate(unique_sorted_levels)}

    for level_key, pert_value_dict in metric_dict.items():

        current_rank = level_to_rank_map.get(level_key)
        for pert_name, metric_val in pert_value_dict.items():
            all_level_ranks_collected.append(current_rank)
            all_metric_values_collected.append(metric_val)

    # Convert collected lists to numpy arrays. np.array(..., dtype=float) handles None -> np.nan.
    level_ranks_arr = np.array(all_level_ranks_collected, dtype=float)
    metric_values_arr = np.array(all_metric_values_collected, dtype=float) 

    # Filter out pairs where metric_values_arr (dependent variable) is NaN
    valid_indices = ~np.isnan(metric_values_arr)
    
    filtered_level_ranks_arr = level_ranks_arr[valid_indices]
    filtered_metric_values_arr = metric_values_arr[valid_indices]

    corr, _ = pearsonr(filtered_level_ranks_arr, filtered_metric_values_arr)
    return corr

def plot_metrics_as_clustermap(dict2plot, subdir, title, dataset_name, pearson=False):
    df = pd.DataFrame.from_dict(dict2plot, orient='index')
    df = df.dropna()
    if pearson:
        g = sns.clustermap(df, cmap='coolwarm', vmin=-1, vmax=1, center=0, yticklabels=True, col_cluster=False)
    else:
        g = sns.clustermap(df, cmap='coolwarm', yticklabels=True, col_cluster=False)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=4)
    full_title = f'{title} ({dataset_name})'
    g.fig.suptitle(full_title, y=1.02, fontsize=14, fontweight='bold')
    filepath_title = title.replace(' ', '_')
    # Replace parentheses with underscores
    filepath_title = re.sub(r'\(|\)', '_', filepath_title)
    os.makedirs(f'{ANALYSIS_DIR}/{subdir}', exist_ok=True)
    g.fig.savefig(f'{ANALYSIS_DIR}/{subdir}/{filepath_title}.png', dpi=300, bbox_inches='tight')
    print(f'{full_title} saved to {ANALYSIS_DIR}/{subdir}/{filepath_title}.png')
    plt.close(g.fig)

def plot_metrics_as_density(dict_to_plot, subdir, title, dataset_name, xlabel_override=None, legend_title_override=None):
    data_for_df = []
    for pert, level_dict in dict_to_plot.items():
        for level_key, metric_value in level_dict.items():
            if pd.notna(metric_value): # Ensure we don't try to plot NaNs
                data_for_df.append({'perturbation': pert, 'level': level_key, 'metric_value': metric_value})

    if not data_for_df:
        print(f"No data to plot for {title}")
        return

    df = pd.DataFrame(data_for_df)

    try:
        df['level_numeric'] = pd.to_numeric(df['level'].astype(str))
        unique_sorted_levels = sorted(df['level_numeric'].unique())
        df['plot_level'] = pd.Categorical(df['level_numeric'], categories=unique_sorted_levels, ordered=True)
        actual_legend_title = legend_title_override if legend_title_override else 'Level (Numeric)'
    except ValueError: 
        df['level_str'] = df['level'].astype(str)
        unique_sorted_levels = sorted(df['level_str'].unique())
        df['plot_level'] = pd.Categorical(df['level_str'], categories=unique_sorted_levels, ordered=True)
        actual_legend_title = legend_title_override if legend_title_override else 'Level (Categorical)'

    df.rename(columns={'plot_level': actual_legend_title}, inplace=True)

    plt.clf()

    plt.figure(figsize=(7, 6))
    
    num_unique_levels = len(unique_sorted_levels)
    palette = sns.color_palette("viridis", n_colors=num_unique_levels) if num_unique_levels > 0 else None

    sns.kdeplot(data=df, x='metric_value', hue=actual_legend_title, 
                multiple='layer', alpha=0.5, fill=True, 
                palette=palette, hue_order=unique_sorted_levels, 
                common_norm=False, common_grid=False)

    full_title = f'{title} ({dataset_name})'
    plt.title(full_title, fontsize=14, fontweight='bold')
    
    xlabel_text = xlabel_override if xlabel_override else 'Metric Value'
    plt.xlabel(xlabel_text, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    # Add vlines and text for the mean of each level
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    # Create a mapping from level to color
    level_to_color = {level: palette[i] for i, level in enumerate(unique_sorted_levels)}

    for level in unique_sorted_levels:
        mean_val = df[df[actual_legend_title] == level]['metric_value'].mean()
        if pd.notna(mean_val):
            color = level_to_color[level]
            ax.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
            y_text_pos = ymax * (0.9 - (unique_sorted_levels.index(level) % 5) * 0.03)
            ax.text(mean_val, y_text_pos, f'{mean_val:.2f}', color=color, 
                    ha='center', va='bottom', fontweight='bold', backgroundcolor=(1,1,1,0.6))

    # Add a general text explanation for the v-lines
    ax.text(0.01, 0.01, 'Dashed lines indicate mean for each level', 
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    handles, labels = ax.get_legend_handles_labels()
    if handles: 
         plt.legend(handles, labels, title=actual_legend_title, loc='best')

    if xlabel_text == 'Pearson R':
        current_xlim = ax.get_xlim()
        # Ensure the plot doesn't extend much beyond 1.0 for correlation
        ax.set_xlim(left=current_xlim[0], right=min(current_xlim[1], 1.05))

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in title).rstrip()
    filepath_title = filepath_title.replace(' ', '_')
    
    os.makedirs(f'{ANALYSIS_DIR}/{subdir}', exist_ok=True)
    save_path = f'{ANALYSIS_DIR}/{subdir}/{filepath_title}.density.png'
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Density plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    finally:
        plt.close() 

def plot_metrics_as_scatter_trend(
    dict_to_plot, 
    subdir, 
    plot_title, 
    dataset_name, 
    xlabel, 
    ylabel, 
    use_log_x=False, log_x_base=10, x_ticks_override=None, jitter_strength=0.2, plot_trend=True, yaxis_limits=None
):
    """
    Plots metrics as a scatter plot with a smoothed trend line.
    Aggregates data from all perturbations.
    X-axis: levels (e.g., cell counts).
    Y-axis: metric values.
    """
    data_for_df = []
    for pert, level_dict in dict_to_plot.items():
        for level_key, metric_value in level_dict.items():
            # metric_value can be NaN here and should be preserved initially
            data_for_df.append({'perturbation': pert, 'level': level_key, 'metric_value': metric_value})
 

    df = pd.DataFrame(data_for_df)
    
    # Robustly convert 'level' to numeric, coercing errors to NaN
    df['level_numeric'] = pd.to_numeric(df['level'], errors='coerce')

    # Define level_plot_transformed based on use_log_x
    if use_log_x:
        df['level_plot_transformed'] = df['level_numeric'].apply(
            lambda x: np.log(x) / np.log(log_x_base) if pd.notna(x) and x > 0 else np.nan
        )
    else: # Linear scale
        df['level_plot_transformed'] = df['level_numeric'].astype(float) # astype(float) handles NaNs correctly

    # Drop rows only if the x-value for plotting (level_plot_transformed) is NaN.
    # NaNs in 'metric_value' will be handled by seaborn's plotting functions.
    df_plotting = df.dropna(subset=['level_plot_transformed']).copy()

    # Ensure 'metric_value' is also numeric for plotting; coercing errors will make problematic values NaN.
    df_plotting['metric_value'] = pd.to_numeric(df_plotting['metric_value'], errors='coerce')

    # Create ranks from the original numeric levels for ordinal trend analysis
    if not df_plotting['level_numeric'].dropna().empty:
        unique_original_levels = sorted(df_plotting['level_numeric'].dropna().unique())
        level_to_rank_map = {level: i for i, level in enumerate(unique_original_levels)}
        df_plotting['level_rank_for_trend'] = df_plotting['level_numeric'].map(level_to_rank_map)
    else:
        df_plotting['level_rank_for_trend'] = np.nan # Ensure column exists

    plt.clf()
    # plt.style.use('seaborn-v0_8-whitegrid') # Apply style
    plt.figure(figsize=(7, 6)) # Ensure figure is created after clf and style

    # Get current axes to operate on
    ax = plt.gca()
    
    # For debugging, print the DataFrame again after this final processing step
    # print("DataFrame being passed to stripplot:")
    # print(df_plotting.head())
    # print(df_plotting.dtypes)

    # Determine x-column and order for plotting based on use_log_x
    if use_log_x:
        x_column_for_plotting = 'level_plot_transformed' # Already log-numeric
        plot_order = sorted(df_plotting[x_column_for_plotting].dropna().unique())
    else: # Linear scale, treat x-axis as categorical for stripplot and pointplot
        unique_numeric_levels = sorted(df_plotting['level_plot_transformed'].dropna().unique())
        # Create a new column with string categories, ordered by the numeric levels
        df_plotting['x_categorical_for_plotting'] = pd.Categorical(
            df_plotting['level_plot_transformed'].astype(str),
            categories=[str(l) for l in unique_numeric_levels],
            ordered=True
        )
        x_column_for_plotting = 'x_categorical_for_plotting'
        plot_order = [str(l) for l in unique_numeric_levels]

    sns.stripplot(
        data=df_plotting, 
        x=x_column_for_plotting, # Use determined x-column
        y='metric_value', 
        order=plot_order,      # Use determined order
        alpha=0.5, 
        size=5,    
        edgecolor='w',      
        linewidth=0.5,    
        jitter=jitter_strength,
        ax=ax
    )

    # Add pointplot for mean/CI per level, styled as in plot_n_perts_categorical_scatter
    try:
        if not df_plotting.empty: # Check if there's data
            if plot_order: # Proceed only if there are levels/order to plot
                sns.pointplot(
                    data=df_plotting,
                    x=x_column_for_plotting, 
                    y='metric_value',
                    order=plot_order,
                    color='darkblue',     
                    markers="d",          
                    linestyles="--", 
                    scale=0.7,            # Added scale to reduce size
                    errorbar=('ci', 95),
                    capsize=.2,           
                    ax=ax,
                    zorder=10             
                )
    except Exception as e:
        print(f"Could not plot pointplot for {plot_title}: {e}")

    full_plot_title = f'{plot_title} ({dataset_name})'
    # plt.title(full_plot_title, fontsize=18, fontweight='bold')
    # Add 20 pts padding to title
    plt.title(full_plot_title, fontsize=18, fontweight='bold', y=1.05)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    
    # Axis ticks and limits based on whether log scale was used
    if use_log_x and not df_plotting.empty:
        actual_ticks_original_values = sorted(df_plotting['level_numeric'].dropna().unique()) # Use original numeric for labels
        actual_ticks_original_values = [t for t in actual_ticks_original_values if t > 0]

        if actual_ticks_original_values:
            # Ticks are positioned at log-transformed values
            plot_ticks_transformed = [np.log(t)/np.log(log_x_base) for t in actual_ticks_original_values]
            if plot_ticks_transformed[0] == 1:
                # Subtract 1 from all ticks
                plot_ticks_transformed = [t - 1 for t in plot_ticks_transformed]
            ax.set_xticks(plot_ticks_transformed)
            ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.1f}" for t in actual_ticks_original_values], rotation=45, ha='right', va='top', rotation_mode='anchor')
            
            if plot_ticks_transformed: # Ensure not empty
                min_transformed_tick = min(plot_ticks_transformed)
                max_transformed_tick = max(plot_ticks_transformed)
                padding = 0.5 if len(plot_ticks_transformed) > 1 else 0.5 
                ax.set_xlim(min_transformed_tick - padding, max_transformed_tick + padding)
        else:
            # No valid ticks to set, let matplotlib decide or leave blank if no data
            pass
            
    elif x_ticks_override: # Linear scale with specific ticks (numeric)
        ax.set_xticks(x_ticks_override)
        ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.1f}" for t in x_ticks_override], rotation=45, ha='right', va='top', rotation_mode='anchor')
        if x_ticks_override:
            min_tick = min(x_ticks_override)
            max_tick = max(x_ticks_override)
            padding_val = (max_tick - min_tick) * 0.1 if (max_tick - min_tick) > 0 else 1.0
            if padding_val == 0 and len(x_ticks_override) > 0: # Handle single unique tick value
                 padding_val = abs(x_ticks_override[0] * 0.1) if x_ticks_override[0] != 0 else 1.0
            ax.set_xlim(min_tick - padding_val, max_tick + padding_val)
    else: # Default linear scale (now treated as categorical for plotting elements)
        if not df_plotting.empty and x_column_for_plotting == 'x_categorical_for_plotting':
            # Ticks are original numeric values from level_plot_transformed (before categorization)
            actual_numeric_labels = sorted(df_plotting['level_plot_transformed'].dropna().unique())
            if actual_numeric_labels:
                # Positions are 0, 1, 2... for categories
                tick_positions = range(len(actual_numeric_labels))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.1f}" for t in actual_numeric_labels], rotation=45, ha='right', va='top', rotation_mode='anchor')
                
                # Adjust x-limits for categorical display
                ax.set_xlim(-0.5, len(actual_numeric_labels) - 0.5)
        # Fallback if somehow not categorical and no override (should be rare with new logic)
        elif not df_plotting.empty and pd.api.types.is_numeric_dtype(df_plotting['level_plot_transformed']):
            actual_ticks = sorted(df_plotting['level_plot_transformed'].dropna().unique())
            if actual_ticks:
                ax.set_xticks(actual_ticks)
                ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.1f}" for t in actual_ticks], rotation=45, ha='right', va='top', rotation_mode='anchor')
                min_tick = min(actual_ticks)
                max_tick = max(actual_ticks)
                padding = (max_tick - min_tick) * 0.1 if (max_tick - min_tick) > 0 else 1.0
                if padding == 0 and len(actual_ticks) > 0: 
                    padding = abs(actual_ticks[0] * 0.1) if actual_ticks[0] != 0 else 1.0
                ax.set_xlim(min_tick - padding, max_tick + padding)

    # Explicitly set Y-axis limits based on data
    if not df_plotting['metric_value'].dropna().empty and yaxis_limits is None:
        min_y = df_plotting['metric_value'].min()
        max_y = df_plotting['metric_value'].max()
        padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 1.0
        if padding_y == 0 : padding_y = abs(min_y * 0.1) if min_y != 0 else 1.0 # Handle single unique y-value or y=0
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
    elif yaxis_limits is not None:
        ax.set_ylim(yaxis_limits[0], yaxis_limits[1])
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Calculate and display linear trend statistics (R-squared and P-value from ANOVA-LM)
    # Use 'level_rank_for_trend' for regression, which is based on the original numeric order of levels
    trend_data = df_plotting[['level_rank_for_trend', 'metric_value']].dropna()
    pearson_r_value, pearson_p_val = pearsonr(trend_data['level_rank_for_trend'], trend_data['metric_value'])
    trend_text = f'Pearson R={pearson_r_value:.2f}, P={pearson_p_val:.2e}'
    ax.text(0.05, 0.97, trend_text, transform=ax.transAxes, fontsize=15, 
            verticalalignment='bottom', zorder=20)

    
    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title).rstrip()
    filepath_title = filepath_title.replace(' ', '_')
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.scatter_trend.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Scatter trend plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving scatter trend plot {save_path}: {e}")
    finally:
        plt.close()

def plot_n_perts_categorical_scatter(dict_to_plot, subdir, plot_title, dataset_name, xlabel, ylabel):
    """
    Simplified scatter plot for 'N perts' data, treating N perts as categorical.
    Shows individual points with jitter and a pointplot for mean/CI trends.
    """
    data_for_df = []
    for pert, level_dict in dict_to_plot.items():
        for level_key, metric_value in level_dict.items():
            # level_key is N_perts, metric_value is the Pearson R (can be NaN)
            data_for_df.append({'perturbation': pert, 'level': level_key, 'metric_value': metric_value})

    if not data_for_df:
        print(f"No data to plot for {plot_title}")
        return

    df = pd.DataFrame(data_for_df)

    # Ensure 'level' (N perts) is numeric for sorting, then treat as categorical for plotting
    df['level_numeric'] = pd.to_numeric(df['level'], errors='coerce')
    df = df.dropna(subset=['level_numeric']) # Remove if N perts itself is not a number
    df = df.sort_values(by='level_numeric')
    # Convert to string for stripplot/pointplot to treat as categorical if direct numeric causes issues
    # Or, ensure levels are passed to x_order if using numeric directly with stripplot/pointplot
    df['level_category'] = df['level_numeric'].astype(str) 
    # Get sorted unique categories for ordering the plot
    sorted_categories = df['level_numeric'].unique().astype(str) # Use unique numeric then convert to str for order
    
    # Filter out rows where metric_value is NaN for plotting, as stripplot/pointplot handle this.
    # df_plotting = df.dropna(subset=['metric_value']) 
    # Actually, stripplot and pointplot should handle NaNs in y (metric_value) gracefully.
    df_plotting = df # Use the full df with potential NaNs in metric_value

    if df_plotting.empty:
        print(f"No data points with valid levels to plot for {plot_title}")
        return

    plt.clf()
    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # 1. Jittered scatter plot for all points
    sns.stripplot(
        data=df_plotting,
        x='level_category', 
        y='metric_value',
        order=sorted_categories, # Ensure correct order
        jitter=0.2,
        alpha=0.5,
        size=5,
        color='steelblue',
        ax=ax
    )

    # 2. Point plot for mean and confidence intervals
    sns.pointplot(
        data=df_plotting,
        x='level_category',
        y='metric_value',
        order=sorted_categories, # Ensure correct order
        color='darkblue',      # Changed to darkblue
        markers="d",
        linestyles="--",
        errorbar=('ci', 95), 
        capsize=.2,
        ax=ax,
        dodge=True, # Dodge if other elements are on the same categorical axis
        zorder=10 # Ensure it's on top
    )

    full_plot_title = f'{plot_title} ({dataset_name})'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-labels if they are crowded
    plt.tight_layout() # Adjust layout

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title).rstrip()
    filepath_title = filepath_title.replace(' ', '_') + ".categorical"
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Categorical scatter plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving categorical scatter plot {save_path}: {e}")
    finally:
        plt.close()

def plot_categorical_scatter_trend(dict_to_plot, subdir, plot_title, dataset_name, xlabel, ylabel, jitter_strength=0.2):
    """
    Plots metrics as a scatter plot with a smoothed trend line for categorical levels.
    Aggregates data from all perturbations.
    X-axis: categorical levels (e.g., quantile ranges).
    Y-axis: metric values.
    """
    data_for_df = []
    for pert, level_dict in dict_to_plot.items():
        for level_key, metric_value in level_dict.items():
            data_for_df.append({'perturbation': pert, 'level': level_key, 'metric_value': metric_value})

    if not data_for_df:
        print(f"No data to plot for {plot_title}")
        plt.close() # Ensure plt is closed if we exit early
        return

    df = pd.DataFrame(data_for_df)
    
    # metric_value can have NaNs; stripplot/pointplot handle them.
    # Filter out rows where 'level' is NaN, if any, before sorting.
    df_plotting = df.dropna(subset=['level']).copy()

    if df_plotting.empty or df_plotting['metric_value'].isnull().all():
        print(f"No valid data points for plotting '{plot_title}' after filtering. Skipping plot.")
        plt.close()
        return

    plt.clf() # Clear figure
    # plt.style.use('seaborn-v0_8-whitegrid') # Apply style
    plt.figure(figsize=(7, 6)) # Ensure figure is created after clf and style
    ax = plt.gca()
    
    # Determine sorted categories for the x-axis
    unique_levels = df_plotting['level'].dropna().unique()
    try:
        # Attempt to sort numerically based on the start of the range (e.g., "0.0-0.1")
        plot_order = sorted(unique_levels, key=lambda x: float(str(x).split('-')[0]))
    except (ValueError, IndexError):
        # Fallback to simple string sort if conversion or split fails
        plot_order = sorted(unique_levels, key=str)

    if not plot_order:
        print(f"No categories to plot for {plot_title}. Skipping plot.")
        plt.close()
        return

    sns.stripplot(
        data=df_plotting, 
        x='level', 
        y='metric_value', 
        order=plot_order,
        alpha=0.5, 
        size=5,    
        edgecolor='w',      
        linewidth=0.5,    
        jitter=jitter_strength,
        ax=ax
    )

    try:
        if not df_plotting.empty and plot_order:
            sns.pointplot(
                data=df_plotting,
                x='level', 
                y='metric_value',
                order=plot_order,
                color='darkblue',     
                markers="d",          
                linestyles="--", 
                scale=0.7,
                errorbar=('ci', 95),
                capsize=.2,           
                ax=ax,
                zorder=10             
            )
    except Exception as e:
        print(f"Could not plot pointplot for {plot_title}: {e}")

    full_plot_title = f'{plot_title} ({dataset_name})'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    if plot_order:
        ax.set_xticks(range(len(plot_order)))
        ax.set_xticklabels(plot_order, rotation=45, ha='right')
        ax.set_xlim(-0.5, len(plot_order) - 0.5)

    if not df_plotting['metric_value'].dropna().empty:
        min_y = df_plotting['metric_value'].dropna().min()
        max_y = df_plotting['metric_value'].dropna().max()
        padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 1.0
        if padding_y == 0 and pd.notna(min_y): 
            padding_y = abs(min_y * 0.1) if min_y != 0 else 1.0
        if pd.notna(min_y) and pd.notna(max_y):
             ax.set_ylim(min_y - padding_y, max_y + padding_y)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout() 

    # Calculate and display linear trend statistics (R-squared and P-value from ANOVA-LM)
    if plot_order:
        level_to_rank = {level_val: rank for rank, level_val in enumerate(plot_order)}
        trend_df = df_plotting.copy()
        trend_df['level_rank'] = trend_df['level'].map(level_to_rank)
        trend_data_for_stat = trend_df[['level_rank', 'metric_value']].dropna()
        
        if len(trend_data_for_stat) >= 2 and trend_data_for_stat['level_rank'].nunique() >= 2:
            try:
                model = smf.ols('metric_value ~ level_rank', data=trend_data_for_stat).fit()
                anova_results = anova_lm(model, type=2)
                p_value_anova = anova_results.loc['level_rank', 'PR(>F)']
                # Calculate Pearson R using scipy.stats.pearsonr
                pearson_r_value, _ = pearsonr(trend_data_for_stat['level_rank'], trend_data_for_stat['metric_value'])
                trend_text = f'Trend (ANOVA-LM): Pearson R={pearson_r_value:.2f}, P={p_value_anova:.2e}'
                ax.text(0.02, 0.02, trend_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
            except Exception as e:
                print(f"Could not calculate ANOVA/regression for trend in '{plot_title}': {e}")
                ax.text(0.02, 0.02, 'Trend (ANOVA-LM): Error', transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
        else:
            ax.text(0.02, 0.02, 'Trend (ANOVA-LM): N/A', transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
    
    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title).rstrip()
    filepath_title = filepath_title.replace(' ', '_')
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.categorical_scatter_trend.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Categorical scatter trend plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving categorical scatter trend plot {save_path}: {e}")
    finally:
        plt.close()

def plot_pert_strength_scatter(dict_to_plot, subdir, plot_title, dataset_name, xlabel, ylabel):
    """
    Scatter plot for perturbation strength data (DEG quantile ranges).
    Treats quantile ranges as ordered categories.
    Shows individual points with jitter and a pointplot for mean/CI trends.
    Aesthetics aligned with plot_n_perts_categorical_scatter.
    """
    data_for_df = []
    # The dict_to_plot has quantile ranges as primary keys, and inner dicts are pert:metric_value
    for quantile_range, pert_metric_dict in dict_to_plot.items():
        for pert, metric_value in pert_metric_dict.items():
            data_for_df.append({
                'level': quantile_range, # This is our x-axis category, e.g., "0.0-0.2"
                'perturbation': pert,
                'metric_value': metric_value
            })

    if not data_for_df:
        print(f"No data to plot for {plot_title}")
        return

    df = pd.DataFrame(data_for_df)

    # The 'level' column (quantile ranges) is already string, sort them for order
    # A natural sort might be better if strings don't sort as expected numerically, but for "0.0-0.2" format, standard sort is usually fine.
    sorted_categories = sorted(df['level'].unique())
    
    df_plotting = df # Use the full df with potential NaNs in metric_value, seaborn handles these.

    if df_plotting.empty:
        print(f"No data points with valid levels to plot for {plot_title}")
        return

    plt.clf()
    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7, 6)) # Wider to accommodate potentially many categories
    ax = plt.gca()

    # 1. Jittered scatter plot for all points
    sns.stripplot(
        data=df_plotting,
        x='level', 
        y='metric_value',
        order=sorted_categories,
        jitter=0.2,
        alpha=0.5,
        size=5,
        color='steelblue',
        ax=ax
    )

    # 2. Point plot for mean and confidence intervals
    sns.pointplot(
        data=df_plotting,
        x='level',
        y='metric_value',
        order=sorted_categories,
        color='darkblue',
        markers="d",
        linestyles="--",
        errorbar=('ci', 95),
        capsize=.2,
        ax=ax,
        zorder=10
    )

    full_plot_title = f'{plot_title} ({dataset_name})'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Calculate and display linear trend statistics (R-squared and P-value from ANOVA-LM)
    if sorted_categories:
        level_to_rank = {level_val: rank for rank, level_val in enumerate(sorted_categories)}
        trend_df = df_plotting.copy()
        trend_df['level_rank'] = trend_df['level'].map(level_to_rank)
        trend_data_for_stat = trend_df[['level_rank', 'metric_value']].dropna()

        if len(trend_data_for_stat) >= 2 and trend_data_for_stat['level_rank'].nunique() >= 2:
            try:
                model = smf.ols('metric_value ~ level_rank', data=trend_data_for_stat).fit()
                anova_results = anova_lm(model, type=2)
                p_value_anova = anova_results.loc['level_rank', 'PR(>F)']
                # Calculate Pearson R using scipy.stats.pearsonr
                pearson_r_value, _ = pearsonr(trend_data_for_stat['level_rank'], trend_data_for_stat['metric_value'])
                trend_text = f'Trend (ANOVA-LM): Pearson R={pearson_r_value:.2f}, P={p_value_anova:.2e}'
                ax.text(0.02, 0.02, trend_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
            except Exception as e:
                print(f"Could not calculate ANOVA/regression for trend in '{plot_title}': {e}")
                ax.text(0.02, 0.02, 'Trend (ANOVA-LM): Error', transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
        else:
            ax.text(0.02, 0.02, 'Trend (ANOVA-LM): N/A', transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)
    else:
        ax.text(0.02, 0.02, 'Trend (ANOVA-LM): No x-categories', transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5), zorder=20)

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title).rstrip()
    filepath_title = filepath_title.replace(' ', '_')
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.png')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Perturbation strength scatter plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving perturbation strength scatter plot {save_path}: {e}")
    finally:
        plt.close()

def plot_n_perts_categorical_scatter_multiseed(
    data_df, subdir, plot_title, dataset_name, xlabel, ylabel
):
    """
    Plots metrics for 'N perts' data from multiple random seeds.
    Shows individual points with jitter and a pointplot for mean/CI trends, colored by seed.
    Args:
        data_df (pd.DataFrame): DataFrame directly from analysis.py, expected to contain columns:
                                'seed', 'n_perts' (convertible to numeric), 'pert', 'metric_value'.
        subdir (str): The full path to the subdirectory where the plot will be saved.
        plot_title (str): Base title for the plot.
        dataset_name (str): Name of the dataset (e.g., 'norman19').
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    if data_df.empty:
        print(f"No data to plot for {plot_title}")
        return

    df_plotting = data_df.copy() # Use the input DataFrame directly

    # Ensure 'n_perts' is numeric for sorting, then treat as categorical for plotting
    df_plotting['n_perts_numeric'] = pd.to_numeric(df_plotting['n_perts'], errors='coerce')
    
    # Drop NaNs for essential columns for plotting and stats
    df_plotting = df_plotting.dropna(subset=['n_perts_numeric', 'metric_value', 'seed'])
    
    if df_plotting.empty: # Check after dropping NaNs
        print(f"No valid data points after NaN drop for {plot_title}")
        return

    df_plotting = df_plotting.sort_values(by=['seed', 'n_perts_numeric']) # Sort for consistency
    
    df_plotting['n_perts_category'] = df_plotting['n_perts_numeric'].astype(int).astype(str) 
    sorted_categories = sorted(df_plotting['n_perts_category'].unique(), key=int)

    df_plotting['seed'] = df_plotting['seed'].astype('category')

    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 6)) 

    num_seeds = len(df_plotting['seed'].unique())
    palette = sns.color_palette("viridis", n_colors=num_seeds) if num_seeds > 0 else None

    sns.stripplot(
        data=df_plotting,
        x='n_perts_category',
        y='metric_value',
        hue='seed',
        order=sorted_categories,
        jitter=True,
        alpha=0.4, 
        size=3,    
        palette=palette,
        dodge=True,
        ax=ax
    )

    sns.pointplot(
        data=df_plotting,
        x='n_perts_category',
        y='metric_value',
        hue='seed',
        order=sorted_categories,
        markers="d",
        linestyles="--",
        errorbar=('ci', 95),
        capsize=.05, 
        scale=0.5,   
        dodge=0.5,   
        ax=ax,
        palette=palette, 
        zorder=10
    )

    full_plot_title_text = f'{plot_title} ({dataset_name})'
    plt.title(full_plot_title_text, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits to -1.1 to 1.1
    ax.set_ylim(-1.1, 1.1)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles and num_seeds > 0 and len(handles) >= num_seeds: # Check num_seeds for safety
        ax.legend(handles[:num_seeds], labels[:num_seeds], title='Seed', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    elif handles:
        ax.legend(title='Seed', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 

    # Calculate and display linear trend statistics (R-squared and P-value from ANOVA-LM) on an_metrics_per_seed_npert
    if not df_plotting.empty:
        mean_metrics_per_seed_npert = df_plotting.groupby(['seed', 'n_perts_numeric'])['metric_value'].mean().reset_index()
        trend_data = mean_metrics_per_seed_npert.dropna() 

        # Calculate Pearson R using scipy.stats.pearsonr
        pearson_r_value, pval = pearsonr(trend_data['n_perts_numeric'], trend_data['metric_value'])
        trend_text = f'Pearson R={pearson_r_value:.2f}, P={pval:.2e}'
        ax.text(0.02, 0.96, trend_text, transform=ax.transAxes, fontsize=10, 
                    verticalalignment='bottom', zorder=20)


    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title).rstrip()
    filepath_title = filepath_title.replace(' ', '_') + "_multiseed"
    
    # subdir is expected to be the full path from analysis.py
    os.makedirs(subdir, exist_ok=True) # Ensure plot directory exists
    save_path = os.path.join(subdir, f'{filepath_title}.png')
    try:
        plt.savefig(save_path, dpi=300) 
        print(f'Multi-seed categorical scatter plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving multi-seed categorical scatter plot {save_path}: {e}")
    finally:
        plt.close(fig)

def plot_mse_vs_wmse_pert_strength(mse_dict, wmse_dict, subdir, plot_title_prefix, dataset_name, xlabel):
    """
    Plots MSE and WMSE on the same plot with dual y-axes for perturbation strength data (DEG quantile ranges).
    Args:
        mse_dict (dict): Dictionary of MSE values {quantile_range: {pert: value}}.
        wmse_dict (dict): Dictionary of WMSE values {quantile_range: {pert: value}}.
        subdir (str): Subdirectory to save the plot.
        plot_title_prefix (str): Prefix for the plot title.
        dataset_name (str): Name of the dataset.
        xlabel (str): Label for the x-axis.
    """
    data_for_df = []
    for quantile_range, pert_metric_dict_mse in mse_dict.items():
        for pert, mse_value in pert_metric_dict_mse.items():
            wmse_value = wmse_dict.get(quantile_range, {}).get(pert, np.nan)
            data_for_df.append({
                'level': quantile_range,
                'perturbation': pert,
                'mse': mse_value,
                'wmse': wmse_value
            })

    if not data_for_df:
        print(f"No data to plot for {plot_title_prefix}")
        return

    df = pd.DataFrame(data_for_df)
    df_plotting = df.dropna(subset=['level', 'mse', 'wmse'], how='all').copy()

    if df_plotting.empty:
        print(f"No valid data points after NaN drop for {plot_title_prefix}")
        return

    # Sort categories for x-axis
    try:
        sorted_categories = sorted(df_plotting['level'].unique(), key=lambda x: float(str(x).split('-')[0]))
    except (ValueError, IndexError):
        sorted_categories = sorted(df_plotting['level'].unique(), key=str)

    if not sorted_categories:
        print(f"No categories to plot for {plot_title_prefix}. Skipping plot.")
        return
    
    df_plotting['level_rank'] = df_plotting['level'].apply(lambda x: sorted_categories.index(x) if x in sorted_categories else -1)

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Plot MSE on primary y-axis (ax1)
    sns.stripplot(data=df_plotting, x='level', y='mse', order=sorted_categories, jitter=0.1, alpha=0.6, size=4, color='steelblue', ax=ax1, zorder=1) 
    sns.pointplot(data=df_plotting, x='level', y='mse', order=sorted_categories, color='darkblue', markers="o", linestyles="-", errorbar=('ci', 95), capsize=.1, ax=ax1, zorder=2)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel('MSE (pert vs mean of all perts)', color='darkblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(False)

    # Create secondary y-axis for WMSE (ax2)
    ax2 = ax1.twinx()
    sns.stripplot(data=df_plotting, x='level', y='wmse', order=sorted_categories, jitter=0.1, alpha=0.6, size=4, color='lightcoral', ax=ax2, zorder=1) 
    sns.pointplot(data=df_plotting, x='level', y='wmse', order=sorted_categories, color='darkred', markers="s", linestyles="--", errorbar=('ci', 95), capsize=.1, ax=ax2, zorder=2)
    ax2.set_ylabel('WMSE (MSE weighted by DEG strength)', color='darkred', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.grid(False)
    ax2.spines["right"].set_color('darkred') 
    ax2.spines["right"].set_linewidth(ax1.spines["left"].get_linewidth())

    # Title
    full_plot_title = f'{plot_title_prefix} ({dataset_name}) - MSE vs WMSE'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')
    
    # ANOVA and Regression for MSE trend
    mse_anova_data = df_plotting[['level_rank', 'mse']].dropna()
    if len(mse_anova_data) >= 2 and mse_anova_data['level_rank'].nunique() >= 2: 
        try:
            model_mse = smf.ols('mse ~ level_rank', data=mse_anova_data).fit()
            anova_mse = anova_lm(model_mse, type=2)
            p_mse = anova_mse.loc['level_rank', 'PR(>F)']
            pearson_r_mse, _ = pearsonr(mse_anova_data['level_rank'], mse_anova_data['mse'])
            ax1.text(0.02, 0.98, f'MSE Trend (ANOVA-LM): Pearson R={pearson_r_mse:.2f}, P={p_mse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for MSE trend: {e}")
            ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
    else:
        ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)

    # ANOVA and Regression for WMSE trend
    wmse_anova_data = df_plotting[['level_rank', 'wmse']].dropna()
    if len(wmse_anova_data) >= 2 and wmse_anova_data['level_rank'].nunique() >= 2:
        try:
            model_wmse = smf.ols('wmse ~ level_rank', data=wmse_anova_data).fit()
            anova_wmse = anova_lm(model_wmse, type=2)
            p_wmse = anova_wmse.loc['level_rank', 'PR(>F)']
            pearson_r_wmse, _ = pearsonr(wmse_anova_data['level_rank'], wmse_anova_data['wmse'])
            ax1.text(0.02, 0.88, f'WMSE Trend (ANOVA-LM): Pearson R={pearson_r_wmse:.2f}, P={p_wmse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for WMSE trend: {e}")
            ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
    else:
        ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)

    fig.tight_layout()

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title_prefix).rstrip()
    filepath_title = filepath_title.replace(' ', '_') + "_MSE_vs_WMSE"
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.png')
    
    try:
        plt.savefig(save_path, dpi=300)
        print(f'MSE vs WMSE plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving MSE vs WMSE plot {save_path}: {e}")
    finally:
        plt.close(fig)

def plot_mse_vs_wmse_cell_counts(mse_dict, wmse_dict, subdir, plot_title_prefix, dataset_name, xlabel, use_log_x=False, log_x_base=10):
    """
    Plots MSE and WMSE on the same plot with dual y-axes for cell count sweeps (Step 1).
    Args:
        mse_dict (dict): Dict of MSE values {pert: {cell_count: value}}.
        wmse_dict (dict): Dict of WMSE values {pert: {cell_count: value}}.
        subdir (str): Subdirectory to save the plot.
        plot_title_prefix (str): Prefix for the plot title.
        dataset_name (str): Name of the dataset.
        xlabel (str): Label for the x-axis.
        use_log_x (bool): Whether to use a log scale for the x-axis.
        log_x_base (int): Base for the log transformation if use_log_x is True.
    """
    data_for_df = []
    for pert, level_metric_dict_mse in mse_dict.items():
        for level_key, mse_value in level_metric_dict_mse.items(): # level_key is cell_count
            wmse_value = wmse_dict.get(pert, {}).get(level_key, np.nan)
            data_for_df.append({
                'level': level_key, # cell_count
                'perturbation': pert,
                'mse': mse_value,
                'wmse': wmse_value
            })

    if not data_for_df:
        print(f"No data to plot for {plot_title_prefix}")
        return

    df = pd.DataFrame(data_for_df)
    df_plotting = df.copy()

    # Handle numeric conversion for x-axis (level = cell_count)
    df_plotting['level_numeric'] = pd.to_numeric(df_plotting['level'], errors='coerce')

    # Prepare x-axis for plotting: treat as categorical, but order numerically
    # The actual numeric values (original or log-transformed) will be used for trend calculation.
    df_plotting = df_plotting.dropna(subset=['level_numeric'])
    df_plotting['x_categorical'] = df_plotting['level_numeric'].astype(int).astype(str)
    sorted_x_categories = sorted(df_plotting['x_categorical'].unique(), key=int)
    
    # Create a numerical rank of the categories for trend calculation
    category_to_rank = {cat: i for i, cat in enumerate(sorted_x_categories)}
    df_plotting['level_rank_for_trend'] = df_plotting['x_categorical'].map(category_to_rank)

    # The use_log_x parameter will now primarily affect visual tick labeling if we re-add custom log-style ticks later.
    # For regression, we use the rank.
    # df_plotting = df_plotting.dropna(subset=['level_trend_calc', 'mse', 'wmse'], how='all') # Old line
    df_plotting = df_plotting.dropna(subset=['level_rank_for_trend', 'mse', 'wmse'], how='all')

    if df_plotting.empty:
        print(f"No valid data points after NaN drop for {plot_title_prefix}")
        return

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    x_axis_var_for_plot = 'x_categorical' # Use categorical for plotting positions

    # Plot MSE on primary y-axis (ax1)
    sns.stripplot(data=df_plotting, x=x_axis_var_for_plot, y='mse', order=sorted_x_categories, jitter=0.1, alpha=0.6, size=4, color='steelblue', ax=ax1, zorder=1) 
    sns.pointplot(data=df_plotting, x=x_axis_var_for_plot, y='mse', order=sorted_x_categories, color='darkblue', markers="o", linestyles="-", errorbar=('ci', 95), capsize=.1, ax=ax1, zorder=2)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel('MSE (pert vs mean of all perts)', color='darkblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(False)

    # Create secondary y-axis for WMSE (ax2)
    ax2 = ax1.twinx()
    sns.stripplot(data=df_plotting, x=x_axis_var_for_plot, y='wmse', order=sorted_x_categories, jitter=0.1, alpha=0.6, size=4, color='lightcoral', ax=ax2, zorder=1) 
    sns.pointplot(data=df_plotting, x=x_axis_var_for_plot, y='wmse', order=sorted_x_categories, color='darkred', markers="s", linestyles="--", errorbar=('ci', 95), capsize=.1, ax=ax2, zorder=2)
    ax2.set_ylabel('WMSE (MSE weighted by DEG strength)', color='darkred', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.grid(False)
    ax2.spines["right"].set_color('darkred')
    ax2.spines["right"].set_linewidth(ax1.spines["left"].get_linewidth())

    # Title
    full_plot_title = f'{plot_title_prefix} ({dataset_name}) - MSE vs WMSE'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')

    # ANOVA and Regression for MSE trend
    # Use 'level_rank_for_trend' for the regression model
    mse_trend_data = df_plotting[['level_rank_for_trend', 'mse']].dropna()
    if len(mse_trend_data) >= 2 and mse_trend_data['level_rank_for_trend'].nunique() >= 2:
        try:
            model_mse = smf.ols('mse ~ level_rank_for_trend', data=mse_trend_data).fit()
            anova_mse = anova_lm(model_mse, type=2)
            p_mse = anova_mse.loc['level_rank_for_trend', 'PR(>F)']
            pearson_r_mse, _ = pearsonr(mse_trend_data['level_rank_for_trend'], mse_trend_data['mse'])
            ax1.text(0.02, 0.98, f'MSE Trend (ANOVA-LM): Pearson R={pearson_r_mse:.2f}, P={p_mse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for MSE trend in '{plot_title_prefix}': {e}")
            ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
    else:
        ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)

    # ANOVA and Regression for WMSE trend
    # Use 'level_rank_for_trend' for the regression model
    wmse_trend_data = df_plotting[['level_rank_for_trend', 'wmse']].dropna()
    if len(wmse_trend_data) >= 2 and wmse_trend_data['level_rank_for_trend'].nunique() >= 2:
        try:
            model_wmse = smf.ols('wmse ~ level_rank_for_trend', data=wmse_trend_data).fit()
            anova_wmse = anova_lm(model_wmse, type=2)
            p_wmse = anova_wmse.loc['level_rank_for_trend', 'PR(>F)']
            pearson_r_wmse, _ = pearsonr(wmse_trend_data['level_rank_for_trend'], wmse_trend_data['wmse'])
            ax1.text(0.02, 0.88, f'WMSE Trend (ANOVA-LM): Pearson R={pearson_r_wmse:.2f}, P={p_wmse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for WMSE trend in '{plot_title_prefix}': {e}")
            ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
    else:
        ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)

    fig.tight_layout()

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title_prefix).rstrip()
    filepath_title = filepath_title.replace(' ', '_') + "_MSE_vs_WMSE_cells"
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.png')
    
    try:
        plt.savefig(save_path, dpi=300)
        print(f'MSE vs WMSE plot (cell counts) saved to {save_path}')
    except Exception as e:
        print(f"Error saving MSE vs WMSE plot (cell counts) {save_path}: {e}")
    finally:
        plt.close(fig)

def plot_mse_vs_wmse_categorical(mse_dict, wmse_dict, subdir, plot_title_prefix, dataset_name, xlabel):
    """
    Plots MSE and WMSE on the same plot with dual y-axes for categorical levels (Step 9 - Theta).
    Args:
        mse_dict (dict): Dict of MSE values {pert: {category_key: value}}.
        wmse_dict (dict): Dict of WMSE values {pert: {category_key: value}}.
        subdir (str): Subdirectory to save the plot.
        plot_title_prefix (str): Prefix for the plot title.
        dataset_name (str): Name of the dataset.
        xlabel (str): Label for the x-axis.
    """
    data_for_df = []
    for pert, level_metric_dict_mse in mse_dict.items():
        for level_key, mse_value in level_metric_dict_mse.items(): # level_key is quantile_string
            wmse_value = wmse_dict.get(pert, {}).get(level_key, np.nan)
            data_for_df.append({
                'level': level_key, # quantile_string
                'perturbation': pert,
                'mse': mse_value,
                'wmse': wmse_value
            })

    if not data_for_df:
        print(f"No data to plot for {plot_title_prefix}")
        return

    df = pd.DataFrame(data_for_df)
    df_plotting = df.dropna(subset=['level', 'mse', 'wmse'], how='all').copy()

    if df_plotting.empty:
        print(f"No valid data points after NaN drop for {plot_title_prefix}")
        return

    # Sort categories for x-axis (e.g., "0.0-0.1", "0.1-0.2", ...)
    try:
        sorted_categories = sorted(df_plotting['level'].unique(), key=lambda x: float(str(x).split('-')[0]))
    except (ValueError, IndexError):
        sorted_categories = sorted(df_plotting['level'].unique(), key=str) # Fallback sort

    if not sorted_categories:
        print(f"No categories to plot for {plot_title_prefix}. Skipping plot.")
        return
    
    # Create a numerical rank for the categorical levels for regression
    df_plotting['level_rank'] = df_plotting['level'].apply(lambda x: sorted_categories.index(x) if x in sorted_categories else -1)

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Plot MSE on primary y-axis (ax1)
    sns.stripplot(data=df_plotting, x='level', y='mse', order=sorted_categories, jitter=0.1, alpha=0.6, size=4, color='steelblue', ax=ax1, zorder=1)
    sns.pointplot(data=df_plotting, x='level', y='mse', order=sorted_categories, color='darkblue', markers="o", linestyles="-", errorbar=('ci', 95), capsize=.1, ax=ax1, zorder=2)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel('MSE (pert vs mean of all perts)', color='darkblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(False)

    # Create secondary y-axis for WMSE (ax2)
    ax2 = ax1.twinx()
    sns.stripplot(data=df_plotting, x='level', y='wmse', order=sorted_categories, jitter=0.1, alpha=0.6, size=4, color='lightcoral', ax=ax2, zorder=1)
    sns.pointplot(data=df_plotting, x='level', y='wmse', order=sorted_categories, color='darkred', markers="s", linestyles="--", errorbar=('ci', 95), capsize=.1, ax=ax2, zorder=2)
    ax2.set_ylabel('WMSE (MSE weighted by DEG strength)', color='darkred', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.grid(False)
    ax2.spines["right"].set_color('darkred')
    ax2.spines["right"].set_linewidth(ax1.spines["left"].get_linewidth())

    # Title
    full_plot_title = f'{plot_title_prefix} ({dataset_name}) - MSE vs WMSE'
    plt.title(full_plot_title, fontsize=14, fontweight='bold')

    # ANOVA and Regression for MSE trend
    # Use 'level_rank' for the regression model
    mse_trend_data = df_plotting[['level_rank', 'mse']].dropna()
    if len(mse_trend_data) >= 2 and mse_trend_data['level_rank'].nunique() >= 2:
        try:
            model_mse = smf.ols('mse ~ level_rank', data=mse_trend_data).fit()
            anova_mse = anova_lm(model_mse, type=2)
            p_mse = anova_mse.loc['level_rank', 'PR(>F)']
            pearson_r_mse, _ = pearsonr(mse_trend_data['level_rank'], mse_trend_data['mse'])
            ax1.text(0.02, 0.98, f'MSE Trend (ANOVA-LM): Pearson R={pearson_r_mse:.2f}, P={p_mse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for MSE trend in '{plot_title_prefix}': {e}")
            ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)
    else:
        ax1.text(0.02, 0.98, 'MSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkblue', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkblue'), zorder=20)

    # ANOVA and Regression for WMSE trend
    # Use 'level_rank' for the regression model
    wmse_trend_data = df_plotting[['level_rank', 'wmse']].dropna()
    if len(wmse_trend_data) >= 2 and wmse_trend_data['level_rank'].nunique() >= 2:
        try:
            model_wmse = smf.ols('wmse ~ level_rank', data=wmse_trend_data).fit()
            anova_wmse = anova_lm(model_wmse, type=2)
            p_wmse = anova_wmse.loc['level_rank', 'PR(>F)']
            pearson_r_wmse, _ = pearsonr(wmse_trend_data['level_rank'], wmse_trend_data['wmse'])
            ax1.text(0.02, 0.88, f'WMSE Trend (ANOVA-LM): Pearson R={pearson_r_wmse:.2f}, P={p_wmse:.2e}', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
        except Exception as e:
            print(f"Could not calculate ANOVA/regression for WMSE trend in '{plot_title_prefix}': {e}")
            ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): Error', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)
    else:
        ax1.text(0.02, 0.88, 'WMSE Trend (ANOVA-LM): N/A', transform=ax1.transAxes, fontsize=9, color='darkred', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7, ec='darkred'), zorder=20)

    fig.tight_layout()

    filepath_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "" for c in plot_title_prefix).rstrip()
    filepath_title = filepath_title.replace(' ', '_') + "_MSE_vs_WMSE_categorical"
    
    full_subdir_path = os.path.join(ANALYSIS_DIR, subdir)
    os.makedirs(full_subdir_path, exist_ok=True)
    save_path = os.path.join(full_subdir_path, f'{filepath_title}.png')
    
    try:
        plt.savefig(save_path, dpi=300)
        print(f'MSE vs WMSE categorical plot saved to {save_path}')
    except Exception as e:
        print(f"Error saving MSE vs WMSE categorical plot {save_path}: {e}")
    finally:
        plt.close(fig)

