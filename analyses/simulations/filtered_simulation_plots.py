import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import argparse
from scipy import stats

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


def analyze_metrics_filtered(csv_path, filter_dict=None, output_folder='filtered_sweep_results_plots'):
    """
    Analyze metrics with parameter filtering capability.
    
    Args:
        csv_path (str): Path to the CSV file with sweep results
        filter_dict (dict, optional): Dictionary of parameters to filter on.
                                      Format: {'param_name': [min_value, max_value]}
                                      Only rows where min_value <= param <= max_value will be included
        output_folder (str): Folder to save the plots to
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Remove failed trials if any
    if 'status' in df.columns:
        df = df[df['status'] == 'success']
    
    # Apply filters if provided
    if filter_dict is not None:
        for param, range_vals in filter_dict.items():
            if param in df.columns:
                if len(range_vals) != 2:
                    print(f"Warning: Filter for {param} needs exactly 2 values [min, max]. Skipping this filter.")
                    continue
                min_val, max_val = range_vals
                df = df[(df[param] >= min_val) & (df[param] <= max_val)]
    
    # Check if we have data after filtering
    if len(df) == 0:
        print("No data left after applying filters. Please check your filter criteria.")
        return
    
    print(f"Analysis running on {len(df)} rows after filtering.")
    
    # Create output folder (overwrite if exists)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Define parameters and metrics
    parameters = ['G', 'N0', 'Nk', 'P', 'p_effect', 'effect_factor', 'B', 'mu_l']
    
    # Organize metrics by type and case
    metric_types = ['pearson', 'mae', 'mse']
    metric_labels = ['Pearson Delta', 'MAE (log)', 'MSE (log)']  # Updated labels to indicate log scale
    cases = ['all', 'affected', 'degs']
    
    # Set style for all plots - use ticks style with no grid
    sns.set_style("ticks")
    # Set blue color palette
    sns.set_palette("Blues_d")
    blue_palette = sns.color_palette("Blues", 5)
    
    # Define specific scaling types for each parameter as requested
    log_scale_params = {
        'G': False,         # Linear
        'N0': True,         # Log
        'Nk': True,         # Log
        'P': True,          # Log
        'p_effect': False,  # Linear
        'effect_factor': False,  # Linear
        'B': False,         # Linear
        'mu_l': True        # Log
    }
    
    # Determine y-axis limits for each metric type
    y_limits = {}
    
    # Set fixed limits for pearson (-1 to 1)
    y_limits['pearson'] = (0.0, 1.0)
    
    # Determine limits for MAE and MSE based on data (for log scale)
    for metric_type in ['mae', 'mse']:
        # Get all metrics of this type
        metrics_of_type = [f'{metric_type}_{case}_median' for case in cases]
        # Find min and max across all metrics of this type
        all_values = df[metrics_of_type].values.flatten()
        all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
        all_values = all_values[all_values > 0]  # Remove zeros for log scale
        
        if len(all_values) > 0:
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            # For log scale, give a bit more range
            y_limits[metric_type] = (min_val * 0.8, max_val * 1.2)
        else:
            # Fallback if no valid values
            y_limits[metric_type] = (0.001, 1.0)
    
    # Create one plot per parameter
    for param in parameters:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'Relationship between {param} and All Metrics', fontsize=20, y=0.98)
        
        # Get the specified scale type for this parameter
        use_log = log_scale_params[param]
        
        # Loop through each metric type and case combination
        for i, metric_type in enumerate(metric_types):
            for j, case in enumerate(cases):
                ax = axes[i, j]
                metric = f'{metric_type}_{case}_median'
                
                # Create scatter plot
                sns.scatterplot(x=param, y=metric, data=df, ax=ax, alpha=0.3, color=blue_palette[3], s=20)
                
                # Add LOWESS smoothing curve with shading for confidence interval
                sns.regplot(x=param, y=metric, data=df, ax=ax, scatter=False, 
                          lowess=True, ci=95, line_kws={'color': blue_palette[4], 'lw': 2})
                
                # Apply log scale to x-axis if specified
                if use_log:
                    ax.set_xscale('log')
                    ax.set_xlabel(f'{param} (log scale)')
                    
                    # Calculate correlation using log-transformed data for annotation
                    corr, p_value = stats.spearmanr(np.log10(df[param]), df[metric])
                else:
                    ax.set_xlabel(param)
                    
                    # Calculate correlation for annotation
                    corr, p_value = stats.spearmanr(df[param], df[metric])
                
                # Apply log scale to y-axis for MAE and MSE metrics
                if metric_type in ['mae', 'mse']:
                    ax.set_yscale('log')
                
                # Add correlation coefficient to the plot
                ax.annotate(f'r = {corr:.2f}, p = {p_value:.4f}', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           ha='left', va='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc=blue_palette[1], alpha=0.3))
                
                # Set consistent y-axis limits for all plots in the same row
                if metric_type == 'pearson':
                    ax.set_ylim(y_limits[metric_type])
                else:
                    # For log scale, we need to set limits after setting scale
                    ax.set_ylim(y_limits[metric_type])
                
                # Set titles for rows and columns
                if j == 0:  # First column - only place with y-axis labels
                    ax.set_ylabel(f'{metric_labels[i]}', fontsize=14)
                else:
                    # Remove y-labels for non-first columns
                    ax.set_ylabel('')
                    
                if i == 0:  # First row
                    ax.set_title(f'{case.capitalize()}', fontsize=14)
                
                # Turn off grid
                ax.grid(False)
                
                # Despine the plot
                sns.despine(ax=ax)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the plot
        plt.savefig(os.path.join(output_folder, f'{param}_vs_all_metrics.png'), dpi=300)
        plt.close()
    
    # Create a parameter vs metrics correlation heatmap
    # Get all metric columns
    metric_columns = [f'{t}_{c}_median' for t in metric_types for c in cases]
    
    # Create correlation matrix between parameters and metrics only
    corr_matrix = np.zeros((len(parameters), len(metric_columns)))
    
    for i, param in enumerate(parameters):
        param_data = df[param]
        # Apply log transformation for parameters that benefit from it
        if log_scale_params[param]:
            param_data = np.log10(param_data)
        
        for j, metric in enumerate(metric_columns):
            metric_data = df[metric]
            # Apply log transformation for error metrics (mae, mse)
            if 'mae_' in metric or 'mse_' in metric:
                metric_data = np.log10(metric_data)
            
            # Calculate Spearman correlation
            corr, _ = stats.spearmanr(param_data, metric_data)
            corr_matrix[i, j] = corr
    
    # Create a DataFrame for better visualization
    corr_df = pd.DataFrame(corr_matrix, index=parameters, columns=metric_columns)
    
    # Plot as a clustermap with a divergent colormap
    g = sns.clustermap(corr_df, annot=True, cmap='bwr', fmt='.2f', linewidths=.5, 
                    center=0, vmin=-1, vmax=1, figsize=(15, 12),
                    cbar_kws={'label': 'Spearman Correlation Coefficient'})
    
    # Add title
    plt.suptitle('Spearman Correlation Clustermap: Parameters vs Metrics (with log transforms)', 
                fontsize=16, y=1.02)
    
    plt.savefig(os.path.join(output_folder, 'correlation_clustermap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save filtering information to a text file
    if filter_dict:
        with open(os.path.join(output_folder, 'filter_info.txt'), 'w') as f:
            f.write("Applied filters (ranges):\n")
            for param, range_vals in filter_dict.items():
                if len(range_vals) == 2:
                    f.write(f"{param}: {range_vals[0]} to {range_vals[1]}\n")
            f.write(f"\nTotal data points after filtering: {len(df)}")
    
    print(f"Analysis complete. Results saved to {output_folder}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results with filtering")
    parser.add_argument("csv_path", help="Path to the CSV file with sweep results")
    parser.add_argument("--output", default="simulations/filtered_sweep_results_plots", 
                        help="Output folder for plots (default: simulations/filtered_sweep_results_plots)")
    parser.add_argument("--filter", nargs='+', default=None,
                        help="Filter parameters in format: param:min,max (where min <= param <= max)")
    
    args = parser.parse_args()
    
    # Replace @ with the actual path if provided as an argument
    csv_path = args.csv_path
    if csv_path.startswith('@'):
        csv_path = csv_path.replace('@', 'simulations/random_sweep_results/')
    
    # Parse filter arguments
    filter_dict = {}
    if args.filter:
        for filter_str in args.filter:
            try:
                param, values_str = filter_str.split(':')
                values = [float(v) if '.' in v else int(v) for v in values_str.split(',')]
                
                if len(values) != 2:
                    print(f"Warning: Filter for {param} needs exactly 2 values [min, max]. Format: param:min,max")
                    continue
                    
                filter_dict[param] = values
            except ValueError:
                print(f"Warning: Couldn't parse filter string: {filter_str}. Format should be param:min,max")
    
    analyze_metrics_filtered(csv_path, filter_dict, args.output) 