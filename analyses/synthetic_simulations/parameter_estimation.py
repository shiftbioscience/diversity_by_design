# %% [markdown]
# # Estimate the control negative binomial distributions of the Norman19 Dataset

# %%
# Imports and and reading of the data
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import os


# Set the random seed for reproducibility
np.random.seed(42)

# Local or global handling of data
local_run = False
cache_path = "../" if local_run else ""

# Read the data
# adata = sc.read_h5ad("data/norman19/norman19_processed.h5ad")
adata = sc.read_h5ad(cache_path + "data/norman19/norman19_processed.h5ad")

# Read the list of norman19 genes to use
print("Reading norman19 genes list...")
norman19_genes = pd.read_csv(cache_path + "data/norman19/norman19_genes.csv.gz", compression='gzip', index_col=0)
norman19_gene_ids = set(norman19_genes.index)

# Filter the dataset to only include the selected genes
print(f"Before filtering: {adata.n_vars} genes")
adata = adata[:, adata.var_names.isin(norman19_gene_ids)].copy()
print(f"After filtering: {adata.n_vars} genes")

# Create three datasets: control, perturbed, and all cells
adata_control = adata[adata.obs["condition"] == "control"].copy()
adata_perturbed = adata[adata.obs["condition"] != "control"].copy()
adata_all = adata.copy()  # All cells dataset

print(f"Control dataset: {adata_control.n_obs} cells, {adata_control.n_vars} genes")
print(f"Perturbed dataset: {adata_perturbed.n_obs} cells, {adata_perturbed.n_vars} genes")
print(f"All dataset: {adata_all.n_obs} cells, {adata_all.n_vars} genes")


# %%


def fit_negative_binomial(counts):
    """
    Fit a Negative Binomial distribution to count data and return parameters.
    
    Parameters:
    -----------
    counts : array-like
        1D array of count data to fit
        
    Returns:
    --------
    dict
        Dictionary containing fitted parameters:
        - 'mu': mean parameter of the negative binomial
        - 'alpha': dispersion parameter
        - 'n': size parameter for scipy.stats.nbinom
        - 'p': probability parameter for scipy.stats.nbinom
    """
    
    # Ensure counts is a 1D numpy array
    if hasattr(counts, "toarray"):  # Check if it's a sparse matrix
        counts_array = counts.toarray().flatten()
    else:  # If it's already dense
        counts_array = np.asarray(counts).flatten()
    
    # Set up exogenous variable (just an intercept)
    exog = np.ones(len(counts_array))
    
    # Initial parameters
    initial_log_mu = np.log(np.mean(counts_array) + 1e-8)  # Add epsilon for stability
    start_params = [initial_log_mu, 1.0]
    
    # Fit the model
    try:
        nb_model = sm.NegativeBinomial(counts_array, exog, loglike_method='nb2')
        nb_results = nb_model.fit(start_params=start_params, disp=False)
        
        # Extract parameters
        log_mu_fit = nb_results.params[0]
        mu_fit = np.exp(log_mu_fit)
        alpha_fit = nb_results.params[1]
        
        # Ensure alpha is positive for calculation
        alpha_fit_calc = max(alpha_fit, 1e-6)
        
        # Calculate parameters for scipy.stats.nbinom
        n_scipy = 1 / alpha_fit_calc
        p_scipy = 1 / (1 + alpha_fit_calc * mu_fit)
        
        # Prepare return values
        params = {
            'mu': mu_fit,
            'alpha': alpha_fit,
            'n': n_scipy,
            'p': p_scipy
        }
        
        return params
    
    except Exception as e:
        return {'error': str(e)}


def fit_all_negative_binomials(adata, layer='counts', plot_path=None):
    """
    Fit negative binomial distributions to all genes in an AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression data
    layer : str, default='counts'
        Layer in AnnData to use for fitting
    plot_path : str, optional
        If provided, create a folder at this path and save parameter distribution plots
        and a grid of sample gene fits
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with fitted parameters for all genes
    """
    
    # Create a dictionary to store all fitted parameters
    all_fitted_params = {}
    
    # Always use tqdm for progress bar
    for gene_idx in tqdm(range(adata.n_vars)):
        gene_name = adata.var_names[gene_idx]
        gene_data = adata.layers[layer][:, gene_idx]
        
        # Fit negative binomial without plotting
        params = fit_negative_binomial(gene_data)
        all_fitted_params[gene_name] = params
    
    # Convert to DataFrame
    params_df = pd.DataFrame.from_dict(all_fitted_params, orient='index')
    
    # Generate plots if plot_path is provided
    if plot_path:
        # Create directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        
        # Create a 2x2 subplot figure for parameter distributions
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Size parameter (n) - top left
        sns.histplot(params_df['n'], bins=50, kde=True, log_scale=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Negative Binomial Size Parameter (n)')
        
        # Mean parameter (mu) - top right
        sns.histplot(params_df['mu'], bins=50, kde=True, log_scale=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Negative Binomial Mean Parameter (μ)')
        
        # Probability parameter (p) - bottom left
        sns.histplot(params_df['p'], bins=50, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Negative Binomial Probability Parameter (p)')
        
        # Alpha parameter - bottom right
        sns.histplot(params_df['alpha'], bins=50, kde=True, log_scale=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Negative Binomial Alpha Parameter (α)')
        
        # Apply basic styling to all plots
        for ax in axes.flatten():
            sns.despine(ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'parameter_histograms.png'), dpi=300)
        plt.close()
        
        # Create a grid of plots for 20 randomly selected genes
        # Select 20 random genes
        if len(params_df) >= 20:
            random_genes = np.random.choice(params_df.index, size=20, replace=False)
            # Order by mu parameter
            random_genes_df = params_df.loc[random_genes].sort_values('mu', ascending=False)
            selected_genes = random_genes_df.index
        else:
            # If fewer than 20 genes, use all available genes
            selected_genes = params_df.index.sort_values(key=lambda x: params_df.loc[x, 'mu'], ascending=False)
        
        # Create a 4x5 grid for the selected genes
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, gene_name in enumerate(selected_genes):
            gene_idx = np.where(adata.var_names == gene_name)[0][0]
            gene_data = adata.layers[layer][:, gene_idx]
            
            # Ensure gene_data is a 1D numpy array
            if hasattr(gene_data, "toarray"):  # Check if it's a sparse matrix
                gene_counts = gene_data.toarray().flatten()
            else:  # If it's already dense
                gene_counts = np.asarray(gene_data).flatten()
            
            # Get the parameters
            params = params_df.loc[gene_name]
            n = params['n']
            p = params['p']
            mu = params['mu']
            alpha = params['alpha']
            
            # Determine plot range
            max_count = int(np.max(gene_counts))
            x_plot = np.arange(0, max_count + 1)
            
            # Calculate PMF
            pmf_fitted = nbinom.pmf(x_plot, n, p)
            
            # Plot histogram on the corresponding axis
            ax = axes[i]
            ax.hist(gene_counts, bins=np.arange(0, max_count + 2) - 0.5,
                   density=True, alpha=0.7, color='skyblue', 
                   edgecolor='black', label='Observed Counts')
            
            # Plot fitted PMF
            ax.plot(x_plot, pmf_fitted, 'ro-', 
                   label=f'Fitted NB (μ={mu:.2f}, α={alpha:.2f})', 
                   markersize=5)
            
            ax.set_xlabel('Count')
            ax.set_ylabel('Density')
            ax.set_title(f'Random gene: {i}')
            ax.legend(fontsize='small')
            sns.despine(ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, 'gene_fits.png'), dpi=300)
        plt.close()
    
    return params_df

results_path = 'parameter_estimation' if local_run else 'analyses/synthetic_simulations/parameter_estimation'

# Fit negative binomials for all genes in control data
print("\nFitting negative binomials for control cells...")
all_fitted_params_df_control = fit_all_negative_binomials(
    adata_control, 
    layer='counts', 
    plot_path=results_path + "/control_parameter_estimation"
)

# Fit negative binomials for all genes in perturbed data
print("\nFitting negative binomials for perturbed cells...")
all_fitted_params_df_perturbed = fit_all_negative_binomials(
    adata_perturbed, 
    layer='counts', 
    plot_path=results_path + "/perturbed_parameter_estimation"
)

# Fit negative binomials for all genes in all cells combined
print("\nFitting negative binomials for all cells combined...")
all_fitted_params_df_all = fit_all_negative_binomials(
    adata_all, 
    layer='counts', 
    plot_path=results_path + "/all_parameter_estimation"
)

# Save the fitted parameters
all_fitted_params_df_control.to_csv(results_path + "/control_fitted_params.csv")
all_fitted_params_df_perturbed.to_csv(results_path + "/perturbed_fitted_params.csv")
all_fitted_params_df_all.to_csv(results_path + "/all_fitted_params.csv")

# %% Plot library size distributions for control vs perturbed vs all cells
# Calculate library sizes (total counts per cell)
control_lib_sizes = adata_control.layers['counts'].sum(axis=1)
perturbed_lib_sizes = adata_perturbed.layers['counts'].sum(axis=1)
all_lib_sizes = adata_all.layers['counts'].sum(axis=1)

# Convert to dense arrays if needed
if hasattr(control_lib_sizes, "toarray"):
    control_lib_sizes = control_lib_sizes.toarray().flatten()
if hasattr(perturbed_lib_sizes, "toarray"):
    perturbed_lib_sizes = perturbed_lib_sizes.toarray().flatten()
if hasattr(all_lib_sizes, "toarray"):
    all_lib_sizes = all_lib_sizes.toarray().flatten()

# Create directory for plots if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Calculate statistics in log space
control_log = np.log10(control_lib_sizes + 1)  # Adding 1 to avoid log(0)
perturbed_log = np.log10(perturbed_lib_sizes + 1)
all_log = np.log10(all_lib_sizes + 1)

control_log_mean = np.mean(control_log)
control_log_std = np.std(control_log)
perturbed_log_mean = np.mean(perturbed_log)
perturbed_log_std = np.std(perturbed_log)
all_log_mean = np.mean(all_log)
all_log_std = np.std(all_log)

# Use distinct colors
control_color = 'blue'
perturbed_color = 'red'
all_color = 'green'

# Create a single 2x2 grid figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Add super title with statistics
fig.suptitle(
    f'Library Size Distributions\n'
    f'Control: log10_mean={control_log_mean:.4f}, log10_std={control_log_std:.4f}\n'
    f'Perturbed: log10_mean={perturbed_log_mean:.4f}, log10_std={perturbed_log_std:.4f}\n'
    f'All: log10_mean={all_log_mean:.4f}, log10_std={all_log_std:.4f}', 
    fontsize=16, y=0.98
)

# 1. Histogram with KDE in regular scale (top-left)
# First create the histogram
axes[0, 0].hist(control_lib_sizes, bins=50, alpha=0.4, color=control_color, 
              label='Control', density=True)
axes[0, 0].hist(perturbed_lib_sizes, bins=50, alpha=0.4, color=perturbed_color, 
              label='Perturbed', density=True)
axes[0, 0].hist(all_lib_sizes, bins=50, alpha=0.4, color=all_color, 
              label='All', density=True)

axes[0, 0].set_title('Library Sizes (Regular Scale)', fontsize=14)
axes[0, 0].set_xlabel('Library Size (Total Counts)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].legend()
sns.despine(ax=axes[0, 0])

# 2. Histogram with KDE in log scale (top-right)
# Convert to log scale for manual plotting
log_control = np.log10(control_lib_sizes + 1)
log_perturbed = np.log10(perturbed_lib_sizes + 1)
log_all = np.log10(all_lib_sizes + 1)

# Create histograms
axes[0, 1].hist(log_control, bins=50, alpha=0.4, color=control_color, 
              label='Control', density=True)
axes[0, 1].hist(log_perturbed, bins=50, alpha=0.4, color=perturbed_color, 
              label='Perturbed', density=True)
axes[0, 1].hist(log_all, bins=50, alpha=0.4, color=all_color, 
              label='All', density=True)

axes[0, 1].set_title('Library Sizes (Log Scale)', fontsize=14)
axes[0, 1].set_xlabel('Library Size (Total Counts, Log Scale)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].legend()
sns.despine(ax=axes[0, 1])

# 3. KDE plot in regular scale (bottom-left)
sns.kdeplot(data=control_lib_sizes, label='Control', fill=True, alpha=0.3, ax=axes[1, 0], palette=[control_color])
sns.kdeplot(data=perturbed_lib_sizes, label='Perturbed', fill=True, alpha=0.3, ax=axes[1, 0], palette=[perturbed_color])
sns.kdeplot(data=all_lib_sizes, label='All', fill=True, alpha=0.3, ax=axes[1, 0], palette=[all_color])

axes[1, 0].set_title('KDE of Library Sizes (Regular Scale)', fontsize=14)
axes[1, 0].set_xlabel('Library Size (Total Counts)', fontsize=12)
axes[1, 0].set_ylabel('Density', fontsize=12)
axes[1, 0].legend()
sns.despine(ax=axes[1, 0])

# 4. KDE plot with log scale (bottom-right)
# For log scale, need to use the log-transformed data for proper KDE
sns.kdeplot(data=log_control, label='Control', fill=True, alpha=0.3, ax=axes[1, 1], palette=[control_color])
sns.kdeplot(data=log_perturbed, label='Perturbed', fill=True, alpha=0.3, ax=axes[1, 1], palette=[perturbed_color])
sns.kdeplot(data=log_all, label='All', fill=True, alpha=0.3, ax=axes[1, 1], palette=[all_color])

axes[1, 1].set_title('KDE of Library Sizes (Log Scale)', fontsize=14)
axes[1, 1].set_xlabel('Library Size (Total Counts, Log Scale)', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].legend()
sns.despine(ax=axes[1, 1])

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
plt.savefig(results_path + "/library_size_distributions.png", dpi=300)
plt.close()

# Create a parameter comparison plot between the three datasets
print("\nCreating parameter comparison plots...")

# Prepare combined data for plotting
params = ['mu', 'n', 'alpha', 'p']
combined_data = []

for param in params:
    # Control data
    control_df = pd.DataFrame({
        'Value': all_fitted_params_df_control[param],
        'Parameter': param,
        'Dataset': 'Control'
    })
    
    # Perturbed data
    perturbed_df = pd.DataFrame({
        'Value': all_fitted_params_df_perturbed[param],
        'Parameter': param,
        'Dataset': 'Perturbed'
    })
    
    # All data
    all_df = pd.DataFrame({
        'Value': all_fitted_params_df_all[param],
        'Parameter': param,
        'Dataset': 'All'
    })
    
    combined_data.append(control_df)
    combined_data.append(perturbed_df)
    combined_data.append(all_df)

comparison_df = pd.concat(combined_data)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Parameter Distribution Comparison', fontsize=20, y=0.98)

for i, param in enumerate(params):
    ax = axes[i//2, i%2]
    
    # Filter for this parameter
    param_data = comparison_df[comparison_df['Parameter'] == param]
    
    # Determine if we need log scale
    use_log = param in ['mu', 'n', 'alpha']
    
    # Create appropriate plot
    if use_log:
        # Add small value to ensure all values are positive for log scale
        param_data['Value'] = param_data['Value'].apply(lambda x: max(x, 1e-10))
        
        sns.boxplot(x='Dataset', y='Value', data=param_data, ax=ax)
        ax.set_yscale('log')
        ax.set_title(f'{param} Parameter Distribution (Log Scale)', fontsize=14)
    else:
        sns.boxplot(x='Dataset', y='Value', data=param_data, ax=ax)
        ax.set_title(f'{param} Parameter Distribution', fontsize=14)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    sns.despine(ax=ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(results_path + "/parameter_comparison.png", dpi=300)
plt.close()

print("Parameter estimation complete. Results saved to", results_path)
