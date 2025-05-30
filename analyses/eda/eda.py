
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import r2_score # Import r2_score


# Set matplotlib parameters to create professional plots similar to R's cowplot package
plt.style.use(["science", "nature"])
plt.rcParams.update({
    "text.usetex": False,
})

# Set matplotlib parameters to create professional plots similar to R's cowplot package
plt.rcParams.update({
    # Figure aesthetics
    'figure.facecolor': 'white',
    'figure.figsize': (4, 3),
    'figure.dpi': 150,
    
    # Text properties
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    
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
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    
    # Saving properties
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

np.random.seed(42) # Initial seed for the whole script, if desired for other parts

# Set up a small argparse to select the dataset to use
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['norman19', 'replogle22'])
args = parser.parse_args()


DATA_CACHE_DIR = f'../../../data/'
if args.dataset == 'norman19':
    data_path = f'{DATA_CACHE_DIR}/norman19/norman19_processed.h5ad'
    ANALYSIS_DIR = './norman19'
    DATASET_NAME = 'norman19'
    DATASET_CELL_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256]
    DATASET_PERTS_TO_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 175]
elif args.dataset == 'replogle22':
    data_path = f'{DATA_CACHE_DIR}/replogle22/replogle22_processed.h5ad'
    ANALYSIS_DIR = './replogle22'
    DATASET_NAME = 'replogle22'
    DATASET_CELL_COUNTS = [2, 4, 8, 16, 32, 64]
    DATASET_PERTS_TO_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1334]

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Clear all png files in this directory
os.system('find ' + ANALYSIS_DIR + ' -type f -name "*.png" -exec rm -f {} \;')


#### 0. Preliminary: Set up the metrics MAE, MSE, Pearson Delta.

## Load the data
adata = sc.read_h5ad(data_path)

def get_pert_means(adata):
    # Get the meann vector for each perturbation
    perturbations = adata.obs['condition'].unique()
    pert_means = {}
    for pert in tqdm(perturbations):
        pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
        pert_counts = adata[pert_cells].X.toarray()
        pert_means[pert] = np.mean(pert_counts, axis=0)
    return pert_means

## Set up the metrics MAE, MSE, Pearson Delta.
# MAE
def mae(x1, x2):
    return np.mean(np.abs(x1 - x2))

# MSE
def mse(x1, x2):
    return np.mean((x1 - x2) ** 2)

# Weighted MSE (WMSE)
def wmse(x1, x2, weights):
    # Ensure weights are a numpy array for vectorized operations
    weights_arr = np.array(weights)
    # Ensure x1, x2 are also numpy arrays
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)

    # Normalize weights to sum to 1
    normalized_weights = weights_arr / np.sum(weights_arr)
    
    # Calculate weighted MSE with normalized weights
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))

# Pearson Delta
def pearson(x1, x2):
    return np.corrcoef(x1, x2)[0, 1]

# R2 Score on deltas (helper)
def r2_score_on_deltas(delta_true, delta_pred, weights=None):
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan # r2_score needs at least 2 samples and matching shapes
    if weights is not None:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)

## Get the means for all the perturbations, total mean, and control mean
pert_means = get_pert_means(adata)
total_mean_original = np.mean(list(pert_means.values()), axis=0)
ctrl_mean_original = adata[adata.obs['condition'] == 'control'].X.mean(axis=0).A1

#### 0.1: Get a density plot of the delta per gene of all perts vs control
delta_all_vs_control = total_mean_original - ctrl_mean_original

# Create directory for exploratory plots if it doesn't exist
os.makedirs(f'{ANALYSIS_DIR}/00_exploratory_plots', exist_ok=True)

# Calculate mean and std for annotation
data_to_plot = delta_all_vs_control
mean_val = np.mean(data_to_plot)
std_val = np.std(data_to_plot)

# Plot density of delta_all_vs_control
plt.figure(figsize=(12, 7)) # Increased figure size for better annotation visibility
sns.kdeplot(data_to_plot, fill=True)
plt.title('Density of Delta (All Perturbations Mean vs Control Mean) per Gene', fontsize=16)
plt.xlabel('Delta (All Perts Mean - Control Mean)', fontsize=14)
plt.ylabel('Density', fontsize=14)
# Add vline for the mean
plt.axvline(mean_val, color='r', linestyle='--', linewidth=2)
# Annotate mean and std
ymin, ymax = plt.ylim()
plt.text(mean_val + (0.02 * (plt.xlim()[1] - plt.xlim()[0])), ymax * 0.9, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
         color='r', ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
plt.grid(True)
plot_save_path = f'{ANALYSIS_DIR}/00_exploratory_plots/delta_all_genes_vs_control_density.png'
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f'Density plot of delta_all_vs_control saved to {plot_save_path}')
plt.close()

#### 0.2: Get a density plot of the delta per gene of all perts vs control in raw counts space
total_mean_raw = adata.layers['counts'].mean(axis=0)
ctrl_mean_raw = adata[adata.obs['condition'] == 'control'].layers['counts'].mean(axis=0)
delta_all_vs_control_raw = total_mean_raw - ctrl_mean_raw

# Calculate mean and std for annotation
data_to_plot = delta_all_vs_control_raw.A1 if isinstance(delta_all_vs_control_raw, np.matrix) else delta_all_vs_control_raw
mean_val = np.mean(data_to_plot)
std_val = np.std(data_to_plot)

# Plot density of delta_all_vs_control_raw
plt.figure(figsize=(12, 7)) # Increased figure size for better annotation visibility
sns.kdeplot(data_to_plot, fill=True)
plt.title('Density of Delta (All Perturbations Mean vs Control Mean) per Gene', fontsize=16)
plt.xlabel('Delta (All Perts Mean - Control Mean) (raw counts)', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Add vline for the mean
plt.axvline(mean_val, color='r', linestyle='--', linewidth=2)

PLOT_DIR = f'{ANALYSIS_DIR}/00_exploratory_plots'
# Annotate mean and std
ymin, ymax = plt.ylim()
plt.text(mean_val + (0.02 * (plt.xlim()[1] - plt.xlim()[0])), ymax * 0.9, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
         color='r', ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.grid(True)
plot_save_path = f'{PLOT_DIR}/delta_all_genes_vs_control_density_raw.png'
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f'Density plot of delta_all_vs_control_raw saved to {plot_save_path}')
plt.close()

# Get the perts with at least N cells
max_cells_per_pert = max(DATASET_CELL_COUNTS)
pert_counts = adata.obs['condition'].value_counts()
pert_counts = pert_counts[(pert_counts >= max_cells_per_pert) & (pert_counts.index != 'control')]

