# Adapted from https://github.com/altoslabs/perturbench/blob/4dd6a03c54ef348f6be6e46e21ea7f80eb2daaa2/notebooks/neurips2024/data_curation/curate_Norman19.ipynb
# This is performed using the base installation of 
# FIXME: Remove unnecessary imports
import scanpy as sc
import os
import subprocess as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np


np.random.seed(42)

data_url = 'https://plus.figshare.com/ndownloader/files/35775606'
data_cache_dir = './data/replogle22'

def compute_degs(adata, mode='vsrest', pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.
    
    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)
    
    Returns:
        dict: rank_genes_groups results dictionary
        
    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == 'vsrest':
        # Remove control cells for vsrest analysis
        adata_subset = adata[adata.obs['condition'] != 'control'].copy()
        reference = 'rest'
    elif mode == 'vscontrol':
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = 'control'
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # Compute DEGs
    sc.tl.rank_genes_groups(adata_subset, 'condition', method='t-test_overestim_var', reference=reference)
    
    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])
    
    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs['condition'].unique(), desc=f"Computing DEGs {mode}"):
        if mode == 'vscontrol' and pert == 'control':
            continue  # Skip control when comparing vs control
            
        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]
        
        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]
        
        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()
        
        deg_dict[pert] = {'up': pert_degs_sig_up, 'down': pert_degs_sig_down}
    
    # Save results to adata.uns
    adata.uns[f'deg_dict_{mode}'] = deg_dict
    adata.uns[f'rank_genes_groups_{mode}'] = adata_subset.uns['rank_genes_groups'].copy()
    
    return adata_subset.uns['rank_genes_groups']

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/replogle22_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget -q {data_url} -O {tmp_data_dir}', shell=True)

adata = sc.read_h5ad(tmp_data_dir)

# Rename columns
adata.obs.rename(columns = {
    'UMI_count': 'ncounts',
    'mitopercent': 'percent_mito',
}, inplace=True)
adata.obs['cell_type'] = "RPE1"
adata.obs['condition'] = adata.obs['gene'].str.replace('non-targeting', 'control')
adata.obs['condition'] = adata.obs['condition'].astype('category')
adata.X = csr_matrix(adata.X)

# Filter cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Stash raw counts
adata.layers['counts'] = adata.X.copy()

# For every perturbation, for every gene, calculate the mean and variance of the counts
mean_df = pd.DataFrame(index=adata.var_names, columns=adata.obs['condition'].unique())
disp_df = pd.DataFrame(index=adata.var_names, columns=adata.obs['condition'].unique())
for pert in tqdm(adata.obs['condition'].unique()):
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    pert_counts = adata[pert_cells].X.toarray()
    mean_df.loc[:, pert] = np.mean(pert_counts, axis=0)
    disp_df.loc[:, pert] = np.var(pert_counts, axis=0)

# Save to the uns dictionary
mean_df_dict = mean_df.to_dict(orient='list')
disp_df_dict = disp_df.to_dict(orient='list')
adata.uns['mean_dict'] = mean_df_dict
adata.uns['disp_dict'] = disp_df_dict
adata.uns['mean_disp_dict_genes'] = disp_df.index.tolist()


# Do library size norm and log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Downsample each perturbation to have no more than N cells
MAX_CELLS = 64
MAX_CELLS_CONTROL = 8192

pert_counts = adata.obs['condition'].value_counts()
pert_counts = pert_counts[pert_counts > MAX_CELLS]
cells_to_keep = []
for pert in pert_counts.index:
    pert_cells = adata.obs[adata.obs['condition'] == pert].index.tolist()
    if pert == 'control':
        pert_cells = np.random.choice(pert_cells, size=MAX_CELLS_CONTROL, replace=False)
    else:
        pert_cells = np.random.choice(pert_cells, size=MAX_CELLS, replace=False)
    cells_to_keep.extend(pert_cells)

# Subset the adata object
adata = adata[cells_to_keep]

# Get 8192 HVGs -- subset the adata object to only include the HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=True)

# Do PCA
sc.pp.pca(adata)

# Do UMAP
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Do clustering
sc.tl.leiden(adata)

# Calculate DEGs between each perturbation and all other perturbations
_ = compute_degs(adata, mode='vsrest')

# Calculate DEGs with respect to the control perturbation
_ = compute_degs(adata, mode='vscontrol')

# Convert to format that can be saved
SCORE_TYPE = 'scores' # or 'logfoldchanges'
names_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"]["names"])
scores_df_vsrest = pd.DataFrame(adata.uns["rank_genes_groups_vsrest"][SCORE_TYPE])
names_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"]["names"])
scores_df_vsctrl = pd.DataFrame(adata.uns["rank_genes_groups_vscontrol"][SCORE_TYPE])

# Save dataframes to csv
names_df_vsrest.to_pickle(f'{data_cache_dir}/replogle22_names_df_vsrest.pkl')
scores_df_vsrest.to_pickle(f'{data_cache_dir}/replogle22_scores_df_vsrest.pkl')
names_df_vsctrl.to_pickle(f'{data_cache_dir}/replogle22_names_df_vsctrl.pkl')
scores_df_vsctrl.to_pickle(f'{data_cache_dir}/replogle22_scores_df_vsctrl.pkl')

# Remove these from the adata object
adata.uns.pop('rank_genes_groups_vsrest', None)
adata.uns.pop('rank_genes_groups_vscontrol', None)
adata.uns.pop('rank_genes_groups', None)

# Save the data
output_data_path = f'{data_cache_dir}/replogle22_processed.h5ad'
adata.write_h5ad(output_data_path)
# Save adata.var to a CSV
genes_path = f'{data_cache_dir}/replogle22_genes.csv.gz'
adata.var[['gene_name', 'highly_variable', 'means', 'dispersions', 'dispersions_norm']].to_csv(genes_path)


### Last addition: Get DEGs for different library size quantiles
SCORE_TYPE = 'scores' # or 'logfoldchanges'

n_counts = adata.obs['ncounts'].values
all_perts = adata.obs['condition'].unique()

# Get the quantiles for the perturbed and the control cells
quantiles = np.arange(0, 1, 0.1)

import scanpy as sc


# For each perturbation get the quantiles of the library size and the associated cells
# Then create an Adata for each quantile
pert_normalized_abs_scores_vsrest_quantiles = {}
for i, quantile in tqdm(enumerate(quantiles)):
    quantile_next = quantiles[i+1] if i < len(quantiles) - 1 else 1
    quantile_string = f"{quantile:.1f}-{quantile_next:.1f}"
    adatas_quantiles = []
    for pert in all_perts:
        pert_cells_idx = adata.obs['condition'] == pert
        pert_n_counts = n_counts[pert_cells_idx]
        cell_ids = adata.obs_names[pert_cells_idx]
        # Rank the pert_n_counts
        pert_n_counts_ranked = pd.Series(pert_n_counts).rank(pct=True).values
        # Get the cells that are in the quantile
        quantile_cells_idx = (pert_n_counts_ranked >= quantile) & (pert_n_counts_ranked < quantile_next)
        # Get the mean of the pert cells
        adatas_quantiles.append(adata[cell_ids[quantile_cells_idx]])
    adata_quantile = sc.concat(adatas_quantiles)
    # Get DEGs vs rest
    curr_deg_results = compute_degs(adata_quantile, mode='vsrest')
    names_df_vsrest = pd.DataFrame(curr_deg_results["names"])
    scores_df_vsrest = pd.DataFrame(curr_deg_results[SCORE_TYPE])

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
        stronger_normalized_weights = np.square(normalized_weights)
        
        weights = pd.Series(stronger_normalized_weights, index=names_df_vsrest[pert].values, name=pert)
        # Order by the var_names
        weights = weights.reindex(adata.var_names)
        pert_normalized_abs_scores_vsrest[pert] = weights

    pert_normalized_abs_scores_vsrest_quantiles[quantile_string] = pert_normalized_abs_scores_vsrest
    # FIXME: Ensure weights add up to 1 for each perturbation in the final computation of the WMSE
    

# Write to pickle
import pickle
with open(f'{data_cache_dir}/replogle22_pert_normalized_abs_scores_vsrest_quantiles.pkl', 'wb') as f:
    pickle.dump(pert_normalized_abs_scores_vsrest_quantiles, f)

### Do the same for the number of cells per perturbation
DATASET_CELL_COUNTS = [2, 4, 8, 16, 32, 64]

# For each perturbation, downsample to the number of cells in DATASET_CELL_COUNTS
# Then calculate the DEGs vs rest
# Save the selected cells inside the dictionary
pert_normalized_abs_scores_vsrest_cells_per_pert = {}
pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells = {}
for i, n_cells in enumerate(DATASET_CELL_COUNTS):
    pert_normalized_abs_scores_vsrest_cells_per_pert[n_cells] = {}
    adatas_cells_per_pert = []
    for pert in all_perts:
        pert_cells_idx = adata.obs_names[adata.obs['condition'] == pert]
        # Randomly select n_cells cells
        pert_cells = np.random.choice(pert_cells_idx, size=n_cells, replace=False)
        adatas_cells_per_pert.append(adata[pert_cells])
    adata_n_cells = sc.concat(adatas_cells_per_pert)
    # Get DEGs vs rest
    curr_deg_results = compute_degs(adata_n_cells, mode='vsrest')
    names_df_vsrest = pd.DataFrame(curr_deg_results["names"])
    scores_df_vsrest = pd.DataFrame(curr_deg_results[SCORE_TYPE])
    
    # Get the weights for each perturbation
    pert_normalized_abs_scores_vsrest_cells_per_pert[n_cells] = {}
    pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells[n_cells] = adata_n_cells.obs_names
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
                normalized_weights = np.ones_like(abs_scores) 
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
        
        # Ensure no NaNs in weights, replace with 0 if any (e.g. if a gene had NaN score originally)
        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
        
        # Make weighting stronger by squaring the normalized weights
        stronger_normalized_weights = np.square(normalized_weights)
        
        weights = pd.Series(stronger_normalized_weights, index=names_df_vsrest[pert].values, name=pert)
        # Order by the var_names
        weights = weights.reindex(adata.var_names)
        pert_normalized_abs_scores_vsrest_cells_per_pert[n_cells][pert] = weights
        # FIXME: Ensure weights add up to 1 for each perturbation in the final computation of the WMSE

# Write to pickle
with open(f'{data_cache_dir}/replogle22_pert_normalized_abs_scores_vsrest_cells_per_pert.pkl', 'wb') as f:
    pickle.dump(pert_normalized_abs_scores_vsrest_cells_per_pert, f)
with open(f'{data_cache_dir}/replogle22_pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells.pkl', 'wb') as f:
    pickle.dump(pert_normalized_abs_scores_vsrest_cells_per_pert_selectedcells, f)