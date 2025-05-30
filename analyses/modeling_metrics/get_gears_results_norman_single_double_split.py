# %% [markdown]
# To run this notebook, create a conda environment with scgpt and other deps installed.
# 
# ```bash
# conda create -y -n scgpt python=3.11
# conda activate scgpt
# # pip install -r requirements.txt
# pip install scgpt
# ```
# 
# 

# %%
# Disable all warnings
import warnings
warnings.filterwarnings('ignore')

DATASET_NAME = 'norman19'

# %%
import numpy as np
import pandas as pd

# Read the numpy files
try:
    names_df_vsrest = np.load(f'../../data/{DATASET_NAME}/{DATASET_NAME}_names_df_vsrest.pkl', allow_pickle=True)
    print("Successfully loaded names_df_vsrest")
except Exception as e:
    print(f"Error loading names_df_vsrest: {e}")

try:
    scores_df_vsrest = np.load(f'../../data/{DATASET_NAME}/{DATASET_NAME}_scores_df_vsrest.pkl', allow_pickle=True)
    print("Successfully loaded scores_df_vsrest")
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


# Define function to parse perturbation strings
def parse_perturbation(pert):
    """Parse perturbation string into list of genes"""
    if pert == 'ctrl':
        return []
    elif '+ctrl' in pert:
        # Single perturbation with +ctrl
        return [pert.replace('+ctrl', '')]
    elif '+' in pert:
        # Double perturbation
        return pert.split('+')
    else:
        # Single perturbation without +ctrl (shouldn't happen in our case)
        return [pert]

# %%

adata.obs['condition'].unique()

# Get the single perts (lack a "+")
single_perts = [pert for pert in adata.obs['condition'].unique() if '+' not in pert]

# Get the double perts (have a "+")
double_perts = [pert for pert in adata.obs['condition'].unique() if '+' in pert]


# %%
# ===================GEARS DATA PREPARATION ===================

# Import required libraries
from gears import PertData, GEARS
import pickle


print("Starting GEARS data preparation...")

# Create a copy of the original data and subsample cells
print("Creating data copy...")
adata_gears = adata.copy()
np.random.seed(42)  # Set random seed for reproducibility

adata_gears = adata_gears.copy()

# Process condition labels
print("Processing condition labels...")
# Only add +ctrl to single perturbations (those without a '+')
def process_condition(cond):
    if cond == 'control':
        return 'ctrl'
    elif '+' not in cond:
        return cond + '+ctrl'
    else:
        return cond
    
adata_gears.obs['condition'] = adata_gears.obs['condition'].astype(str).apply(process_condition)

# Get unique perturbations
print("Getting unique perturbations...")
all_perturbations = adata_gears.obs['condition'].unique()
all_perturbations = all_perturbations[all_perturbations != 'ctrl']
print(f"Found {len(all_perturbations)} unique perturbations")

# Split perturbations based on single vs double
print("Splitting perturbations based on single vs double...")

# Separate single and double perturbations (accounting for the +ctrl suffix)
single_perturbations = [p for p in all_perturbations if '+' not in p.replace('+ctrl', '')]
double_perturbations = [p for p in all_perturbations if '+' in p.replace('+ctrl', '')]

print(f"Found {len(single_perturbations)} single perturbations")
print(f"Found {len(double_perturbations)} double perturbations")

# Split double perturbations 50/50
np.random.shuffle(double_perturbations)
double_split_idx = len(double_perturbations) // 2
double_train_perturbations = double_perturbations[:double_split_idx]
double_test_perturbations = double_perturbations[double_split_idx:]

print(f"Double perturbations for training: {len(double_train_perturbations)}")
print(f"Double perturbations for testing: {len(double_test_perturbations)}")

# Training set: all single + 50% double
train_perturbations = np.concatenate([single_perturbations, double_train_perturbations])
np.random.shuffle(train_perturbations)

# Test set: remaining 50% double
test_perturbations = double_test_perturbations

print(f"\nTotal training perturbations: {len(train_perturbations)}")
print(f"Total testing perturbations: {len(test_perturbations)}")

# Split training data into train/val (90/10)
print("\nSplitting training data into train/val...")
split_idx = int(len(train_perturbations) * 0.9)
train_split = train_perturbations[:split_idx]
val_split = train_perturbations[split_idx:]

split_dict = {
    'train': train_split,
    'val': val_split,
    'test': test_perturbations
}
print(f"Final splits - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_perturbations)}")

# Prepare data for GEARS
print("\nPreparing data for GEARS...")
# Add control to perturbation list
all_perturbations_with_ctrl = np.concatenate([['ctrl'], all_perturbations])

# Create dataset with all perturbations
print("Creating dataset...")
adata_gears_full = adata_gears[adata_gears.obs['condition'].isin(all_perturbations_with_ctrl)].copy()
print(f"Dataset size: {adata_gears_full.n_obs} cells")

# %%

adata_gears_full.var['gene_name'] = adata_gears_full.var.index.astype(str)

# Process data with GEARS
print("\nProcessing data with GEARS...")
pert_data = PertData('../../data')

# %%

pert_data.new_data_process(dataset_name='norman19_gears_single_double', adata=adata_gears_full)

# %%

# Filter out genes not in gene2go from each split
for split in ['train', 'val', 'test']:
    original_count = len(split_dict[split])
    filtered_perts = []
    for pert in split_dict[split]:
        genes = parse_perturbation(pert)
        if all(gene in pert_data.gene2go.keys() for gene in genes):
            filtered_perts.append(pert)
    split_dict[split] = filtered_perts
    filtered_count = len(split_dict[split])
    print(f"{split} split: {filtered_count}/{original_count} genes kept ({original_count - filtered_count} removed)")

with open('../../data/norman19_single_double_split_dict.pkl', 'wb') as f:
    pickle.dump(split_dict, f)


# %%

pert_data.load(data_path='../../data/norman19_gears_single_double')
pert_data.prepare_split(split='custom', seed=42, split_dict_path='../../data/norman19_single_double_split_dict.pkl')
print("Data processing complete")

print("\nGEARS data preparation completed successfully!")

# %%


# ===================GEARS MODEL TRAINING ===================

pert_data.get_dataloader(batch_size = 32, test_batch_size = 512)

gears_model = GEARS(pert_data, device = 'cuda', 
                    weight_bias_track = False, 
                    proj_name = 'norman19_single_double', 
                    exp_name = 'single_double_split')
gears_model.model_initialize()

gears_model.train(epochs = 10)

os.makedirs("../../data/gears_models", exist_ok=True)

gears_model.save_model('../../data/gears_models/norman19_single_double')


# %%

# ===================GEARS PREDICTIONS ===================

pert_data.get_dataloader(batch_size = 32, test_batch_size = 512)

gears_model = GEARS(pert_data, device = 'cuda', 
                    weight_bias_track = False, 
                    proj_name = 'norman19_single_double', 
                    exp_name = 'single_double_split')
gears_model.model_initialize()
gears_model.load_pretrained('../../data/gears_models/norman19_single_double')

# %%

# Get test perturbations for prediction
# Filter test perturbations to only include those with genes in gene2go
test_perturbations_filtered = []
test_perturbations_parsed = []
for pert in test_perturbations:
    genes = parse_perturbation(pert)
    if all(gene in pert_data.gene2go.keys() for gene in genes):
        test_perturbations_filtered.append(pert)
        test_perturbations_parsed.append(genes)

# Get predictions on test set
print(f"Predicting on {len(test_perturbations_parsed)} test perturbations...")
predictions = gears_model.predict(test_perturbations_parsed)

# Also get predictions for training set to have complete predictions
train_val_perturbations = np.concatenate([train_split, val_split])
train_val_perturbations_filtered = []
train_val_perturbations_parsed = []
for pert in train_val_perturbations:
    genes = parse_perturbation(pert)
    if all(gene in pert_data.gene2go.keys() for gene in genes):
        train_val_perturbations_filtered.append(pert)
        train_val_perturbations_parsed.append(genes)

print(f"Predicting on {len(train_val_perturbations_parsed)} train/val perturbations...")
train_predictions = gears_model.predict(train_val_perturbations_parsed)

# %%

# Combine all predictions
gears_predictions = {}
for pert in tqdm(predictions.keys()):
    gears_predictions[pert] = predictions[pert]
# for pert in tqdm(train_predictions.keys()):
#     gears_predictions[pert] = train_predictions[pert]
# gears_predictions['control'] = pert_means['control']

# Save GEARS predictions to pickle file
with open('../../data/gears_predictions_single_double_split.norman19.pkl', 'wb') as f:
    pickle.dump(gears_predictions, f)

print("\nGEARS single/double split analysis complete!")
print(f"Total predictions: {len(gears_predictions)}") 