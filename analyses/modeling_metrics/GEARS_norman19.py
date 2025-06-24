# %%
# Disable all warnings
import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', action='store_true', default=False)
args = parser.parse_args()

MULTIPROCESSING = args.multiprocessing

import os
import multiprocessing as mp

if MULTIPROCESSING:
    # Set multiprocessing start method to spawn to avoid CUDA initialization errors
    mp.set_start_method('spawn', force=True)

if not os.path.exists('../../data/norman19/norman19_processed.h5ad'):
    raise FileNotFoundError('../../data/norman19/norman19_processed.h5ad')

if not os.path.exists('../../data/norman19/norman19_names_df_vsrest.pkl'):
    raise FileNotFoundError('../../data/norman19/norman19_names_df_vsrest.pkl')

if not os.path.exists('../../data/norman19/norman19_scores_df_vsrest.pkl'):
    raise FileNotFoundError('../../data/norman19/norman19_scores_df_vsrest.pkl')

# %%
import numpy as np

names_df_vsrest = np.load('../../data/norman19/norman19_names_df_vsrest.pkl', allow_pickle=True)
print("Successfully loaded names_df_vsrest")
scores_df_vsrest = np.load('../../data/norman19/norman19_scores_df_vsrest.pkl', allow_pickle=True)
print("Successfully loaded scores_df_vsrest")


# %%

import numpy as np
import os
import pickle

import sys
sys.path.append(os.path.dirname(os.getcwd())) # For finding the 'analyses' package
from common import *


DATASET_NAME = 'norman19'

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

loss_weights_dict = {}
for key in pert_normalized_abs_scores_vsrest.keys():
    new_key = key+'+ctrl' if '+' not in key else key
    loss_weights_dict[new_key] = 100*pert_normalized_abs_scores_vsrest.get(key).values

# Save loss_weights_dict for multiprocessing
with open('../../data/norman19/norman19_loss_weights_dict.pkl', 'wb') as f:
    pickle.dump(loss_weights_dict, f)

adata.var['gene_name'] = adata.var.index.tolist()

# %%
# Import required libraries
from gears import PertData
from gears_icml import GEARS
import pickle

# Check if all prediction files exist
prediction_files = [
    '../../data/gears_predictions_mse_unweighted_norman19.pkl',
    '../../data/gears_predictions_mse_weighted_norman19.pkl',
    '../../data/gears_predictions_default_loss_unweighted_norman19.pkl',
]

# %%
# ===================GEARS DATA PREPARATION ===================

print("Starting GEARS data preparation...")

gears_predictions = {}

# Create a copy of the original data and subsample cells
print("Creating data copy...")
adata_gears = adata.copy()
np.random.seed(42)  # Set random seed for reproducibility

# Process condition labels
print("Processing condition labels...")
adata_gears.obs['condition'] = adata_gears.obs['condition'].astype(str)
adata_gears.obs['condition'] = adata_gears.obs['condition'].apply(lambda x: x + '+ctrl' if '+' not in x else x)
adata_gears.obs['condition'] = adata_gears.obs['condition'].str.replace('control+ctrl', 'ctrl')

# Get unique perturbations
print("Getting unique perturbations...")
all_perturbations = adata_gears.obs['condition'].unique()
all_perturbations = all_perturbations[all_perturbations != 'ctrl']
print(f"Found {len(all_perturbations)} unique perturbations")

# Split perturbations into first and second half
print("Splitting perturbations into halves...")
# Split perturbations into single gene and two gene perturbations
single_gene_perturbations = np.array([p for p in all_perturbations if '+ctrl' in p])
two_gene_perturbations = np.array([p for p in all_perturbations if 'ctrl' not in p])

print(f"Found {len(single_gene_perturbations)} single gene perturbations")
print(f"Found {len(two_gene_perturbations)} two gene perturbations")

# Use single gene perturbations for the split
first_half_perturbations = two_gene_perturbations[:len(two_gene_perturbations)//2]
second_half_perturbations = two_gene_perturbations[len(two_gene_perturbations)//2:]
print(f"First half: {len(first_half_perturbations)} perturbations")
print(f"Second half: {len(second_half_perturbations)} perturbations")

# Process first half splits
print("\nProcessing first half splits...")
first_half_train_val = first_half_perturbations
np.random.shuffle(first_half_train_val)
split_idx = int(len(first_half_train_val) * 0.9) # 90% train, 10% val
first_half_train = first_half_train_val[:split_idx]
first_half_val = first_half_train_val[split_idx:]

first_half_split_dict = {
    'train': np.concatenate([first_half_train, single_gene_perturbations]),
    'val': first_half_val,
    'test': second_half_perturbations
}
print(f"First half splits - Train: {len(first_half_train)}, Val: {len(first_half_val)}")

# Process second half splits
print("\nProcessing second half splits...")
second_half_train_val = second_half_perturbations
np.random.shuffle(second_half_train_val) 
split_idx = int(len(second_half_train_val) * 0.9)  # 90% train, 10% val
second_half_train = second_half_train_val[:split_idx]
second_half_val = second_half_train_val[split_idx:]

second_half_split_dict = {
    'train': np.concatenate([second_half_train, single_gene_perturbations]),
    'val': second_half_val,
    'test': first_half_perturbations
}
print(f"Second half splits - Train: {len(second_half_train)}, Val: {len(second_half_val)}")

# Assert no test perts are in the train or val
assert not any(pert in first_half_split_dict['train'] for pert in first_half_split_dict['test'])
assert not any(pert in first_half_split_dict['val'] for pert in first_half_split_dict['test'])
assert not any(pert in second_half_split_dict['train'] for pert in second_half_split_dict['test'])
assert not any(pert in second_half_split_dict['val'] for pert in second_half_split_dict['test'])
# Assert that first half and second half are disjoint
assert set(first_half_split_dict['val']).isdisjoint(set(second_half_split_dict['val']))
assert set(first_half_split_dict['test']).isdisjoint(set(second_half_split_dict['test']))

# %%

# Prepare data for GEARS
print("\nPreparing data for GEARS...")
all_perts = [val.tolist() for k,val in first_half_split_dict.items()]
all_perts = list(set([item for sublist in all_perts for item in sublist]))
# Concatenate with 'ctrl'
all_perturbations = np.concatenate([['ctrl'], all_perts])

# Create subsetted datasets
print("Creating subsetted datasets...")
adata_gears = adata_gears[adata_gears.obs['condition'].isin(all_perturbations)].copy()
print(f"Dataset size: {adata_gears.n_obs} cells")

# %%

# Process first half data
print("\nProcessing first half data with GEARS...")
pert_data = PertData('../../data/norman19')
if not os.path.exists('../../data/norman19/norman19/data_pyg/cell_graphs.pkl'):
    pert_data.new_data_process(dataset_name='norman19', adata=adata_gears)

pert_data_first_half = PertData('../../data/norman19')
pert_data_first_half.load(data_path='../../data/norman19/norman19/')
pert_data_second_half = PertData('../../data/norman19')
pert_data_second_half.load(data_path='../../data/norman19/norman19/')

# %%

# Filter out genes not in gene2go from each split
for split in ['train', 'val', 'test']:
    original_count = len(first_half_split_dict[split])
    first_half_split_dict[split] = [gene for gene in first_half_split_dict[split] 
                                if gene.split('+')[0] in ['ctrl'] + list(pert_data_first_half.gene2go.keys()) and gene.split('+')[1] in ['ctrl'] + list(pert_data_first_half.gene2go.keys())]
    filtered_count = len(first_half_split_dict[split])
    print(f"{split} split: {filtered_count}/{original_count} genes kept ({original_count - filtered_count} removed)")

with open('../../data/norman19/norman19_gears_first_half_split_dict.pkl', 'wb') as f:
    pickle.dump(first_half_split_dict, f)
pert_data_first_half.prepare_split(split='custom', seed=42, split_dict_path='../../data/norman19/norman19_gears_first_half_split_dict.pkl')
print("First half data processing complete")


# Filter out genes not in gene2go from each split
for split in ['train', 'val', 'test']:
    original_count = len(second_half_split_dict[split])
    second_half_split_dict[split] = [gene for gene in second_half_split_dict[split] 
                                if gene.split('+')[0] in ['ctrl'] + list(pert_data_second_half.gene2go.keys()) and gene.split('+')[1] in ['ctrl'] + list(pert_data_second_half.gene2go.keys())]
    filtered_count = len(second_half_split_dict[split])
    print(f"{split} split: {filtered_count}/{original_count} genes kept ({original_count - filtered_count} removed)")
with open('../../data/norman19/norman19_gears_second_half_split_dict.pkl', 'wb') as f:
    pickle.dump(second_half_split_dict, f)
pert_data_second_half.prepare_split(split='custom', seed=42, split_dict_path='../../data/norman19/norman19_gears_second_half_split_dict.pkl')
print("Second half data processing complete")

# %%

# Define function to train a single model
def train_single_model(config):
    # Import necessary libraries within the process
    from gears import PertData
    from gears_icml import GEARS
    import pickle
    import os
    
    loss, weight, model_half, gpu_id = config
    
    # Use the specific GPU directly
    if MULTIPROCESSING:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cuda'
    
    # Check if predictions already exist
    if os.path.exists(f'../../data/norman19/gears_predictions_{loss}_{weight}_norman19.pkl'):
        print(f"Predictions already exist for {loss} loss and {weight} weight, skipping...")
        return
    
    print(f"Training GEARS {model_half} model with {loss} loss and {weight} weight on GPU {gpu_id}")
    
    # Load the data within each process to avoid CUDA memory sharing issues
    # Load the split dictionaries
    with open('../../data/norman19/norman19_gears_first_half_split_dict.pkl', 'rb') as f:
        first_half_split_dict = pickle.load(f)
    with open('../../data/norman19/norman19_gears_second_half_split_dict.pkl', 'rb') as f:
        second_half_split_dict = pickle.load(f)
    
    # Load PertData
    if model_half == 'first_half':
        pert_data_first_half = PertData('../../data/norman19')
        pert_data_first_half.load(data_path='../../data/norman19/norman19/')
        pert_data_first_half.prepare_split(split='custom', seed=42, split_dict_path='../../data/norman19/norman19_gears_first_half_split_dict.pkl')
        pert_data_first_half.get_dataloader(batch_size = 32, test_batch_size = 512)

    elif model_half == 'second_half':
        pert_data_second_half = PertData('../../data/norman19')
        pert_data_second_half.load(data_path='../../data/norman19/norman19/')
        pert_data_second_half.prepare_split(split='custom', seed=42, split_dict_path='../../data/norman19/norman19_gears_second_half_split_dict.pkl')
        pert_data_second_half.get_dataloader(batch_size = 32, test_batch_size = 512)
    

    # Load loss weights if needed
    if weight == 'weighted':
        # Load the saved loss weights dictionary
        with open('../../data/norman19/norman19_loss_weights_dict.pkl', 'rb') as f:
            loss_weights_dict = pickle.load(f)
    else:
        loss_weights_dict = None
    
    # ===================GEARS MODEL TRAINING ===================
    
    if model_half == 'first_half':
        # Train first half model
        gears_model = GEARS(pert_data_first_half, device = device, 
                           weight_bias_track = False, 
                           proj_name = 'first_half_norman19', 
                           exp_name = f'first_half_{loss}_{weight}',
                           loss_weights_dict = loss_weights_dict,
                           use_mse_loss = True if loss == 'mse' else False)
        gears_model.model_initialize()
        gears_model.train(epochs = 10)
        
        os.makedirs("../../data/norman19/gears_models", exist_ok=True)
        gears_model.save_model(f'../../data/norman19/gears_models/norman19_first_half_{loss}_{weight}')
        
        # Save partial results
        assert not any(pert in first_half_split_dict['test'] for pert in first_half_split_dict['train'])
        to_predict = [gene.split('+') for gene in first_half_split_dict['test'] if gene.split('+')[0] in pert_data_first_half.gene2go.keys() and gene.split('+')[1] in pert_data_first_half.gene2go.keys()]
        predictions = gears_model.predict(to_predict)
        with open(f'../../data/norman19/gears_predictions_{loss}_{weight}_first_half_temp.pkl', 'wb') as f:
            pickle.dump(predictions, f)
            
    else:  # second_half
        # Train second half model
        gears_model = GEARS(pert_data_second_half, device = device, 
                           weight_bias_track = False, 
                           proj_name = 'second_half_norman19', 
                           exp_name = f'second_half_{loss}_{weight}',
                           loss_weights_dict = loss_weights_dict,
                           use_mse_loss = True if loss == 'mse' else False)
        gears_model.model_initialize()
        gears_model.train(epochs = 10)

        os.makedirs("../../data/norman19/gears_models", exist_ok=True)
        gears_model.save_model(f'../../data/norman19/gears_models/norman19_second_half_{loss}_{weight}')
        
        # Save partial results
        assert not any(pert in second_half_split_dict['test'] for pert in second_half_split_dict['train'])
        to_predict = [gene.split('+') for gene in second_half_split_dict['test'] if gene.split('+')[0] in pert_data_second_half.gene2go.keys() and gene.split('+')[1] in pert_data_second_half.gene2go.keys()]
        predictions = gears_model.predict(to_predict)
        with open(f'../../data/norman19/gears_predictions_{loss}_{weight}_second_half_temp.pkl', 'wb') as f:
            pickle.dump(predictions, f)
    
    print(f"Completed training for {model_half} with {loss} loss and {weight} weight on GPU {gpu_id}")

# Function to combine predictions after all models are trained
def combine_predictions(loss, weight):
    # Import necessary libraries
    import pickle
    import os
    import scanpy as sc
    import numpy as np
    
    # Check if both temp files exist
    first_half_file = f'../../data/norman19/gears_predictions_{loss}_{weight}_first_half_temp.pkl'
    second_half_file = f'../../data/norman19/gears_predictions_{loss}_{weight}_second_half_temp.pkl'
    
    if os.path.exists(first_half_file) and os.path.exists(second_half_file):
        # Load predictions
        with open(first_half_file, 'rb') as f:
            predictions_from_first_half = pickle.load(f)
        with open(second_half_file, 'rb') as f:
            predictions_from_second_half = pickle.load(f)
        
        # Load control mean from the original adata
        adata = sc.read_h5ad('../../data/norman19/norman19_processed.h5ad')
        ctrl_mean = np.array(adata[adata.obs['condition'] == 'control'].X.mean(axis=0)).flatten()
        
        # Combine predictions
        gears_predictions = {}
        for pert in predictions_from_second_half.keys():
            gears_predictions[pert.replace('_', '+')] = predictions_from_second_half[pert]
        for pert in predictions_from_first_half.keys():
            gears_predictions[pert.replace('_', '+')] = predictions_from_first_half[pert]
        gears_predictions['control'] = ctrl_mean
        
        # Save combined predictions
        with open(f'../../data/norman19/gears_predictions_{loss}_{weight}_norman19.pkl', 'wb') as f:
            pickle.dump(gears_predictions, f)
        
        # Clean up temp files
        os.remove(first_half_file)
        os.remove(second_half_file)
        
        print(f"Combined and saved predictions for {loss} loss and {weight} weight")


# %%

# Train and evaluate models with different loss functions and weights
losses = ['mse', 'default_loss']
weights = ['unweighted', 'weighted']

if __name__ == '__main__':
    # Create configurations for parallel training (8 jobs for 8 GPUs)
    configurations = []
    gpu_id = 0
    for loss in losses:
        for weight in weights:
            if loss == 'default_loss' and weight == 'weighted':
                continue
            # Add both first_half and second_half as separate jobs
            configurations.append((loss, weight, 'first_half', gpu_id))
            gpu_id += 1
            configurations.append((loss, weight, 'second_half', gpu_id))
            gpu_id += 1

    # Run training in parallel across all 8 GPUs
    if MULTIPROCESSING:
        with mp.Pool(processes=8) as pool:
            pool.map(train_single_model, configurations)
    else:
        for config in configurations:
            train_single_model(config)

    # Combine predictions for each loss/weight combination
    for loss in losses:
        for weight in weights:
            combine_predictions(loss, weight)

    # Load all predictions into gears_predictions dictionary
    gears_predictions = {}
    for loss in losses:
        for weight in weights:
            if os.path.exists(f'../../data/norman19/gears_predictions_{loss}_{weight}_norman19.pkl'):
                with open(f'../../data/norman19/gears_predictions_{loss}_{weight}_norman19.pkl', 'rb') as f:
                    gears_predictions[f'{loss}_{weight}'] = pickle.load(f)
    
    print("All GEARS models trained and predictions saved!")

# %%



