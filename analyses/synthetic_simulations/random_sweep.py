import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import multiprocessing
from tqdm import tqdm
from scipy import stats

# No scanpy or anndata imports needed if we are truly removing them

# Set OpenBLAS threads early if it was found to be helpful, otherwise optional
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Set seed
np.random.seed(42)

def nb_cells(mean, l_c, theta, rng): # theta kept as generic parameter name for this utility function
    """
    Generate individual cell profiles from NB distribution
    Returns an array of shape (len(l_c), G)
    """
    # Ensure mean and theta are numpy arrays for element-wise operations
    mean_arr = np.asarray(mean)
    theta_arr = np.asarray(theta)
    l_c_arr = np.asarray(l_c)

    # Correct mean for library size
    lib_size_corrected_mean = np.outer(l_c_arr, mean_arr)

    # Prevent division by zero or negative p if theta + mean is zero or mean is much larger than theta
    # This can happen if means are very low and theta is also low.
    # Add a small epsilon to the denominator to stabilize.
    # Also ensure p is within (0, 1)
    p_denominator = theta_arr + lib_size_corrected_mean
    p_denominator[p_denominator <= 0] = 1e-9 # Avoid zero or negative denominator
    
    p = theta_arr / p_denominator
    p = np.clip(p, 1e-9, 1 - 1e-9) # Ensure p is in a valid range for negative_binomial

    # Negative binomial expects n (number of successes, our theta) to be > 0.
    # And p (probability of success) to be in [0, 1].
    # If theta contains zeros or negatives, np.random.negative_binomial will fail.
    # Assuming theta values are appropriate (positive).

    predicted_counts = rng.negative_binomial(theta_arr, p)
    return predicted_counts

def simulate_one_run_numpy( # Renamed to signify it's the numpy-only version
    G=10_000,
    N0=3_000,
    Nk=150,
    P=50,
    p_effect=0.01,
    effect_factor=2.0,
    B=0.0,
    mu_l=1.0,
    all_theta=None, # Theta parameter for all cells 
    control_mu=None, # Control mu parameters
    pert_mu=None, # Perturbed mu parameters 
    trial_id_for_rng=None # Optional for seeding RNG per trial
):
    """
    Simulate an experiment using only NumPy/Pandas for calculations.
    """
    # Setup random number generator for this trial
    if trial_id_for_rng is not None:
        rng = np.random.RandomState(trial_id_for_rng)
    else:
        rng = np.random.RandomState(42) # Fallback, but ideally seeded per trial
    
    # --- Parameter Preparation with assertions ---
    # Assert that control_mu, pert_mu, and all_theta are provided
    assert control_mu is not None, "control_mu must be provided. None value is not allowed."
    assert pert_mu is not None, "pert_mu must be provided. None value is not allowed."
    assert all_theta is not None, "all_theta must be provided. None value is not allowed."
    # Assert that inputs are already arrays
    assert isinstance(control_mu, np.ndarray), "control_mu must be a numpy array"
    assert isinstance(pert_mu, np.ndarray), "pert_mu must be a numpy array"
    assert isinstance(all_theta, np.ndarray), "all_theta must be a numpy array"
    # Assert that they have the same length
    assert len(control_mu) == len(all_theta), "control_mu and all_theta must have the same length."
    assert len(control_mu) == len(pert_mu), "control_mu and pert_mu must have the same length."
    # Assert that G is not larger than the provided arrays
    assert len(control_mu) >= G, f"G parameter ({G}) cannot be larger than the length of provided arrays ({len(control_mu)})"
    # --- End of assertions ---
    
    # Sample G elements from control_mu and all_theta
    indices = rng.choice(len(control_mu), size=G, replace=False)
    local_control_mu = control_mu[indices]
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_pert_mu = pert_mu[indices]

    # --- Data Generation (counts) ---
    # Pre-allocate x_mat for raw counts
    x_mat_dtype = np.int32 
    x_mat = np.empty((N0 + P * Nk, G), dtype=x_mat_dtype)

    # 1. Sample control cells with bias (B, dispersion set to all_theta from all cells, fixed dispersion assumption)
    lib_size_control = rng.lognormal(mean=mu_l, sigma=0.1714, size=N0) # 0.1714 from all cells of the Norman19 dataset
    control_cells_counts = nb_cells(mean=local_control_mu, l_c=lib_size_control, theta=local_all_theta, rng=rng)
    x_mat[:N0, :] = control_cells_counts
    del control_cells_counts
    
    all_affected_masks = [] 
    current_row = N0
    
    # Define global perturbation bias
    delta_b = local_pert_mu - local_control_mu
    local_pert_mu_biased = np.clip(local_control_mu + B * delta_b, 0.0, np.inf)

    # 2. For each perturbation generate the cells
    for _ in range(P):
        affected_mask_loop = rng.random(G) < p_effect
        all_affected_masks.append(affected_mask_loop)
        
        mu_k_loop = local_pert_mu_biased.copy()
        if affected_mask_loop.sum() > 0:
            effect_directions = rng.choice([effect_factor, 1.0/effect_factor], size=affected_mask_loop.sum())
            mu_k_loop[affected_mask_loop] *= effect_directions

        lib_size_pert = rng.lognormal(mean=mu_l, sigma=0.1714, size=Nk) # 0.1714 from all cells of the Norman19 dataset
        pert_cells_counts = nb_cells(mean=mu_k_loop, l_c=lib_size_pert, theta=local_all_theta, rng=rng)
        x_mat[current_row : current_row + Nk, :] = pert_cells_counts
        current_row += Nk

        del mu_k_loop
        del lib_size_pert
        del pert_cells_counts
        del affected_mask_loop
        
    # --- Manual Normalization (Counts per 10k) ---
    library_sizes = x_mat.sum(axis=1)
    # Avoid division by zero for cells with no counts
    library_sizes[library_sizes == 0] = 1 
    
    # Ensure library_sizes is a column vector for broadcasting
    norm_factor = 1e4 / library_sizes[:, np.newaxis]
    norm_mat = x_mat * norm_factor
    norm_mat = norm_mat.astype(np.float32) # Cast to float32 after normalization
    del library_sizes
    del norm_factor
    # x_mat (raw counts) can be deleted if not needed directly later, 
    # but it's good to keep if comparison to raw is ever needed.
    # For this specific metric calculation, we proceed with norm_mat.
    # Let's keep x_mat for now, and delete it at the very end if it's large.

    # --- Manual Log1p Transformation ---
    log_norm_mat = np.log1p(norm_mat)
    del norm_mat # norm_mat is now transformed into log_norm_mat

    # --- Metric Calculation on log_norm_mat ---
    # Define perturbation identities for indexing
    pert_identities = ['control'] * N0 + [f'perturbation_{p_idx}' for p_idx in range(P) for _ in range(Nk)]
    
    control_cells_idx_np = np.array([True if name == 'control' else False for name in pert_identities])
    
    # Compute both observed mean and variance of control cells
    hat_mu0 = log_norm_mat[control_cells_idx_np, :].mean(axis=0)
    hat_var0 = log_norm_mat[control_cells_idx_np, :].var(axis=0)

    # Compute mean and var for the pooled cells (all non-control cells)
    assert np.sum(~control_cells_idx_np) > 0, "No perturbed cells found"
    hat_mu_pool = log_norm_mat[~control_cells_idx_np, :].mean(axis=0)
    
    # Compute delta_pred for pooled cells
    delta_pred = hat_mu_pool - hat_mu0

    delta_obs_list = []
    degs_list = []

    for i, p_idx in enumerate(range(P)):

        pert_name = f'perturbation_{p_idx}'
        # Create boolean index for current perturbation
        pert_cells_idx_np = np.array([True if name == pert_name else False for name in pert_identities])
        n_cells_pert = np.sum(pert_cells_idx_np)
        assert n_cells_pert > 0, "No perturbed cells found"
        
        # Compute mean, var and delta for the current perturbation
        hat_muk = log_norm_mat[pert_cells_idx_np, :].mean(axis=0)
        hat_vark = log_norm_mat[pert_cells_idx_np, :].var(axis=0)
        delta_obs = hat_muk - hat_mu0

        # Store results
        delta_obs_list.append(delta_obs)

        _, pvals = stats.ttest_ind_from_stats(
                    mean1=hat_muk,
                    std1=np.sqrt(hat_vark),
                    nobs1=n_cells_pert,
                    mean2=hat_mu0,
                    std2=np.sqrt(hat_var0),
                    nobs2=n_cells_pert, # NOTE: Left like this for the over estimation of variance
                    equal_var=False,  # Welch's
                )

        # Add to DEG list a binary mask of the top all_affected_masks[i].sum() DEGs
        n_degs = all_affected_masks[i].sum()
        # Get indices of genes with lowest p-values
        top_deg_indices = np.argsort(pvals)[:n_degs]
        # Create binary mask for DEGs
        deg_mask = np.zeros(G, dtype=bool)
        deg_mask[top_deg_indices] = True
        degs_list.append(deg_mask)
        

        del delta_obs
        del hat_muk
        del hat_vark


    # --- Compute evaluation metrics (similar to original but on NumPy arrays) ---
    results_accumulator = {
        'pearson_all': [], 'pearson_affected': [],
        'mae_all': [], 'mae_affected': [],
        'mse_all': [], 'mse_affected': [],
        'pearson_degs': [], 'mae_degs': [], 'mse_degs': []  # Added metrics for DEGs
    }
    
    any_affected_calculate = np.zeros(G, dtype=bool)
    if all_affected_masks:
        for mask_item in all_affected_masks:
            any_affected_calculate = np.logical_or(any_affected_calculate, mask_item)
    
    for i_loop, delta_obs_item_loop in enumerate(delta_obs_list):
        current_affected_mask = all_affected_masks[i_loop]
        current_degs_mask = degs_list[i_loop]  # Get the DEGs mask for current perturbation
        
        delta_obs_item_loop = delta_obs_item_loop.astype(np.float32)
        delta_pred_local = delta_pred.astype(np.float32)

        if np.std(delta_obs_item_loop) > 1e-6 and np.std(delta_pred_local) > 1e-6:
            corr_all = np.corrcoef(delta_obs_item_loop, delta_pred_local)[0, 1]
        else:
            corr_all = np.nan
        results_accumulator['pearson_all'].append(corr_all)
        
        if current_affected_mask.sum() > 1:
            # Slice arrays first, then check std
            delta_obs_affected = delta_obs_item_loop[current_affected_mask]
            delta_pred_affected = delta_pred_local[current_affected_mask]
            if np.std(delta_obs_affected) > 1e-6 and np.std(delta_pred_affected) > 1e-6:
                corr_affected = np.corrcoef(delta_obs_affected, delta_pred_affected)[0, 1]
            else:
                corr_affected = np.nan
            results_accumulator['pearson_affected'].append(corr_affected)
            del delta_obs_affected
            del delta_pred_affected
        else:
            results_accumulator['pearson_affected'].append(np.nan)
        
        results_accumulator['mae_all'].append(np.mean(np.abs(delta_obs_item_loop - delta_pred_local)))
        results_accumulator['mse_all'].append(np.mean(np.square(delta_obs_item_loop - delta_pred_local)))
        
        if current_affected_mask.sum() > 0:
            results_accumulator['mae_affected'].append(np.mean(np.abs(delta_obs_item_loop[current_affected_mask] - delta_pred_local[current_affected_mask])))
            results_accumulator['mse_affected'].append(np.mean(np.square(delta_obs_item_loop[current_affected_mask] - delta_pred_local[current_affected_mask])))
        else:
            results_accumulator['mae_affected'].append(np.nan)
            results_accumulator['mse_affected'].append(np.nan)
        
        # Calculate metrics for DEGs in the same way as for affected genes
        if current_degs_mask.sum() > 1:
            # Slice arrays first, then check std
            delta_obs_degs = delta_obs_item_loop[current_degs_mask]
            delta_pred_degs = delta_pred_local[current_degs_mask]
            if np.std(delta_obs_degs) > 1e-6 and np.std(delta_pred_degs) > 1e-6:
                corr_degs = np.corrcoef(delta_obs_degs, delta_pred_degs)[0, 1]
            else:
                corr_degs = np.nan
            results_accumulator['pearson_degs'].append(corr_degs)
            del delta_obs_degs
            del delta_pred_degs
        else:
            results_accumulator['pearson_degs'].append(np.nan)
        
        if current_degs_mask.sum() > 0:
            results_accumulator['mae_degs'].append(np.mean(np.abs(delta_obs_item_loop[current_degs_mask] - delta_pred_local[current_degs_mask])))
            results_accumulator['mse_degs'].append(np.mean(np.square(delta_obs_item_loop[current_degs_mask] - delta_pred_local[current_degs_mask])))
        else:
            results_accumulator['mae_degs'].append(np.nan)
            results_accumulator['mse_degs'].append(np.nan)
        # del current_affected_mask # Mask is from all_affected_masks, will be deleted later

    # Clean up large arrays
    del local_control_mu
    del local_all_theta  # Theta parameter from all cells estimation
    del x_mat # Raw counts matrix
    del log_norm_mat # Final processed matrix for metrics
    del all_affected_masks
    del hat_mu0
    del delta_pred
    del delta_obs_list # List of G-sized arrays
    # any_affected_calculate is deleted after final_metrics_to_return in some versions, ensure it's covered.

    final_metrics_to_return = {
        'pearson_all_median': np.nanmedian(results_accumulator['pearson_all']) if results_accumulator['pearson_all'] else np.nan,
        'pearson_affected_median': np.nanmedian(results_accumulator['pearson_affected']) if results_accumulator['pearson_affected'] else np.nan,
        'pearson_degs_median': np.nanmedian(results_accumulator['pearson_degs']) if results_accumulator['pearson_degs'] else np.nan,
        'mae_all_median': np.nanmedian(results_accumulator['mae_all']) if results_accumulator['mae_all'] else np.nan,
        'mae_affected_median': np.nanmedian(results_accumulator['mae_affected']) if results_accumulator['mae_affected'] else np.nan,
        'mae_degs_median': np.nanmedian(results_accumulator['mae_degs']) if results_accumulator['mae_degs'] else np.nan,
        'mse_all_median': np.nanmedian(results_accumulator['mse_all']) if results_accumulator['mse_all'] else np.nan,
        'mse_affected_median': np.nanmedian(results_accumulator['mse_affected']) if results_accumulator['mse_affected'] else np.nan,
        'mse_degs_median': np.nanmedian(results_accumulator['mse_degs']) if results_accumulator['mse_degs'] else np.nan,
        'fraction_genes_affected': any_affected_calculate.mean() if G > 0 and hasattr(any_affected_calculate, 'size') and any_affected_calculate.size == G and any_affected_calculate.dtype == bool else np.nan
    }
    del results_accumulator
    del any_affected_calculate # Ensure this is deleted
    return final_metrics_to_return

def sample_parameters(param_ranges): # Unchanged from original
    params = {}
    for param, range_info in param_ranges.items():
        if range_info['type'] == 'int':
            params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
        elif range_info['type'] == 'float':
            params[param] = np.random.uniform(range_info['min'], range_info['max'])
        elif range_info['type'] == 'log_float':
            log_min = np.log10(range_info['min'])
            log_max = np.log10(range_info['max'])
            params[param] = 10 ** np.random.uniform(log_min, log_max)
        elif range_info['type'] == 'log_int':
            log_min = np.log10(range_info['min'])
            log_max = np.log10(range_info['max'])
            log_value = np.random.uniform(log_min, log_max)
            params[param] = int(round(10 ** log_value))
        elif range_info['type'] == 'fixed':
            params[param] = range_info['value']
    return params

# Revised _pool_worker to include timing (matches spirit of original)
def _pool_worker_timed(task_info_dict):
    trial_id = task_info_dict['trial_id']
    params_dict = task_info_dict['params_dict']
    control_mu_from_main = task_info_dict['control_mu_main']
    all_theta_from_main = task_info_dict['all_theta_main']
    pert_mu_from_main = task_info_dict['pert_mu_main']

    # Add trial_id for RNG seeding within simulate_one_run_numpy
    params_for_sim = params_dict.copy() # Avoid modifying original params_dict
    params_for_sim['trial_id_for_rng'] = trial_id
    params_for_sim['control_mu'] = control_mu_from_main
    params_for_sim['all_theta'] = all_theta_from_main
    params_for_sim['pert_mu'] = pert_mu_from_main
    
    start_time = time.time()
    try:
        # Ensure all required keys by simulate_one_run_numpy are in params_for_sim
        # G, N0, Nk, P, p_effect, effect_factor are expected from sample_parameters
        metrics = simulate_one_run_numpy(**params_for_sim)
        execution_time = time.time() - start_time
        
        # Prepare results: original sampled params + metrics + supporting info
        # `params_dict` is the original sampled params.
        final_result = {**params_dict, **metrics, 
                        'execution_time': execution_time, 
                        'trial_id': trial_id, 'status': 'success'}
        return final_result
        
    except Exception as e:
        execution_time = time.time() - start_time
        # Define metrics_error_keys locally for safety
        metrics_error_keys_local = { 
            'pearson_all_median', 'pearson_affected_median',
            'mae_all_median', 'mae_affected_median',
            'mse_all_median', 'mse_affected_median',
            'fraction_genes_affected'
        }
        metrics_error = {key: np.nan for key in metrics_error_keys_local}

        final_result_error = {
            **params_dict, # original sampled params
            **metrics_error,
            'execution_time': execution_time,
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e)
        }
        return final_result_error

# And run_random_sweep needs to use _pool_worker_timed and prepare tasks for it:
def run_random_sweep_final(n_trials, param_ranges, output_dir, control_mu=None, all_theta=None, pert_mu=None, num_workers=None): # Renamed from run_random_sweep
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"random_sweep_results_{timestamp}.csv")
    error_log_file = os.path.join(output_dir, f"error_log_{timestamp}.txt")

    if num_workers is None:
        num_workers = os.cpu_count()
    print(f"Starting NumPy-based random parameter sweep with {n_trials} trials using {num_workers} worker processes (spawn context).")

    tasks_for_pool = []
    for i in range(n_trials):
        params = sample_parameters(param_ranges)
        tasks_for_pool.append({'trial_id': i, 'params_dict': params, 
                               'control_mu_main': control_mu, 'all_theta_main': all_theta,
                               'pert_mu_main': pert_mu})

    all_results_data = []
    
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        print("\nProcessing trials (NumPy version with worker timing):")
        with tqdm(total=n_trials, desc="Running Trials (NumPy)") as pbar:
            for result_from_worker in pool.imap_unordered(_pool_worker_timed, tasks_for_pool):
                all_results_data.append(result_from_worker)
                pbar.update(1)

    results_df = pd.DataFrame(all_results_data)
    
    success_count = results_df[results_df['status'] == 'success'].shape[0] if 'status' in results_df else 0
    failure_count = n_trials - success_count

    if failure_count > 0 and 'status' in results_df: # Ensure 'status' column exists
        failed_trials = results_df[results_df['status'] == 'failed']
        # Define metrics_error keys for excluding them from params logging
        metrics_error_keys = { 
            'pearson_all_median', 'pearson_affected_median',
            'mae_all_median', 'mae_affected_median',
            'mse_all_median', 'mse_affected_median',
            'fraction_genes_affected'
        }
        with open(error_log_file, 'a') as f:
            for _, row in failed_trials.iterrows():
                # Ensure 'trial_id' and 'error' exist in row, provide defaults if not
                trial_id_val = int(row.get('trial_id', -1))
                error_val = row.get('error', 'Unknown error')
                
                error_params = {k: v for k, v in row.items() if k not in metrics_error_keys and k not in ['status', 'error', 'trial_id', 'execution_time']}
                f.write(f"Trial {trial_id_val + 1} failed\n")
                f.write(f"Parameters: {str(error_params)}\n")
                f.write(f"Error: {error_val}\n")
                f.write("-" * 80 + "\n")

    if not results_df.empty:
        results_df.to_csv(csv_file, index=False)
        print(f"\nSweep complete. Results saved to '{csv_file}'")
    else:
        print("\nSweep complete. No results to save.")

    print(f"Success: {success_count}/{n_trials} trials")
    print(f"Failed: {failure_count}/{n_trials} trials")
    if failure_count > 0:
        print(f"See error log for details: {error_log_file}")
    return csv_file


if __name__ == "__main__":
    output_dir = "analyses/synthetic_simulations/random_sweep_results"
    
    # Load fitted parameters from parameter estimation files
    control_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/control_fitted_params.csv", index_col=0)
    perturbed_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/perturbed_fitted_params.csv", index_col=0)
    all_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/all_fitted_params.csv", index_col=0)
    
    print("Using theta estimates from all cells combined")
    
    # Extract parameters for simulation
    main_control_mu_loaded = control_params_df['mu'].values
    main_pert_mu_loaded = perturbed_params_df['mu'].values
    
    # Use theta (n) from all cells estimation
    main_all_theta_loaded = all_params_df['n'].values
    
    print(f"Using {len(main_control_mu_loaded)} genes for simulation.")

    param_ranges = {
        'G': {'type': 'int', 'min': 1000, 'max': 8192}, # 1000, 8192
        'N0': {'type': 'log_int', 'min': 10, 'max': 8192}, # 10, 8192
        'Nk': {'type': 'log_int', 'min': 10, 'max': 256}, # 10, 256
        'P': {'type': 'log_int', 'min': 10, 'max': 2000}, # 10, 2000
        'p_effect': {'type': 'float', 'min': 0.001, 'max': 0.1}, # 0.001, 0.1
        'effect_factor': {'type': 'float', 'min': 1.2, 'max': 5.0}, # 1.2, 5.0
        'B': {'type': 'float', 'min': 0.0, 'max': 2.0}, # 0.0, 2.0
        'mu_l': {'type': 'log_float', 'min': 0.2, 'max': 5.0} # 0.2, 5.0
    }
    
    n_trials = 10000
    # Call the final version of run_random_sweep
    print("Running the sweep...")
    print("Multiprocessing can be memory intensive, so if running into swap, reduce the number of workers.")
    csv_file = run_random_sweep_final(n_trials, param_ranges, output_dir, 
                                      control_mu=main_control_mu_loaded, 
                                      all_theta=main_all_theta_loaded,
                                      pert_mu=main_pert_mu_loaded,
                                      num_workers=64)
    print("\nDone doing the sweep. Plotting results...")

    # Run uv run python simulations/simulation_plots.py
    os.system(f"uv run python analyses/synthetic_simulations/paper_plots.py --results {csv_file}")