import pandas as pd
import pickle
import os

def aggregate_simulation_metrics():
    datasets = ['norman19', 'replogle22']
    # These prefixes correspond to the simulation types and part of the .pkl filename
    simulation_prefixes = ['np', 'n0', 'k', 'd', 'E', 'B', 'g', 'mu_l']
    
    all_data_rows = []
    
    # Base directory where dataset folders (norman19, replogle22) are located.
    # This script is expected to be in analyses/real_data_simulations/,
    # and dataset folders (e.g., norman19/) are direct subdirectories.
    base_path = '.' 

    for dataset in datasets:
        for sim_prefix in simulation_prefixes:
            file_name = f"{sim_prefix}_aggregate_vals.pkl"
            # Construct path relative to the script's location
            file_path = os.path.join(base_path, dataset, file_name)
            
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)

            for metric_key, value in data_dict.items():
                # Normalize metric name (e.g., 'corr_delta_dict' -> 'corr_delta', 'mae' -> 'mae')
                normalized_metric_name = metric_key.replace('_dict', '')
                
                row = {
                    'Dataset': dataset,
                    'Simulation': sim_prefix,
                    'Metric': normalized_metric_name,
                    'Value': value
                }
                all_data_rows.append(row)


    df = pd.DataFrame(all_data_rows)
    
    # Output CSV path will also be relative to this script's location
    output_csv_path = os.path.join(base_path, 'aggregated_simulation_metrics.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Aggregated metrics saved to {output_csv_path}")

# Added format_value function
def format_value(x):
    if pd.isna(x):
        return '-' # Changed from '' to '-'
    val_str = f"{x:.2f}"
    if abs(x) > 0.1:
        return f"\\textbf{{{val_str}}}"
    return val_str

# Added generate_latex_table function
def generate_latex_table():
    csv_path = './aggregated_simulation_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run aggregate_simulation_metrics first to generate it.")
        return

    df = pd.read_csv(csv_path)

    metric_name_map = {
        'corr_delta': 'Pearson Delta',
        'mae': 'MAE',
        'mse': 'MSE'
    }
    df['Metric'] = df['Metric'].map(metric_name_map).fillna(df['Metric'])

    simulation_order = ['E', 'g', 'mu_l', 'np', 'd', 'n0', 'B', 'k']
    simulation_latex_symbols = {
        'E': '$\epsilon$',
        'g': '$g$',
        'mu_l': '$\mu_l$',
        'np': '$n_p$',
        'd': '$\delta$',
        'n0': '$n_0$',
        'B': '$\\beta$',
        'k': '$k$'
    }

    try:
        pivot_df = df.pivot_table(index=['Metric', 'Dataset'], columns='Simulation', values='Value')
    except Exception as e:
        print(f"Error pivoting DataFrame: {e}. Check CSV structure.")
        return

    ordered_cols = [col for col in simulation_order if col in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]
    
    formatted_df = pivot_df.applymap(format_value)

    latex_string = "\\begin{table*}[ht]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Spearman correlations between metrics and simulation parameters. Absolute correlation values higher than 0.1 are in \\textbf{bold}.}\n"
    latex_string += "\\label{tab:simulation-results-aggregated}\n"
    
    num_sim_params = len(ordered_cols)
    col_format = "@{}c" + "c" * (1 + num_sim_params) + "@{}"
    latex_string += f"\\begin{{tabular}}{{{col_format}}}\n"
    latex_string += "\\toprule\n"
    
    header_cols_latex = [simulation_latex_symbols[col] for col in ordered_cols]
    latex_string += "\\textbf{Metric} & \\textbf{Dataset} & " + " & ".join(f'\\textbf{{{s}}}' for s in header_cols_latex) + " \\\\ \n"
    latex_string += "\\midrule\n"

    current_metric = None
    metrics_processed = [] # To track metrics for midrule placement
    unique_metrics_in_data = formatted_df.index.get_level_values('Metric').unique()

    for i, ((metric, dataset), row_data) in enumerate(formatted_df.iterrows()):
        row_str = ""
        if metric != current_metric:
            if current_metric is not None: # Add midrule if changing metric, but not before the first one
                latex_string += "\\midrule\n" 
            row_str += f"\\multirow{{2}}{{*}}{{{metric}}} & "
            current_metric = metric
            metrics_processed.append(metric)
        else:
            row_str += " & " 
        
        row_str += f"{dataset.replace('norman19', 'Norman19').replace('replogle22', 'Replogle22')} & "
        row_str += " & ".join(str(val) for val in row_data.values) + " \\\\ \n"
        latex_string += row_str

    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table*}\n"

    print("\nGenerated LaTeX Table:\n")
    print(latex_string)

if __name__ == '__main__':
    aggregate_simulation_metrics()
    generate_latex_table() # Call generate_latex_table after aggregate_simulation_metrics
