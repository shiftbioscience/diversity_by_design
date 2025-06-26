# ICML_workshop
Code for the ICML workshop preprint

## Getting started

Install uv to manage the dependencies
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies
```bash
uv sync
```

## Workflow to run analyses from paper

1. Get the data:

```bash
uv run data/norman19/get_data.py # Will take a few minutes
uv run data/replogle22/get_data.py # Will take a few minutes
```

2. Run synthetic data simulations:

```bash
uv run analyses/synthetic_simulations/parameter_estimation.py
uv run analyses/synthetic_simulations/random_sweep.py
```

Plots will be stored in `analyses/synthetic_simulations/paper_plots`.

3. Run simulations on real datasets:

Dataset can be `norman19` or `replogle22`.

```bash
cd analyses/real_data_simulations/
uv run simulation.py --dataset norman19
uv run simulation.py --dataset replogle22
```

Figures/results are in `analyses/real_data_simulations/<dataset>/`

4. Run the niche signal sensitivity analysis:

```bash
cd analyses/sensitivity_to_niche_signals/
uv run sensitivity_analysis.py --dataset norman19
uv run sensitivity_analysis.py --dataset replogle22
```

Figures/results are in `analyses/sensitivity_to_niche_signals/<dataset>/`

5. Train GEARS +/- WMSE loss & analyze the output:

```bash
cd analyses/modeling_metrics/
uv run GEARS_norman19.py # Include --multiprocessing if you have 6 GPUs available locally
uv run GEARS_replogle22.py # Include --multiprocessing if you have 6 GPUs available locally
uv run plotting.py --dataset norman19
uv run plotting.py --dataset replogle22
```

Figures/results are in `analyses/modeling_metrics/<dataset>/`.

*Note:* GEARS training only with MSE is very unstable so repeated runs may show numerical differences. WMSE actually increases the stability of training results. 
