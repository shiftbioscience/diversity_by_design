# Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics

[Gabriel Mejía](https://scholar.google.com/citations?hl=es&user=yh69hnYAAAAJ)<sup>1</sup>\*, [Henry E. Miller](https://scholar.google.com/citations?user=Sw9t-h0AAAAJ&hl=en)<sup>1</sup>\*, [Francis J. A. Leblanc](https://scholar.google.com/citations?user=yFI4c_0AAAAJ&hl=en)<sup>1</sup>, [Bo Wang](https://scholar.google.ca/citations?user=37FDILIAAAAJ&hl=en)<sup>2</sup>, [Brendan Swain](https://scholar.google.com/citations?user=UH0zyDoAAAAJ&hl=en)<sup>1</sup>, [Lucas Paulo de Lima Camillo](https://scholar.google.com/citations?user=qEpmbq8AAAAJ&hl=en)<sup>1</sup>

<br/>
<font size="1"><sup>*</sup>Equal contribution.</font><br/>
<font size="1"><sup>1 </sup> Shift Bioscience, Cambridge, UK.</font><br/>
<font size="1"><sup>2 </sup> University of Toronto, Vector Institute, Toronto, Canada.</font><br/>

- ArXiv Preprint [here](https://arxiv.org/abs/2506.22641)

### Abstract

Recent benchmarks reveal that models for single-cell perturbation response are often outperformed by simply predicting the dataset mean. We trace this anomaly to a metric artifact: control-referenced deltas and unweighted error metrics reward mode collapse whenever the control is biased or the biological signal is sparse. Large-scale *in silico* simulations and analysis of two real-world perturbation datasets confirm that shared reference shifts, not genuine biological change, drives high performance in these evaluations. We introduce differentially expressed gene (DEG)–aware metrics, weighted mean-squared error (WMSE) and weighted delta $R^{2}~~(R^{2}_{w}(\Delta))$ with respect to all perturbations, that measure error in niche signals with high sensitivity. We further introduce negative and positive performance baselines to calibrate these metrics. With these improvements, the mean baseline sinks to null performance while genuine predictors are correctly rewarded. Finally, we show that using WMSE as a loss function reduces mode collapse and improves model performance.

![Graphical_abstrac](https://github.com/user-attachments/assets/9778527b-d8f4-4e90-9576-68e9479d9491)

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
