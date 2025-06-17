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

Get the data:

```bash
uv run data/norman19/get_data.py
uv run data/replogle22/get_data.py
```

Run simulations:

```bash
uv run analyses/simulations/parameter_estimation.py
uv run analyses/simulations/random_sweep.py
```

Plots will be stored in `analyses/simulations/paper_plots`.

Run the real data analysis:

```bash
uv run analyses/norman19/analysis.py
```



