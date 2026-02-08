# Core Deposit Simulation

Rolling window simulation framework for core deposit balance regression models using Bayesian MCMC and NLS estimation.

## Features

- **Rolling Window Simulation**: Walk-forward validation with configurable training/test periods
- **Multiple Estimators**: NLS, MAP, and full Bayesian MCMC (via NumPyro)
- **MLFlow Integration**: Experiment tracking, artifact storage (S3 support), model logging
- **Result Loading**: Load tracked results and MCMC samples from MLFlow for re-analysis
- **Parallel Execution**: Multi-core CPU support for MCMC chains and window parallelization
- **Covariate Support**: External covariates (e.g., interest rates) with automatic date alignment

## Installation

```bash
# Clone the repository
git clone https://github.com/yurukatsu/core-deposit-simulation.git
cd core-deposit-simulation

# Install with uv
uv sync
```

## Quick Start

```bash
# Validate configuration
uv run simulate validate configs/example.yaml

# Run simulation (without MLFlow)
uv run simulate run configs/example.yaml --no-mlflow

# Run with MLFlow logging
uv run simulate run configs/example.yaml

# Run with all artifacts
uv run simulate run configs/example.yaml --save-plots --save-models
```

## Configuration

Configuration is done via YAML files. See `configs/example.yaml` for all options.

### Basic Structure

```yaml
data:
  path: data/sample.csv
  volume_column: volume
  inflow_column: inflow
  date_column: date
  covariate:
    path: data/covariate.csv
    columns: [rate_1m, rate_10y]
    date_column: date

model:
  estimator: mcmc          # nls, map, or mcmc
  num_warmup: 1000
  num_samples: 2000
  num_chains: 4
  device: cpu              # cpu or gpu
  likelihood: studentt     # normal or studentt

simulation:
  t_train: 60              # Training period (months)
  t_test: 12               # Test period (0 = in-sample only)
  t_gap: 1                 # Sliding step size
  max_windows: null        # Limit number of windows (null = no limit)
  start_date: 2000-01-01

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: core-deposit-simulation
  description: |
    Experiment description (supports markdown)
  artifact_location: s3://bucket/artifacts
  tags:
    version: "1.0"
  experiment_tags:
    project: core-deposit
```

## CLI Options

```
Usage: simulate run [OPTIONS] CONFIG_PATH

Options:
  --dry-run / --no-dry-run     Print config without running
  --no-mlflow                  Disable MLFlow logging
  --save-plots                 Save prediction plots to MLFlow
  --save-models                Save model artifacts (params + MCMC samples)
  --plot-dir PATH              Save plots to local directory
  -j, --jobs INTEGER           Number of parallel workers
  -v, --verbose                Verbose output (DEBUG level)
  --log-file PATH              Path to log file
```

## Project Structure

```
core-deposit-simulation/
├── src/simulation/
│   ├── cli.py           # CLI entry point
│   ├── config.py        # Pydantic configuration models
│   ├── simulator.py     # Rolling window simulation logic
│   ├── plots.py         # Visualization functions
│   ├── mlflow_utils.py  # MLFlow logging utilities
│   └── mlflow_loader.py # MLFlow result loading utilities
├── configs/
│   ├── example.yaml     # Example configuration
│   └── boj_mcmc.yaml    # BOJ data MCMC configuration
└── data/
    ├── boj.csv          # Main data file
    └── covariate.csv    # Covariate data
```

## MLFlow Artifacts

When using `--save-plots` and `--save-models`:

```
artifacts/
├── config/
│   ├── config.json
│   ├── config.yaml
│   └── original.yaml
├── models/
│   └── window_000/
│       ├── params.json        # Parameter summary
│       └── inference_data.nc  # MCMC samples (NetCDF)
├── plots/
│   ├── windows/
│   │   └── window_000.png
│   └── combined_timeseries.png
└── results/
    └── results.json
```

## Loading Results from MLFlow

### CLI Commands

```bash
# List experiments
uv run simulate list-experiments --uri http://localhost:5000

# List runs in an experiment
uv run simulate list-runs core-deposit-simulation --uri http://localhost:5000

# Load a run and download artifacts
uv run simulate load-run <run_id> --uri http://localhost:5000 -o ./artifacts

# Show posterior summary for a specific window
uv run simulate show-posterior <run_id> 0 --uri http://localhost:5000
```

### Python API

```python
from simulation import load_run, load_inference_data, predict_from_posterior

# Load a complete run with all artifacts
loaded = load_run(
    run_id="abc123...",
    tracking_uri="http://localhost:5000",
    load_inference_data=True,
)

# Access configuration and metrics
print(loaded.config)
print(loaded.metrics)

# Get MCMC InferenceData for a specific window
idata = loaded.get_inference_data(window_id=0)

# Get posterior samples for a parameter
lambda_samples = loaded.get_posterior_samples("lambda", window_id=0)
all_lambda = loaded.get_posterior_samples("lambda")  # Dict[window_id, samples]

# Generate predictions from posterior samples
import numpy as np
pred = predict_from_posterior(
    idata=idata,
    inflow=np.array([...]),  # Your inflow data
    V0=1000.0,               # Initial volume
    ci_prob=0.95,
)
# Returns: {"mean": ..., "median": ..., "lower": ..., "upper": ..., "samples": ...}

# Load just the InferenceData for one window (faster)
idata = load_inference_data(
    run_id="abc123...",
    window_id=0,
    tracking_uri="http://localhost:5000",
)
```

## Development

```bash
# Format code
uv run poe format

# Run linter
uv run poe lint

# Run all checks
uv run poe all
```

## License

MIT
