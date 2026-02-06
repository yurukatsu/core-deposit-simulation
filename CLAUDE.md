# CLAUDE.md

This file provides context for Claude Code when working on this project.

## Project Overview

Core Deposit Simulation is a rolling window simulation framework for bank deposit balance regression models. It supports Bayesian MCMC estimation via NumPyro/JAX and integrates with MLFlow for experiment tracking.

## Tech Stack

- **Python 3.13+**
- **NumPyro/JAX**: Bayesian MCMC estimation
- **Pydantic**: Configuration validation
- **Typer**: CLI framework
- **MLFlow**: Experiment tracking and artifact storage
- **ArviZ**: MCMC diagnostics and visualization
- **Loguru**: Logging

## Key Files

| File | Purpose |
|------|---------|
| `src/simulation/cli.py` | CLI entry point, JAX device setup |
| `src/simulation/config.py` | Pydantic models for YAML configuration |
| `src/simulation/simulator.py` | Rolling window simulation logic |
| `src/simulation/plots.py` | Matplotlib visualization functions |
| `src/simulation/mlflow_utils.py` | MLFlow integration utilities |
| `configs/example.yaml` | Reference configuration with all options |

## Architecture

### Rolling Window Simulation

```
Data: [========================================]
       Window 0: [train][test]
       Window 1:    [train][test]
       Window 2:       [train][test]
                  ...
```

- `t_train`: Training period length
- `t_test`: Test period length (0 = in-sample only)
- `t_gap`: Sliding step size
- `max_windows`: Limit number of windows

### Estimation Flow

1. Load data and covariates
2. For each window:
   - Prepare training data
   - Initialize from NLS (optional, for MCMC)
   - Fit model (NLS/MAP/MCMC)
   - Compute predictions and credible intervals
   - Calculate metrics (RMSE, MAPE)
3. Log results to MLFlow

## Important Patterns

### JAX Device Configuration

JAX device must be configured before import:

```python
# cli.py - must be before JAX import
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count}"
```

### MCMC Prediction

For MCMC, predictions use the posterior predictive mean (not the model evaluated at mean parameters):

```python
# Correct: mean(V_model(samples))
pred = estimator.predict(data, result, uncertainty=True)
prediction = pred["mean"]

# Wrong: V_model(mean(samples)) - doesn't propagate uncertainty correctly
```

### Credible Intervals

95% credible intervals are computed from the posterior predictive distribution:

```python
pred = estimator.predict(data, result, uncertainty=True, ci_prob=0.95)
ci = {"lower": pred["lower"], "upper": pred["upper"]}
```

## Common Tasks

### Adding a New Config Option

1. Add field to appropriate Pydantic model in `config.py`
2. Update `example.yaml` with documentation
3. If MCMC-specific, add to MLFlow logging in `mlflow_utils.py`

### Adding a New Plot Type

1. Add function to `plots.py`
2. Handle `t_test=0` case (no out-of-sample data)
3. Call from `cli.py` or `mlflow_utils.py`

### Modifying Simulation Logic

Key methods in `simulator.py`:
- `iter_windows()`: Yields window indices
- `run_window()`: Runs single window estimation
- `_compute_mcmc_ci()`: Computes credible intervals for MCMC

## Testing

```bash
# Validate config
uv run simulate validate configs/example.yaml

# Dry run (print config without executing)
uv run simulate run configs/example.yaml --dry-run

# Quick test with verbose output
uv run simulate run configs/example.yaml --no-mlflow -v
```

## Code Style

- Use `loguru` for logging (not print)
- Use type hints for all function signatures
- Use Pydantic for configuration validation
- Prefer `np.nan` over `None` for missing numeric values

## Dependencies

The `coredeposit` package is installed from:
```
https://github.com/yurukatsu/core-deposit-analysis.git
```

Key classes from `coredeposit`:
- `CoreDepositData`: Input data container
- `MCMCEstimator`: Bayesian MCMC estimator
- `NLSEstimator`: Non-linear least squares estimator
- `EstimationResult`: Estimation results container
