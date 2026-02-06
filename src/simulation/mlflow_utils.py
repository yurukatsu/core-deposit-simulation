"""MLFlow integration utilities."""

import json
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
import numpy as np
from coredeposit import EstimationResult
from loguru import logger

from simulation.config import Config, MLFlowConfig
from simulation.simulator import WindowResult


def setup_mlflow(config: MLFlowConfig) -> str:
    """Setup MLFlow tracking.

    Args:
        config: MLFlow configuration

    Returns:
        Experiment ID
    """
    mlflow.set_tracking_uri(config.tracking_uri)
    logger.debug(f"MLFlow tracking URI: {config.tracking_uri}")

    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(config.experiment_name)

    client = mlflow.tracking.MlflowClient()

    if experiment is None:
        # Create new experiment with artifact location if specified
        logger.debug(f"Creating new experiment: {config.experiment_name}")
        if config.artifact_location:
            logger.debug(f"Artifact location: {config.artifact_location}")
        experiment_id = mlflow.create_experiment(
            config.experiment_name,
            artifact_location=config.artifact_location,
            tags=config.experiment_tags if config.experiment_tags else None,
        )
    else:
        experiment_id = experiment.experiment_id
        logger.debug(
            f"Using existing experiment: {config.experiment_name} (ID: {experiment_id})"
        )

        # Update experiment tags if provided
        if config.experiment_tags:
            for key, value in config.experiment_tags.items():
                client.set_experiment_tag(experiment_id, key, value)

    # Set experiment description if provided
    if config.experiment_description:
        client.set_experiment_tag(
            experiment_id, "mlflow.note.content", config.experiment_description
        )
        logger.debug("Set experiment description")

    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@contextmanager
def mlflow_run(config: Config) -> Iterator[mlflow.ActiveRun]:
    """Context manager for MLFlow run.

    Args:
        config: Full simulation configuration

    Yields:
        Active MLFlow run
    """
    mlflow_config = config.mlflow
    setup_mlflow(mlflow_config)

    tags = {
        "estimator": config.model.estimator,
        "t_train": str(config.simulation.t_train),
        "t_test": str(config.simulation.t_test),
        "t_gap": str(config.simulation.t_gap),
        **mlflow_config.tags,
    }

    with mlflow.start_run(
        run_name=mlflow_config.run_name,
        tags=tags,
        description=mlflow_config.description,
    ) as run:
        # Log configuration as params
        _log_config_params(config)
        yield run


def _log_config_params(config: Config) -> None:
    """Log configuration as MLFlow parameters."""
    # Data config
    mlflow.log_param("data.path", str(config.data.path))
    mlflow.log_param("data.volume_column", config.data.volume_column)
    mlflow.log_param("data.inflow_column", config.data.inflow_column)

    # Model config
    mlflow.log_param("model.estimator", config.model.estimator)
    if config.model.fix_m is not None:
        mlflow.log_param("model.fix_m", config.model.fix_m)

    if config.model.estimator == "mcmc":
        mlflow.log_param("model.num_warmup", config.model.num_warmup)
        mlflow.log_param("model.num_samples", config.model.num_samples)
        mlflow.log_param("model.num_chains", config.model.num_chains)
        mlflow.log_param("model.ar_errors", config.model.ar_errors)
        mlflow.log_param("model.likelihood", config.model.likelihood)
        mlflow.log_param("model.device", config.model.device)

    # Simulation config
    mlflow.log_param("simulation.t_train", config.simulation.t_train)
    mlflow.log_param("simulation.t_test", config.simulation.t_test)
    mlflow.log_param("simulation.t_gap", config.simulation.t_gap)
    if config.simulation.max_windows is not None:
        mlflow.log_param("simulation.max_windows", config.simulation.max_windows)
    if config.simulation.start_index is not None:
        mlflow.log_param("simulation.start_index", config.simulation.start_index)
    if config.simulation.end_index is not None:
        mlflow.log_param("simulation.end_index", config.simulation.end_index)


def log_config_artifact(config: Config, config_path: str | None = None) -> None:
    """Log configuration as artifact.

    Args:
        config: Configuration object
        config_path: Optional path to original config file
    """
    import tempfile
    from pathlib import Path

    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save config as JSON
        json_path = Path(tmpdir) / "config.json"
        with open(json_path, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2, default=str)
        mlflow.log_artifact(str(json_path), artifact_path="config")

        # Save config as YAML
        yaml_path = Path(tmpdir) / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(
                config.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        mlflow.log_artifact(str(yaml_path), artifact_path="config")

        # Copy original config file if provided
        if config_path:
            original_path = Path(config_path)
            if original_path.exists():
                mlflow.log_artifact(str(original_path), artifact_path="config")
                logger.debug(f"Logged original config: {original_path.name}")


def log_model_artifact(result: WindowResult, estimator_type: str) -> None:
    """Log model estimation result as artifact.

    Args:
        result: WindowResult containing estimation result
        estimator_type: Type of estimator ('nls', 'map', or 'mcmc')
    """
    import tempfile
    from pathlib import Path

    if result.result is None:
        logger.warning("No estimation result to log")
        return

    estimation_result = result.result
    window_id = result.window_id

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        artifact_path = f"models/window_{window_id:03d}"

        # Save parameter summary as JSON (for all estimator types)
        params_summary = {}
        for name, value in estimation_result.params.items():
            if hasattr(value, "mean"):
                # MCMC samples
                params_summary[name] = {
                    "mean": float(np.mean(value)),
                    "std": float(np.std(value)),
                    "median": float(np.median(value)),
                    "q5": float(np.percentile(value, 5)),
                    "q95": float(np.percentile(value, 95)),
                }
            else:
                # Point estimate
                params_summary[name] = {"value": float(value)}

        params_path = tmpdir_path / "params.json"
        with open(params_path, "w") as f:
            json.dump(params_summary, f, indent=2)
        mlflow.log_artifact(str(params_path), artifact_path=artifact_path)

        # For MCMC, save InferenceData as NetCDF
        if estimator_type == "mcmc" and "mcmc" in estimation_result.diagnostics:
            try:
                import arviz as az

                mcmc = estimation_result.diagnostics["mcmc"]
                idata = az.from_numpyro(mcmc)

                netcdf_path = tmpdir_path / "inference_data.nc"
                idata.to_netcdf(str(netcdf_path))
                mlflow.log_artifact(str(netcdf_path), artifact_path=artifact_path)
                logger.debug(f"Logged MCMC InferenceData for window {window_id}")
            except Exception as e:
                logger.warning(f"Failed to save InferenceData: {e}")

        logger.debug(f"Logged model artifacts for window {window_id}")


def log_window_result(result: WindowResult, prefix: str = "") -> None:
    """Log a single window result to MLFlow.

    Args:
        result: Window simulation result
        prefix: Prefix for metric names
    """
    p = f"{prefix}_" if prefix else ""

    # Log metrics with step = window_id
    step = result.window_id
    for metric_name, value in result.metrics.items():
        if not np.isnan(value):
            mlflow.log_metric(f"{p}{metric_name}", value, step=step)

    # Log estimated parameters
    _log_estimation_params(result.result, step, prefix)


def _log_estimation_params(
    result: EstimationResult, step: int, prefix: str = ""
) -> None:
    """Log estimated parameters from EstimationResult."""
    p = f"{prefix}_" if prefix else ""

    for param_name in ["lambda", "gamma", "w1", "h", "m"]:
        if param_name in result.params:
            val = result.params[param_name]
            # Handle MCMC (array) vs NLS (scalar)
            if hasattr(val, "mean"):
                mlflow.log_metric(
                    f"{p}param_{param_name}", float(val.mean()), step=step
                )
            else:
                mlflow.log_metric(f"{p}param_{param_name}", float(val), step=step)


def log_summary_metrics(results: list[WindowResult]) -> None:
    """Log summary metrics across all windows.

    Args:
        results: List of window results
    """
    if not results:
        return

    # Aggregate metrics
    metric_names = ["insample_rmse", "insample_mape", "oosample_rmse", "oosample_mape"]

    for name in metric_names:
        values = [r.metrics.get(name, np.nan) for r in results]
        valid_values = [v for v in values if not np.isnan(v)]

        if valid_values:
            mlflow.log_metric(f"mean_{name}", float(np.mean(valid_values)))
            mlflow.log_metric(f"std_{name}", float(np.std(valid_values)))
            mlflow.log_metric(f"median_{name}", float(np.median(valid_values)))

    # Log number of windows
    mlflow.log_metric("n_windows", len(results))


def log_plots_artifact(results: list[WindowResult], has_mcmc_ci: bool = False) -> None:
    """Log prediction plots as MLFlow artifacts.

    Args:
        results: List of window results
        has_mcmc_ci: Whether results include MCMC credible intervals
    """
    import tempfile
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt

    from simulation.plots import (
        plot_all_windows,
        plot_combined_timeseries,
        plot_combined_timeseries_with_ci,
        plot_window_predictions,
        plot_window_predictions_with_ci,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save individual window plots
        plots_dir = tmppath / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            if has_mcmc_ci:
                # Use CI version for MCMC
                for result in results:
                    fig = plot_window_predictions_with_ci(result)
                    filepath = plots_dir / f"window_{result.window_id:03d}.png"
                    fig.savefig(filepath, dpi=150, bbox_inches="tight")
                    plt.close(fig)
            else:
                plot_all_windows(results, plots_dir)
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")
        except Exception as e:
            warnings.warn(f"Failed to log window plots: {e}", UserWarning, stacklevel=2)

        # Save combined plot
        try:
            if has_mcmc_ci:
                fig = plot_combined_timeseries_with_ci(results)
            else:
                fig = plot_combined_timeseries(results)
            combined_path = tmppath / "combined_predictions.png"
            fig.savefig(combined_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(combined_path))
        except Exception as e:
            warnings.warn(
                f"Failed to log combined plot: {e}", UserWarning, stacklevel=2
            )


def log_results_artifact(
    results: list[WindowResult], filename: str = "results.json"
) -> None:
    """Log results as JSON artifact.

    Args:
        results: List of window results
        filename: Artifact filename
    """
    import tempfile
    import warnings
    from pathlib import Path

    data = []
    for r in results:
        entry = {
            "window_id": r.window_id,
            # Index-based bounds
            "train_start": r.train_start,
            "train_end": r.train_end,
            "test_start": r.test_start,  # Same as train_end
            "test_end": r.test_end,
            # Date-based bounds (if available)
            "train_start_date": r.train_start_date.isoformat()
            if r.train_start_date
            else None,
            "train_end_date": r.train_end_date.isoformat()
            if r.train_end_date
            else None,
            "test_start_date": r.test_start_date.isoformat()
            if r.test_start_date
            else None,
            "test_end_date": r.test_end_date.isoformat() if r.test_end_date else None,
            # Metrics and params
            "metrics": r.metrics,
            "params": _extract_params(r.result),
        }
        data.append(entry)

    # Write to temp file and log
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        try:
            mlflow.log_artifact(str(path))
        except Exception as e:
            warnings.warn(
                f"Failed to log artifact to MLFlow: {e}. "
                f"Results were logged as metrics but artifact upload failed.",
                UserWarning,
                stacklevel=2,
            )


def _extract_params(result: EstimationResult) -> dict[str, Any]:
    """Extract parameters as serializable dict.

    For MCMC results, includes: mean, std, median, q5, q95, n_eff, r_hat.
    For NLS/MAP results, includes the scalar value only.
    """
    params = {}

    # Get MCMC diagnostics for n_eff and r_hat if available
    mcmc_summary = None
    if "mcmc" in result.diagnostics:
        try:
            import arviz as az

            idata = az.from_numpyro(result.diagnostics["mcmc"])
            mcmc_summary = az.summary(idata)
        except Exception:
            pass

    # Extract all parameters from result.params
    for name, val in result.params.items():
        if hasattr(val, "mean"):
            # MCMC: val is an array of samples
            param_stats = {
                "mean": float(val.mean()),
                "std": float(val.std()),
                "median": float(np.median(val)),
                "q5": float(np.percentile(val, 5)),
                "q95": float(np.percentile(val, 95)),
            }

            # Add n_eff and r_hat from ArviZ summary if available
            if mcmc_summary is not None and name in mcmc_summary.index:
                try:
                    param_stats["n_eff"] = float(mcmc_summary.loc[name, "ess_bulk"])
                    param_stats["r_hat"] = float(mcmc_summary.loc[name, "r_hat"])
                except (KeyError, ValueError):
                    pass

            params[name] = param_stats
        else:
            # NLS/MAP: val is a scalar
            params[name] = float(val)

    return params


def log_mcmc_diagnostics(results: list[WindowResult]) -> None:
    """Log MCMC diagnostic plots (trace, posterior, pair, forest) for each window.

    Args:
        results: List of window results (must be MCMC)
    """
    import tempfile
    import warnings
    from pathlib import Path

    import arviz as az
    import matplotlib.pyplot as plt

    from simulation.plots import (
        plot_mcmc_forest,
        plot_mcmc_pair,
        plot_mcmc_posterior,
        plot_mcmc_trace,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        for result in results:
            if result.result is None:
                continue

            # Check if this is MCMC result
            if "mcmc" not in result.result.diagnostics:
                continue

            # Convert to ArviZ InferenceData
            try:
                idata = az.from_numpyro(result.result.diagnostics["mcmc"])
            except Exception as e:
                warnings.warn(
                    f"Failed to convert MCMC result to ArviZ: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            window_dir = tmppath / f"mcmc_window_{result.window_id:03d}"
            window_dir.mkdir(parents=True, exist_ok=True)

            # Trace plot
            try:
                fig = plot_mcmc_trace(idata)
                fig.savefig(window_dir / "trace.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(
                    f"Failed to generate trace plot: {e}", UserWarning, stacklevel=2
                )

            # Posterior plot
            try:
                fig = plot_mcmc_posterior(idata)
                fig.savefig(window_dir / "posterior.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(
                    f"Failed to generate posterior plot: {e}", UserWarning, stacklevel=2
                )

            # Pair plot
            try:
                fig = plot_mcmc_pair(idata)
                fig.savefig(window_dir / "pair.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(
                    f"Failed to generate pair plot: {e}", UserWarning, stacklevel=2
                )

            # Forest plot
            try:
                fig = plot_mcmc_forest(idata)
                fig.savefig(window_dir / "forest.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                warnings.warn(
                    f"Failed to generate forest plot: {e}", UserWarning, stacklevel=2
                )

            # Log all plots for this window
            try:
                mlflow.log_artifacts(
                    str(window_dir), artifact_path=f"mcmc/window_{result.window_id:03d}"
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to log MCMC artifacts: {e}", UserWarning, stacklevel=2
                )
