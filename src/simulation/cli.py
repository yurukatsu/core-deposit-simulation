"""CLI entry point for core deposit simulation."""

import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from simulation.config import load_config
from simulation.mlflow_utils import (
    log_config_artifact,
    log_mcmc_diagnostics,
    log_model_artifact,
    log_plots_artifact,
    log_results_artifact,
    log_summary_metrics,
    log_window_result,
    mlflow_run,
)
from simulation.simulator import RollingWindowSimulator

app = typer.Typer(help="Core Deposit Simulation CLI")


def _setup_jax_device(device: str, num_chains: int) -> None:
    """Configure JAX device settings.

    Must be called before any JAX operations.

    Args:
        device: 'cpu' or 'gpu'
        num_chains: Number of MCMC chains (for CPU parallelism)
    """
    import jax

    if device == "cpu":
        # Set CPU device count for parallel chains
        cpu_count = min(os.cpu_count() or 1, max(num_chains, 4))
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count}"
        jax.config.update("jax_platform_name", "cpu")
        logger.debug(f"JAX configured for CPU with {cpu_count} devices")
    else:  # gpu
        jax.config.update("jax_platform_name", "gpu")
        # Check if GPU is available
        try:
            devices = jax.devices("gpu")
            logger.debug(f"JAX configured for GPU: {len(devices)} device(s) available")
        except RuntimeError:
            logger.warning("GPU requested but not available, falling back to CPU")
            jax.config.update("jax_platform_name", "cpu")


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure loguru logging.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional path to log file
    """
    # Remove default handler
    logger.remove()

    # Console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler (if specified)
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )
        logger.info(f"Logging to file: {log_file}")


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML configuration file")
    ],
    dry_run: Annotated[
        bool, typer.Option(help="Print configuration without running")
    ] = False,
    no_mlflow: Annotated[bool, typer.Option(help="Disable MLFlow logging")] = False,
    save_plots: Annotated[
        bool, typer.Option("--save-plots", help="Save prediction plots to MLFlow")
    ] = False,
    save_models: Annotated[
        bool,
        typer.Option(
            "--save-models",
            help="Save model artifacts to MLFlow (params + MCMC samples)",
        ),
    ] = False,
    plot_dir: Annotated[
        Path | None, typer.Option("--plot-dir", help="Save plots to local directory")
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option(
            "-j", "--jobs", help="Number of parallel workers (overrides config)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Verbose output (DEBUG level)")
    ] = False,
    log_file: Annotated[
        Path | None, typer.Option("--log-file", help="Path to log file")
    ] = None,
) -> None:
    """Run rolling window simulation."""
    setup_logging(verbose=verbose, log_file=log_file)

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Override n_jobs from CLI if specified
    if jobs is not None:
        config.simulation.n_jobs = jobs

    if dry_run:
        logger.info("Dry run mode - printing configuration")
        typer.echo("Configuration:")
        typer.echo(config.model_dump_json(indent=2))
        return

    # Configure JAX device for MCMC
    if config.model.estimator == "mcmc":
        _setup_jax_device(config.model.device, config.model.num_chains)

    # Create simulator
    logger.info("Creating simulator")
    simulator = RollingWindowSimulator(config)
    simulator.load_data()

    n_windows = sum(1 for _ in simulator.iter_windows())
    logger.info(f"Data loaded: {simulator.data_length} observations")
    logger.info(f"Number of windows: {n_windows}")

    if no_mlflow:
        logger.info("Running without MLFlow")
        n_jobs = config.simulation.n_jobs

        if n_jobs > 1:
            # Use parallel execution via simulator.run()
            logger.info(f"Running {n_windows} windows in parallel (n_jobs={n_jobs})")
            results = simulator.run()
        else:
            # Sequential execution with progress bar
            results = []
            with typer.progressbar(
                simulator.iter_windows(), length=n_windows, label="Running"
            ) as progress:
                for window_id, train_start, train_end, test_end in progress:
                    logger.debug(
                        f"Window {window_id}: train=[{train_start}:{train_end}], test=[{train_end}:{test_end}]"
                    )
                    result = simulator.run_window(train_start, train_end, test_end)
                    result.window_id = window_id
                    results.append(result)
                    oos_rmse = result.metrics["oosample_rmse"]
                    if np.isnan(oos_rmse):
                        logger.debug(
                            f"Window {window_id} completed: IS RMSE={result.metrics['insample_rmse']:.4f} (in-sample only)"
                        )
                    else:
                        logger.debug(
                            f"Window {window_id} completed: OOS RMSE={oos_rmse:.4f}"
                        )

        _print_summary(results)

        # Save plots locally if requested
        if plot_dir:
            _save_plots_local(
                results, plot_dir, is_mcmc=config.model.estimator == "mcmc"
            )
    else:
        logger.info("Running with MLFlow logging")
        n_jobs = config.simulation.n_jobs

        with mlflow_run(config) as active_run:
            logger.info(f"MLFlow run ID: {active_run.info.run_id}")

            # Log config as artifact
            log_config_artifact(config, str(config_path))

            if n_jobs > 1:
                # Use parallel execution via simulator.run()
                logger.info(
                    f"Running {n_windows} windows in parallel (n_jobs={n_jobs})"
                )
                results = simulator.run()

                # Log results to MLFlow after completion
                for result in results:
                    log_window_result(result)
                    if save_models:
                        log_model_artifact(result, config.model.estimator)
            else:
                # Sequential execution with progress bar
                results = []
                with typer.progressbar(
                    simulator.iter_windows(), length=n_windows, label="Running"
                ) as progress:
                    for window_id, train_start, train_end, test_end in progress:
                        logger.debug(
                            f"Window {window_id}: train=[{train_start}:{train_end}], test=[{train_end}:{test_end}]"
                        )

                        result = simulator.run_window(train_start, train_end, test_end)
                        result.window_id = window_id
                        results.append(result)

                        # Log to MLFlow
                        log_window_result(result)
                        if save_models:
                            log_model_artifact(result, config.model.estimator)
                        oos_rmse = result.metrics["oosample_rmse"]
                        if np.isnan(oos_rmse):
                            logger.debug(
                                f"Window {window_id} completed: IS RMSE={result.metrics['insample_rmse']:.4f} (in-sample only)"
                            )
                        else:
                            logger.debug(
                                f"Window {window_id} completed: OOS RMSE={oos_rmse:.4f}"
                            )

            # Log summary
            log_summary_metrics(results)
            log_results_artifact(results)

            # Detect if this is MCMC with credible intervals
            is_mcmc = config.model.estimator == "mcmc"
            has_mcmc_ci = (
                is_mcmc
                and results
                and (
                    results[0].insample_ci is not None
                    or results[0].oosample_ci is not None
                )
            )

            # Save plots to MLFlow if requested
            if save_plots:
                logger.info("Saving plots to MLFlow")
                log_plots_artifact(results, has_mcmc_ci=has_mcmc_ci)

                # Log MCMC diagnostic plots
                if is_mcmc:
                    logger.info("Saving MCMC diagnostic plots to MLFlow")
                    log_mcmc_diagnostics(results)

            _print_summary(results)
            logger.info(f"MLFlow run ID: {active_run.info.run_id}")

        # Also save plots locally if requested
        if plot_dir:
            _save_plots_local(results, plot_dir, is_mcmc=is_mcmc)


def _save_plots_local(results: list, plot_dir: Path, is_mcmc: bool = False) -> None:
    """Save prediction plots to local directory."""
    import matplotlib.pyplot as plt

    from simulation.plots import (
        plot_all_windows,
        plot_combined_timeseries,
        plot_combined_timeseries_with_ci,
        plot_mcmc_diagnostics,
        plot_window_predictions_with_ci,
    )

    logger.info(f"Saving plots to {plot_dir}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have credible intervals
    has_ci = results and (
        results[0].insample_ci is not None or results[0].oosample_ci is not None
    )

    # Save individual window plots
    windows_dir = plot_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    if has_ci:
        for result in results:
            fig = plot_window_predictions_with_ci(result)
            filepath = windows_dir / f"window_{result.window_id:03d}.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        plot_all_windows(results, windows_dir)
    logger.info(f"Saved {len(results)} window plots")

    # Save combined plot
    if has_ci:
        fig = plot_combined_timeseries_with_ci(results)
    else:
        fig = plot_combined_timeseries(results)
    combined_path = plot_dir / "combined_predictions.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined plot to {combined_path}")

    # Save MCMC diagnostic plots
    if is_mcmc and results:
        import arviz as az

        mcmc_dir = plot_dir / "mcmc"
        for result in results:
            if result.result is None or "mcmc" not in result.result.diagnostics:
                continue
            try:
                idata = az.from_numpyro(result.result.diagnostics["mcmc"])
                plot_mcmc_diagnostics(
                    idata, mcmc_dir / f"window_{result.window_id:03d}"
                )
                logger.info(f"Saved MCMC diagnostics for window {result.window_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to save MCMC diagnostics for window {result.window_id}: {e}"
                )


def _print_summary(results: list) -> None:
    """Print summary statistics."""
    import numpy as np

    logger.info("Simulation completed")

    typer.echo("\n--- Summary ---")
    typer.echo(f"Windows completed: {len(results)}")

    if results:
        is_rmse = [r.metrics["insample_rmse"] for r in results]
        is_mape = [r.metrics["insample_mape"] for r in results]
        oos_rmse = [r.metrics["oosample_rmse"] for r in results]
        oos_mape = [r.metrics["oosample_mape"] for r in results]

        # In-sample metrics
        typer.echo(f"IS RMSE: mean={np.mean(is_rmse):.4f}, std={np.std(is_rmse):.4f}")
        if not any(np.isnan(is_mape)):
            typer.echo(
                f"IS MAPE: mean={np.mean(is_mape):.2f}%, std={np.std(is_mape):.2f}%"
            )

        # Out-of-sample metrics (only if available)
        if not all(np.isnan(oos_rmse)):
            typer.echo(
                f"OOS RMSE: mean={np.nanmean(oos_rmse):.4f}, std={np.nanstd(oos_rmse):.4f}"
            )
            if not all(np.isnan(oos_mape)):
                typer.echo(
                    f"OOS MAPE: mean={np.nanmean(oos_mape):.2f}%, std={np.nanstd(oos_mape):.2f}%"
                )
            logger.info(
                f"OOS RMSE: mean={np.nanmean(oos_rmse):.4f}, std={np.nanstd(oos_rmse):.4f}"
            )
        else:
            typer.echo("OOS metrics: N/A (in-sample only)")
            logger.info(
                f"IS RMSE: mean={np.mean(is_rmse):.4f}, std={np.std(is_rmse):.4f}"
            )


@app.command()
def validate(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML configuration file")
    ],
) -> None:
    """Validate configuration file."""
    setup_logging(verbose=False)

    try:
        logger.info(f"Validating configuration: {config_path}")
        config = load_config(config_path)
        logger.info("Configuration is valid")
        typer.echo("Configuration is valid!")
        typer.echo(config.model_dump_json(indent=2))
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
