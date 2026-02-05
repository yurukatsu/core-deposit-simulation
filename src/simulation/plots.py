"""Plotting utilities for simulation results."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from simulation.simulator import WindowResult

if TYPE_CHECKING:
    import arviz as az


def plot_window_predictions(
    result: WindowResult,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot in-sample and out-of-sample predictions vs actual values for a single window.

    Args:
        result: WindowResult containing predictions and actuals
        title: Optional plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    has_oosample = result.oosample_actual is not None and len(result.oosample_actual) > 0

    if has_oosample:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1 = axes[0]
        ax2 = axes[1]
    else:
        # In-sample only: single plot
        fig, ax1 = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # In-sample plot
    if result.insample_dates is not None:
        x_insample = result.insample_dates
        ax1.set_xlabel("Date")
    else:
        x_insample = np.arange(result.train_start, result.train_end)
        ax1.set_xlabel("Index")

    ax1.plot(x_insample, result.insample_actual, "b-", label="Actual", linewidth=1.5)
    ax1.plot(x_insample, result.insample_pred, "r--", label="Predicted", linewidth=1.5)
    ax1.set_ylabel("Volume")
    ax1.set_title("In-Sample")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rotate x-axis labels if dates
    if result.insample_dates is not None:
        ax1.tick_params(axis="x", rotation=45)

    # Add in-sample metrics when no out-of-sample
    if not has_oosample:
        rmse = result.metrics.get("insample_rmse", np.nan)
        mape = result.metrics.get("insample_mape", np.nan)
        metrics_text = f"RMSE: {rmse:.2f}\nMAPE: {mape:.1f}%"
        ax1.text(
            0.02, 0.98, metrics_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Out-of-sample plot
    if has_oosample and ax2 is not None:
        if result.oosample_dates is not None:
            x_oosample = result.oosample_dates
            ax2.set_xlabel("Date")
        else:
            x_oosample = np.arange(result.train_end, result.test_end)
            ax2.set_xlabel("Index")

        ax2.plot(x_oosample, result.oosample_actual, "b-", label="Actual", linewidth=1.5)
        ax2.plot(x_oosample, result.oosample_pred, "r--", label="Predicted", linewidth=1.5)
        ax2.set_ylabel("Volume")
        ax2.set_title("Out-of-Sample")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Rotate x-axis labels if dates
        if result.oosample_dates is not None:
            ax2.tick_params(axis="x", rotation=45)

        # Add metrics to the out-of-sample plot
        rmse = result.metrics.get("oosample_rmse", np.nan)
        mape = result.metrics.get("oosample_mape", np.nan)
        metrics_text = f"RMSE: {rmse:.2f}\nMAPE: {mape:.1f}%"
        ax2.text(
            0.02, 0.98, metrics_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Overall title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        window_title = f"Window {result.window_id}"
        if result.train_start_date:
            end_date = result.test_end_date if has_oosample else result.train_end_date
            if end_date:
                window_title += f" ({result.train_start_date} to {end_date})"
        fig.suptitle(window_title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_all_windows(
    results: list[WindowResult],
    output_dir: Path | str,
    prefix: str = "window",
) -> list[Path]:
    """Plot predictions for all windows and save to files.

    Args:
        results: List of WindowResult
        output_dir: Directory to save plots
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for result in results:
        fig = plot_window_predictions(result)
        filename = f"{prefix}_{result.window_id:03d}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(filepath)

    return saved_paths


def plot_combined_timeseries(
    results: list[WindowResult],
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot combined time series showing all out-of-sample predictions.

    This creates a single plot showing actual values and a continuous
    prediction line connecting all windows.

    Args:
        results: List of WindowResult
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not results:
        return fig

    # Check if we have out-of-sample data
    has_oosample = any(r.oosample_actual is not None and len(r.oosample_actual) > 0 for r in results)

    if not has_oosample:
        # In-sample only: plot combined in-sample predictions
        ax.set_title("In-Sample Predictions (No Out-of-Sample)")
        ax.text(0.5, 0.5, "No out-of-sample data (t_test=0)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    # Collect all actual values (deduplicated)
    all_actuals: dict = {}  # key -> (date, actual)
    all_preds: dict = {}  # key -> (date, pred, window_id)

    for result in results:
        if result.oosample_actual is None or len(result.oosample_actual) == 0:
            continue

        if result.oosample_dates is not None:
            dates = result.oosample_dates
        else:
            dates = np.arange(result.train_end, result.test_end)

        for d, a, p in zip(dates, result.oosample_actual, result.oosample_pred, strict=False):
            key = str(d)
            # Store actual (only need once)
            if key not in all_actuals:
                all_actuals[key] = (d, a)
            # Store prediction (prefer later windows for overlapping dates)
            if key not in all_preds or result.window_id > all_preds[key][2]:
                all_preds[key] = (d, p, result.window_id)

    # Sort by date/index
    sorted_actuals = sorted(all_actuals.values(), key=lambda x: x[0])
    sorted_preds = sorted(all_preds.values(), key=lambda x: x[0])

    if sorted_actuals:
        actual_dates, actual_values = zip(*[(d, v) for d, v in sorted_actuals], strict=False)
        ax.plot(actual_dates, actual_values, "b-", label="Actual", linewidth=1.5, zorder=10)

    if sorted_preds:
        pred_dates, pred_values, _ = zip(*sorted_preds, strict=False)
        ax.plot(pred_dates, pred_values, "r-", label="Predicted", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Date" if results[0].oosample_dates is not None else "Index")
    ax.set_ylabel("Volume")
    ax.set_title("Out-of-Sample Predictions (Connected)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if results[0].oosample_dates is not None:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MCMC Diagnostic Plots (ArviZ)
# ---------------------------------------------------------------------------


def plot_mcmc_trace(idata: "az.InferenceData", figsize: tuple[float, float] = (12, 10)) -> Figure:
    """Plot MCMC trace plots for convergence check.

    Args:
        idata: ArviZ InferenceData object
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import arviz as az

    var_names = ["lambda", "gamma", "w1", "h", "m"]
    fig, axes = plt.subplots(len(var_names), 2, figsize=figsize)
    az.plot_trace(idata, var_names=var_names, axes=axes)
    plt.tight_layout()
    return fig


def plot_mcmc_posterior(idata: "az.InferenceData", figsize: tuple[float, float] = (12, 6)) -> Figure:
    """Plot posterior distributions.

    Args:
        idata: ArviZ InferenceData object
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import arviz as az

    var_names = ["lambda", "gamma", "w1", "h", "m", "sigma"]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    az.plot_posterior(idata, var_names=var_names, hdi_prob=0.95, ax=axes.flatten())
    plt.tight_layout()
    return fig


def plot_mcmc_pair(idata: "az.InferenceData", figsize: tuple[float, float] = (10, 10)) -> Figure:
    """Plot pair plot showing parameter correlations.

    Args:
        idata: ArviZ InferenceData object
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import arviz as az

    var_names = ["lambda", "gamma", "w1", "h"]
    fig = plt.figure(figsize=figsize)
    az.plot_pair(idata, var_names=var_names, kind="kde", marginals=True)
    plt.tight_layout()
    return plt.gcf()


def plot_mcmc_forest(idata: "az.InferenceData", figsize: tuple[float, float] = (8, 4)) -> Figure:
    """Plot forest plot comparing parameters.

    Args:
        idata: ArviZ InferenceData object
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    import arviz as az

    fig, ax = plt.subplots(figsize=figsize)
    az.plot_forest(idata, var_names=["w1", "h"], combined=True, hdi_prob=0.95, ax=ax)
    plt.tight_layout()
    return fig


def plot_mcmc_diagnostics(
    idata: "az.InferenceData",
    output_dir: Path | str,
) -> list[Path]:
    """Generate all MCMC diagnostic plots.

    Args:
        idata: ArviZ InferenceData object
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # Trace plot
    fig = plot_mcmc_trace(idata)
    path = output_dir / "mcmc_trace.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    # Posterior plot
    fig = plot_mcmc_posterior(idata)
    path = output_dir / "mcmc_posterior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    # Pair plot
    fig = plot_mcmc_pair(idata)
    path = output_dir / "mcmc_pair.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    # Forest plot
    fig = plot_mcmc_forest(idata)
    path = output_dir / "mcmc_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


# ---------------------------------------------------------------------------
# Prediction Plots with Credible Intervals
# ---------------------------------------------------------------------------


def plot_window_predictions_with_ci(
    result: WindowResult,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot predictions with credible intervals for MCMC results.

    Args:
        result: WindowResult containing predictions with credible intervals
        title: Optional plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    has_oosample = result.oosample_actual is not None and len(result.oosample_actual) > 0

    if has_oosample:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1 = axes[0]
        ax2 = axes[1]
    else:
        # In-sample only: single plot
        fig, ax1 = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # In-sample plot
    if result.insample_dates is not None:
        x_insample = result.insample_dates
        ax1.set_xlabel("Date")
    else:
        x_insample = np.arange(result.train_start, result.train_end)
        ax1.set_xlabel("Index")

    # Plot credible intervals if available
    if result.insample_ci is not None:
        ci = result.insample_ci
        ax1.fill_between(
            x_insample, ci["lower"], ci["upper"],
            alpha=0.3, color="red", label="95% CI"
        )

    ax1.plot(x_insample, result.insample_actual, "b-", label="Actual", linewidth=1.5)
    ax1.plot(x_insample, result.insample_pred, "r--", label="Predicted", linewidth=1.5)
    ax1.set_ylabel("Volume")
    ax1.set_title("In-Sample")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if result.insample_dates is not None:
        ax1.tick_params(axis="x", rotation=45)

    # Add in-sample metrics when no out-of-sample
    if not has_oosample:
        rmse = result.metrics.get("insample_rmse", np.nan)
        mape = result.metrics.get("insample_mape", np.nan)
        metrics_text = f"RMSE: {rmse:.2f}\nMAPE: {mape:.1f}%"
        ax1.text(
            0.02, 0.98, metrics_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Out-of-sample plot
    if has_oosample and ax2 is not None:
        if result.oosample_dates is not None:
            x_oosample = result.oosample_dates
            ax2.set_xlabel("Date")
        else:
            x_oosample = np.arange(result.train_end, result.test_end)
            ax2.set_xlabel("Index")

        # Plot credible intervals if available
        if result.oosample_ci is not None:
            ci = result.oosample_ci
            ax2.fill_between(
                x_oosample, ci["lower"], ci["upper"],
                alpha=0.3, color="red", label="95% CI"
            )

        ax2.plot(x_oosample, result.oosample_actual, "b-", label="Actual", linewidth=1.5)
        ax2.plot(x_oosample, result.oosample_pred, "r--", label="Predicted", linewidth=1.5)
        ax2.set_ylabel("Volume")
        ax2.set_title("Out-of-Sample")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if result.oosample_dates is not None:
            ax2.tick_params(axis="x", rotation=45)

        # Add metrics
        rmse = result.metrics.get("oosample_rmse", np.nan)
        mape = result.metrics.get("oosample_mape", np.nan)
        metrics_text = f"RMSE: {rmse:.2f}\nMAPE: {mape:.1f}%"
        ax2.text(
            0.02, 0.98, metrics_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Overall title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        window_title = f"Window {result.window_id}"
        if result.train_start_date:
            end_date = result.test_end_date if has_oosample else result.train_end_date
            if end_date:
                window_title += f" ({result.train_start_date} to {end_date})"
        fig.suptitle(window_title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_combined_timeseries_with_ci(
    results: list[WindowResult],
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot combined time series with credible intervals for MCMC.

    Args:
        results: List of WindowResult with credible intervals
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not results:
        return fig

    # Check if we have out-of-sample data
    has_oosample = any(r.oosample_actual is not None and len(r.oosample_actual) > 0 for r in results)

    if not has_oosample:
        # In-sample only: plot combined in-sample predictions
        ax.set_title("In-Sample Predictions (No Out-of-Sample)")
        ax.text(0.5, 0.5, "No out-of-sample data (t_test=0)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    # Collect all data (deduplicated, prefer later windows)
    all_actuals: dict = {}
    all_preds: dict = {}
    all_ci_lower: dict = {}
    all_ci_upper: dict = {}

    has_ci = any(r.oosample_ci is not None for r in results)

    for result in results:
        if result.oosample_actual is None or len(result.oosample_actual) == 0:
            continue

        if result.oosample_dates is not None:
            dates = result.oosample_dates
        else:
            dates = np.arange(result.train_end, result.test_end)

        for i, d in enumerate(dates):
            key = str(d)
            if key not in all_actuals:
                all_actuals[key] = (d, result.oosample_actual[i])

            if key not in all_preds or result.window_id > all_preds[key][2]:
                all_preds[key] = (d, result.oosample_pred[i], result.window_id)

                if has_ci and result.oosample_ci is not None:
                    ci = result.oosample_ci
                    all_ci_lower[key] = (d, ci["lower"][i])
                    all_ci_upper[key] = (d, ci["upper"][i])

    # Sort by date
    sorted_actuals = sorted(all_actuals.values(), key=lambda x: x[0])
    sorted_preds = sorted(all_preds.values(), key=lambda x: x[0])

    # Plot credible intervals
    if has_ci and all_ci_lower:
        sorted_ci_lower = sorted(all_ci_lower.values(), key=lambda x: x[0])
        sorted_ci_upper = sorted(all_ci_upper.values(), key=lambda x: x[0])

        ci_dates = [d for d, _ in sorted_ci_lower]
        ci_lo = [v for _, v in sorted_ci_lower]
        ci_hi = [v for _, v in sorted_ci_upper]

        ax.fill_between(ci_dates, ci_lo, ci_hi, alpha=0.3, color="red", label="95% CI")

    if sorted_actuals:
        actual_dates, actual_values = zip(*[(d, v) for d, v in sorted_actuals], strict=False)
        ax.plot(actual_dates, actual_values, "b-", label="Actual", linewidth=1.5, zorder=10)

    if sorted_preds:
        pred_dates, pred_values, _ = zip(*sorted_preds, strict=False)
        ax.plot(pred_dates, pred_values, "r-", label="Predicted", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Date" if results[0].oosample_dates is not None else "Index")
    ax.set_ylabel("Volume")
    ax.set_title("Out-of-Sample Predictions with Credible Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if results[0].oosample_dates is not None:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig
