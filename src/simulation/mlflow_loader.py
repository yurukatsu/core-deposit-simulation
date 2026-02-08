"""MLFlow result loading utilities.

Load results and artifacts from MLFlow for analysis and recalculation.
"""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import arviz as az
import mlflow
import numpy as np
from loguru import logger
from numpy.typing import NDArray


@dataclass
class LoadedWindowResult:
    """Loaded window result from MLFlow.

    Similar to WindowResult but reconstructed from MLFlow artifacts.
    """

    window_id: int

    # Index-based bounds
    train_start: int
    train_end: int
    test_end: int

    # Date-based bounds (if available)
    train_start_date: date | None = None
    train_end_date: date | None = None
    test_end_date: date | None = None

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Parameters (from params.json)
    params: dict[str, Any] = field(default_factory=dict)

    # MCMC InferenceData (loaded from NetCDF)
    inference_data: az.InferenceData | None = None


@dataclass
class LoadedRun:
    """Loaded MLFlow run with all artifacts."""

    run_id: str
    experiment_id: str
    run_name: str | None
    status: str

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Run tags and params
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)

    # Metrics (final values)
    metrics: dict[str, float] = field(default_factory=dict)

    # Window results
    windows: list[LoadedWindowResult] = field(default_factory=list)

    def get_inference_data(self, window_id: int) -> az.InferenceData | None:
        """Get InferenceData for a specific window.

        Args:
            window_id: Window ID

        Returns:
            InferenceData or None if not available
        """
        for w in self.windows:
            if w.window_id == window_id:
                return w.inference_data
        return None

    def get_all_inference_data(self) -> dict[int, az.InferenceData]:
        """Get all InferenceData as a dict keyed by window_id."""
        return {
            w.window_id: w.inference_data
            for w in self.windows
            if w.inference_data is not None
        }

    def get_posterior_samples(
        self, param_name: str, window_id: int | None = None
    ) -> NDArray | dict[int, NDArray]:
        """Get posterior samples for a parameter.

        Args:
            param_name: Parameter name (e.g., 'lambda', 'm')
            window_id: Specific window ID, or None for all windows

        Returns:
            Array of samples if window_id specified, else dict of window_id -> samples
        """
        if window_id is not None:
            idata = self.get_inference_data(window_id)
            if idata is None:
                raise ValueError(f"No InferenceData for window {window_id}")
            return idata.posterior[param_name].values.flatten()
        else:
            result = {}
            for w in self.windows:
                if w.inference_data is not None:
                    result[w.window_id] = w.inference_data.posterior[
                        param_name
                    ].values.flatten()
            return result


def set_tracking_uri(tracking_uri: str) -> None:
    """Set MLFlow tracking URI.

    Args:
        tracking_uri: MLFlow tracking server URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    logger.debug(f"MLFlow tracking URI set to: {tracking_uri}")


def list_experiments(tracking_uri: str | None = None) -> list[dict[str, Any]]:
    """List all experiments.

    Args:
        tracking_uri: Optional tracking URI (uses current if not specified)

    Returns:
        List of experiment info dicts
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "tags": dict(exp.tags) if exp.tags else {},
        }
        for exp in experiments
    ]


def list_runs(
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    tracking_uri: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """List runs in an experiment.

    Args:
        experiment_name: Experiment name (mutually exclusive with experiment_id)
        experiment_id: Experiment ID
        tracking_uri: Optional tracking URI
        max_results: Maximum number of runs to return

    Returns:
        List of run info dicts
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        experiment_id = experiment.experiment_id
    elif experiment_id is None:
        raise ValueError("Either experiment_name or experiment_id must be specified")

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "tags": dict(run.data.tags),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
        for run in runs
    ]


def load_run(
    run_id: str,
    tracking_uri: str | None = None,
    load_inference_data: bool = True,
    artifact_download_path: str | Path | None = None,
) -> LoadedRun:
    """Load a complete run from MLFlow.

    Args:
        run_id: MLFlow run ID
        tracking_uri: Optional tracking URI
        load_inference_data: Whether to load MCMC InferenceData artifacts
        artifact_download_path: Optional path to download artifacts to
            (uses temp dir if not specified)

    Returns:
        LoadedRun with all data and artifacts
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    logger.info(f"Loading run: {run_id}")

    # Create LoadedRun
    loaded = LoadedRun(
        run_id=run_id,
        experiment_id=run.info.experiment_id,
        run_name=run.info.run_name,
        status=run.info.status,
        tags=dict(run.data.tags),
        params=dict(run.data.params),
        metrics=dict(run.data.metrics),
    )

    # Download artifacts
    if artifact_download_path:
        download_path = Path(artifact_download_path)
        download_path.mkdir(parents=True, exist_ok=True)
    else:
        download_path = None

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) if download_path is None else download_path

        # Download all artifacts
        logger.debug("Downloading artifacts...")
        try:
            local_path = client.download_artifacts(run_id, "", str(artifact_path))
            artifact_path = Path(local_path)
        except Exception as e:
            logger.warning(f"Failed to download artifacts: {e}")
            return loaded

        # Load config
        config_json = artifact_path / "config" / "config.json"
        if config_json.exists():
            with open(config_json) as f:
                loaded.config = json.load(f)
            logger.debug("Loaded config.json")

        # Load results.json
        results_json = artifact_path / "results.json"
        if results_json.exists():
            with open(results_json) as f:
                results_data = json.load(f)

            # Parse results into LoadedWindowResult
            for entry in results_data:
                window = LoadedWindowResult(
                    window_id=entry["window_id"],
                    train_start=entry["train_start"],
                    train_end=entry["train_end"],
                    test_end=entry["test_end"],
                    train_start_date=_parse_date(entry.get("train_start_date")),
                    train_end_date=_parse_date(entry.get("train_end_date")),
                    test_end_date=_parse_date(entry.get("test_end_date")),
                    metrics=entry.get("metrics", {}),
                    params=entry.get("params", {}),
                )
                loaded.windows.append(window)

            logger.debug(f"Loaded {len(loaded.windows)} window results")

        # Load InferenceData for each window
        if load_inference_data:
            models_dir = artifact_path / "models"
            if models_dir.exists():
                for window in loaded.windows:
                    window_dir = models_dir / f"window_{window.window_id:03d}"
                    netcdf_path = window_dir / "inference_data.nc"

                    if netcdf_path.exists():
                        try:
                            window.inference_data = az.from_netcdf(str(netcdf_path))
                            logger.debug(
                                f"Loaded InferenceData for window {window.window_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load InferenceData for window "
                                f"{window.window_id}: {e}"
                            )

                n_loaded = sum(
                    1 for w in loaded.windows if w.inference_data is not None
                )
                logger.info(
                    f"Loaded InferenceData for {n_loaded}/{len(loaded.windows)} windows"
                )

    return loaded


def load_inference_data(
    run_id: str,
    window_id: int,
    tracking_uri: str | None = None,
) -> az.InferenceData:
    """Load InferenceData for a specific window.

    Args:
        run_id: MLFlow run ID
        window_id: Window ID
        tracking_uri: Optional tracking URI

    Returns:
        ArviZ InferenceData
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = f"models/window_{window_id:03d}/inference_data.nc"

        try:
            local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
            return az.from_netcdf(local_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load InferenceData for window {window_id}: {e}"
            ) from e


def _parse_date(date_str: str | None) -> date | None:
    """Parse ISO date string to date object."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


# Convenience functions for analysis


def compute_posterior_summary(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for posterior samples.

    Args:
        idata: ArviZ InferenceData
        var_names: Variable names to summarize (None = all)

    Returns:
        Dict of variable name -> statistics dict
    """
    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    result = {}
    for var in var_names:
        if var not in idata.posterior:
            continue

        samples = idata.posterior[var].values.flatten()
        result[var] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "q5": float(np.percentile(samples, 5)),
            "q25": float(np.percentile(samples, 25)),
            "q75": float(np.percentile(samples, 75)),
            "q95": float(np.percentile(samples, 95)),
            "hdi_3%": float(az.hdi(samples, hdi_prob=0.94)[0]),
            "hdi_97%": float(az.hdi(samples, hdi_prob=0.94)[1]),
        }

    return result


def predict_from_posterior(
    idata: az.InferenceData,
    inflow: NDArray,
    V0: float,
    z: NDArray | None = None,
    ci_prob: float = 0.95,
) -> dict[str, NDArray]:
    """Generate predictions from posterior samples.

    Args:
        idata: ArviZ InferenceData with posterior samples
        inflow: Inflow array
        V0: Initial volume
        z: Covariate array (optional)
        ci_prob: Credible interval probability

    Returns:
        Dict with 'mean', 'median', 'lower', 'upper' predictions
    """
    from coredeposit.model import V_model

    posterior = idata.posterior

    # Get samples
    lam = posterior["lambda"].values.flatten()
    gam = posterior["gamma"].values.flatten()
    w1 = posterior["w1"].values.flatten()
    h = posterior["h"].values.flatten()
    m = posterior["m"].values.flatten()

    n_samples = len(lam)
    n_periods = len(inflow)

    # Compute predictions for each sample
    predictions = np.zeros((n_samples, n_periods))

    for i in range(n_samples):
        # Compute weight from covariates if beta is present
        if "beta" in posterior and z is not None:
            beta = posterior["beta"].values.reshape(n_samples, -1)[i]
            weight = np.exp(z @ beta)
        else:
            weight = np.ones(n_periods)

        predictions[i] = V_model(
            lam=lam[i],
            gam=gam[i],
            w1=w1[i],
            h=h[i],
            m=m[i],
            inflow=inflow,
            V0=V0,
            weight=weight,
        )

    # Compute summary statistics
    alpha = 1 - ci_prob
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    return {
        "mean": np.mean(predictions, axis=0),
        "median": np.median(predictions, axis=0),
        "lower": np.percentile(predictions, lower_q, axis=0),
        "upper": np.percentile(predictions, upper_q, axis=0),
        "samples": predictions,
    }
