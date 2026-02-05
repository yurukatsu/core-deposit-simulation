"""Rolling Window Simulator for core deposit models."""

from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

import numpy as np
import pandas as pd
from coredeposit import CoreDepositData, EstimationResult, MCMCEstimator, NLSEstimator
from coredeposit.estimators import default_map_priors
from loguru import logger
from numpy.typing import NDArray

from simulation.config import Config, ModelConfig


@dataclass
class WindowResult:
    """Result for a single rolling window."""

    window_id: int

    # Index-based bounds
    train_start: int
    train_end: int  # Also test_start
    test_end: int

    # Date-based bounds (if available)
    train_start_date: date | None = None
    train_end_date: date | None = None  # Also test_start_date
    test_end_date: date | None = None

    # Estimation result
    result: EstimationResult | None = None

    # Predictions
    insample_pred: NDArray[np.floating] | None = None
    insample_actual: NDArray[np.floating] | None = None
    oosample_pred: NDArray[np.floating] | None = None
    oosample_actual: NDArray[np.floating] | None = None

    # Date arrays for predictions (if available)
    insample_dates: NDArray | None = None
    oosample_dates: NDArray | None = None

    # Credible intervals for MCMC (dict with lower, upper for 95% CI)
    insample_ci: dict[str, NDArray] | None = None
    oosample_ci: dict[str, NDArray] | None = None

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def test_start(self) -> int:
        """Alias for train_end (test period starts where training ends)."""
        return self.train_end

    @property
    def test_start_date(self) -> date | None:
        """Alias for train_end_date."""
        return self.train_end_date


def _compute_metrics(actual: NDArray, pred: NDArray) -> dict[str, float]:
    """Compute prediction metrics."""
    residuals = actual - pred
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / actual)) * 100) if np.all(actual != 0) else np.nan
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def _check_mcmc_convergence(result: EstimationResult, threshold: float = 1.1) -> tuple[bool, float]:
    """Check MCMC convergence using R-hat statistic.

    Args:
        result: EstimationResult with MCMC diagnostics
        threshold: R-hat threshold (typically 1.1)

    Returns:
        Tuple of (converged, max_rhat)
    """
    import arviz as az

    if "mcmc" not in result.diagnostics:
        return True, 1.0

    try:
        idata = az.from_numpyro(result.diagnostics["mcmc"])
        rhat = az.rhat(idata)

        # Check main parameters
        max_rhat = 1.0
        for var in ["lambda", "gamma", "w1", "h", "m"]:
            if var in rhat:
                val = float(rhat[var].values)
                if val > max_rhat:
                    max_rhat = val

        return max_rhat <= threshold, max_rhat
    except Exception as e:
        logger.warning(f"Failed to compute R-hat: {e}")
        return True, 1.0


class RollingWindowSimulator:
    """Rolling window simulation for core deposit models.

    Performs walk-forward validation:
    1. Train model on [train_start : train_end]
    2. In-sample prediction on training period
    3. Out-of-sample prediction on [train_end : test_end]
    4. Slide window by t_gap and repeat
    """

    def __init__(self, config: Config):
        """Initialize simulator with configuration.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self._data: pd.DataFrame | None = None
        self._V_obs: NDArray | None = None
        self._inflow: NDArray | None = None
        self._z: NDArray | None = None  # Covariates
        self._dates: pd.DatetimeIndex | None = None  # Date index

    def load_data(self) -> None:
        """Load data from configured path."""
        data_config = self.config.data
        logger.debug(f"Loading data from {data_config.path}")
        self._data = pd.read_csv(data_config.path)

        self._V_obs = self._data[data_config.volume_column].values
        self._inflow = self._data[data_config.inflow_column].values
        logger.debug(f"Loaded {len(self._V_obs)} observations")

        # Load dates if date_column is specified
        if data_config.date_column is not None:
            self._dates = pd.to_datetime(self._data[data_config.date_column])
            logger.debug(f"Date range: {self._dates.iloc[0]} to {self._dates.iloc[-1]}")

        # Load covariates if configured
        if data_config.covariate is not None:
            cov_config = data_config.covariate
            logger.debug(f"Loading covariates from {cov_config.path}, columns={cov_config.columns}")
            cov_data = pd.read_csv(cov_config.path)

            # Join on date column
            if data_config.date_column is None:
                raise ValueError("date_column must be specified in data config when using covariates")

            # Convert dates and create year-month period for matching
            main_dates = pd.to_datetime(self._data[data_config.date_column])
            main_periods = main_dates.dt.to_period("M")

            cov_dates = pd.to_datetime(cov_data[cov_config.date_column])
            cov_periods = cov_dates.dt.to_period("M")

            # Create a mapping from period to covariate values
            cov_data_indexed = cov_data.set_index(cov_periods)
            cov_values = cov_data_indexed[cov_config.columns]

            # Align covariates with main data periods (month-level matching)
            aligned_cov = cov_values.reindex(main_periods.values)

            # Warn about missing values but allow them (user should use start_index to skip)
            if aligned_cov.isna().any().any():
                missing_mask = aligned_cov.isna().any(axis=1).values
                n_missing = missing_mask.sum()
                import warnings
                warnings.warn(
                    f"Missing covariate data for {n_missing} periods. "
                    f"Use start_index to skip periods without covariate data.",
                    UserWarning,
                    stacklevel=2,
                )

            self._z = aligned_cov.values

    @property
    def data_length(self) -> int:
        """Total length of loaded data."""
        if self._V_obs is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return len(self._V_obs)

    def _get_date(self, index: int) -> date | None:
        """Get date for a given index, or None if dates not available."""
        if self._dates is None:
            return None
        return self._dates.iloc[index].date()

    def _get_dates_range(self, start: int, end: int) -> NDArray | None:
        """Get date array for a range, or None if dates not available."""
        if self._dates is None:
            return None
        return self._dates.iloc[start:end].values

    def _create_estimator(
        self, model_config: ModelConfig, init_params: dict | None = None
    ) -> NLSEstimator | MCMCEstimator:
        """Create estimator based on configuration."""
        if model_config.estimator == "nls":
            return NLSEstimator(
                loss=model_config.loss,
                f_scale=model_config.f_scale,
            )
        elif model_config.estimator == "map":
            return NLSEstimator(
                loss=model_config.loss,
                f_scale=model_config.f_scale,
                priors=default_map_priors(),
            )
        else:  # mcmc
            return MCMCEstimator(
                num_warmup=model_config.num_warmup,
                num_samples=model_config.num_samples,
                num_chains=model_config.num_chains,
                ar_errors=model_config.ar_errors,
                likelihood=model_config.likelihood,
                init_params=init_params,
            )

    def _prepare_data(self, start: int, end: int) -> CoreDepositData:
        """Prepare CoreDepositData for a window."""
        if self._V_obs is None or self._inflow is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        z = None
        if self._z is not None:
            z = self._z[start:end]
            # Check for NaN in covariates for this window
            if np.isnan(z).any():
                raise ValueError(
                    f"Covariate data contains NaN for window [{start}:{end}]. "
                    f"Adjust start_index to skip periods without covariate data."
                )

        return CoreDepositData(
            V_obs=self._V_obs[start:end],
            inflow=self._inflow[start:end],
            V0=self._V_obs[start],
            z=z,
        )

    def _predict(
        self, result: EstimationResult, data: CoreDepositData, n_periods: int
    ) -> NDArray[np.floating]:
        """Generate predictions using estimated parameters.

        For NLS, uses point estimates. For MCMC, uses posterior mean.
        """
        from coredeposit.model import V_model

        params = result.params

        # Get parameter values (handle both NLS and MCMC)
        def get_param(name: str) -> float | NDArray:
            val = params[name]
            if hasattr(val, "mean"):
                return float(val.mean())
            return val

        lam = get_param("lambda")
        gam = get_param("gamma")
        w1 = get_param("w1")
        h = get_param("h")
        m = get_param("m")

        # Compute weight from covariates if beta is present
        if "beta" in params and data.z is not None:
            beta = params["beta"]
            # For MCMC, beta has shape (n_samples, n_features), take mean over samples
            # For NLS, beta is already (n_features,)
            if hasattr(beta, "ndim") and beta.ndim > 1:
                beta = beta.mean(axis=0)
            z = np.asarray(data.z[:n_periods])
            weight = np.exp(z @ beta)
        else:
            weight = np.ones(n_periods)

        # Compute model prediction
        V_pred = V_model(
            lam=lam,
            gam=gam,
            w1=w1,
            h=h,
            m=m,
            inflow=data.inflow[:n_periods],
            V0=data.V0,
            weight=weight,
        )
        return np.asarray(V_pred)

    def _compute_mcmc_ci(
        self,
        estimator: MCMCEstimator,
        result: EstimationResult,
        train_data: CoreDepositData,
        full_data: CoreDepositData,
        split_idx: int,
    ) -> tuple[dict[str, NDArray], dict[str, NDArray], NDArray, NDArray]:
        """Compute credible intervals and mean predictions for MCMC.

        Args:
            estimator: MCMCEstimator instance
            result: Estimation result with MCMC samples
            train_data: Training data
            full_data: Full data (train + test)
            split_idx: Index to split in-sample and out-of-sample

        Returns:
            Tuple of (insample_ci, oosample_ci, insample_mean, oosample_mean)
        """
        # In-sample CI (95% only)
        pred_95 = estimator.predict(train_data, result, uncertainty=True, ci_prob=0.95)
        insample_ci = {
            "lower": pred_95["lower"],
            "upper": pred_95["upper"],
        }
        # Use mean from predictive distribution (MSE-optimal point estimate)
        insample_mean = pred_95["mean"]

        # Out-of-sample CI (predict full, then slice)
        full_pred_95 = estimator.predict(full_data, result, uncertainty=True, ci_prob=0.95)
        oosample_ci = {
            "lower": full_pred_95["lower"][split_idx:],
            "upper": full_pred_95["upper"][split_idx:],
        }
        oosample_mean = full_pred_95["mean"][split_idx:]

        return insample_ci, oosample_ci, insample_mean, oosample_mean

    def _fit_mcmc_with_retry(
        self,
        train_data: CoreDepositData,
        init_params: dict | None,
    ) -> tuple[EstimationResult, MCMCEstimator]:
        """Fit MCMC with automatic retry on convergence failure.

        Args:
            train_data: Training data
            init_params: Initial parameter values

        Returns:
            Tuple of (result, estimator)
        """
        model_config = self.config.model
        num_warmup = model_config.num_warmup
        num_samples = model_config.num_samples

        for attempt in range(model_config.max_retries + 1):
            # Create estimator with current settings
            estimator = MCMCEstimator(
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=model_config.num_chains,
                ar_errors=model_config.ar_errors,
                likelihood=model_config.likelihood,
                init_params=init_params,
            )

            logger.trace(f"MCMC attempt {attempt + 1}: warmup={num_warmup}, samples={num_samples}")
            result = estimator.fit(train_data)

            # Check convergence
            converged, max_rhat = _check_mcmc_convergence(result, model_config.rhat_threshold)

            if converged:
                if attempt > 0:
                    logger.info(f"MCMC converged after {attempt + 1} attempts (R-hat={max_rhat:.3f})")
                return result, estimator

            logger.warning(
                f"MCMC did not converge (R-hat={max_rhat:.3f} > {model_config.rhat_threshold})"
            )

            if attempt < model_config.max_retries:
                # Increase warmup and samples for retry
                num_warmup = int(num_warmup * model_config.retry_multiplier)
                num_samples = int(num_samples * model_config.retry_multiplier)
                logger.info(f"Retrying with warmup={num_warmup}, samples={num_samples}")

        logger.warning(
            f"MCMC failed to converge after {model_config.max_retries + 1} attempts. "
            f"Using last result (R-hat={max_rhat:.3f})"
        )
        return result, estimator

    def _resolve_bounds(self) -> tuple[int, int]:
        """Resolve start/end bounds from config (index or date-based).

        Returns:
            (start_index, end_index)
        """
        sim = self.config.simulation
        total_len = self.data_length

        # Date-based bounds
        if sim.start_date is not None or sim.end_date is not None:
            if self._dates is None:
                raise ValueError(
                    "date_column must be specified in data config when using start_date/end_date"
                )

            if sim.start_date is not None:
                # Find first index >= start_date
                mask = self._dates >= pd.Timestamp(sim.start_date)
                if not mask.any():
                    raise ValueError(f"start_date {sim.start_date} is after all data")
                start = int(mask.argmax())
            else:
                start = 0

            if sim.end_date is not None:
                # Find first index >= end_date (exclusive)
                mask = self._dates >= pd.Timestamp(sim.end_date)
                if mask.any():
                    end = int(mask.argmax())
                else:
                    end = total_len
            else:
                end = total_len
        else:
            # Index-based bounds
            start = sim.start_index if sim.start_index is not None else 0
            end = sim.end_index if sim.end_index is not None else total_len

        return start, end

    def iter_windows(self) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over window indices.

        Yields:
            (window_id, train_start, train_end, test_end)
        """
        sim = self.config.simulation

        start, end = self._resolve_bounds()

        # Required length for one window: t_train + t_test
        required = sim.t_train + sim.t_test

        window_id = 0
        train_start = start

        while train_start + required <= end:
            # Check max_windows limit
            if sim.max_windows is not None and window_id >= sim.max_windows:
                break

            train_end = train_start + sim.t_train
            test_end = train_end + sim.t_test

            yield (window_id, train_start, train_end, test_end)

            window_id += 1
            train_start += sim.t_gap

    def run_window(self, train_start: int, train_end: int, test_end: int) -> WindowResult:
        """Run simulation for a single window.

        Args:
            train_start: Start index of training period
            train_end: End index of training period (exclusive)
            test_end: End index of test period (exclusive)

        Returns:
            WindowResult with estimation and predictions
        """
        model_config = self.config.model
        logger.trace(f"Running window: train=[{train_start}:{train_end}], test=[{train_end}:{test_end}]")

        # Prepare training data
        train_data = self._prepare_data(train_start, train_end)

        # Initialize from NLS if configured for MCMC
        init_params = None
        if model_config.estimator == "mcmc" and model_config.init_from_nls:
            nls = NLSEstimator()
            nls_result = nls.fit(train_data, m_fixed=model_config.fix_m)
            init_params = {
                k: v for k, v in nls_result.params.items() if k in ["lambda", "gamma", "w1", "h", "m"]
            }

        # Fit model
        if model_config.estimator in ("nls", "map"):
            estimator = self._create_estimator(model_config, init_params)
            logger.trace(f"Fitting {model_config.estimator} estimator")
            result = estimator.fit(train_data, m_fixed=model_config.fix_m)
        else:  # mcmc
            result, estimator = self._fit_mcmc_with_retry(train_data, init_params)

        # Log estimated parameters (use mean for MCMC arrays)
        lam_val = result.params.get("lambda")
        m_val = result.params.get("m")
        lam_str = f"{float(lam_val.mean()):.4f}" if hasattr(lam_val, "mean") else f"{lam_val:.4f}"
        m_str = f"{float(m_val.mean()):.4f}" if hasattr(m_val, "mean") else f"{m_val:.4f}"
        logger.trace(f"Estimation complete: lambda={lam_str}, m={m_str}")

        # Prepare data for predictions
        insample_actual = train_data.V_obs
        has_oosample = test_end > train_end

        # Compute predictions and credible intervals
        insample_ci = None
        oosample_ci = None
        oosample_pred = None
        oosample_actual = None

        if has_oosample:
            full_data = self._prepare_data(train_start, test_end)
            oosample_actual = self._V_obs[train_end:test_end]

            if model_config.estimator == "mcmc":
                # For MCMC, use mean from predictive distribution (consistent with CI)
                insample_ci, oosample_ci, insample_pred, oosample_pred = self._compute_mcmc_ci(
                    estimator, result, train_data, full_data, train_end - train_start
                )
            else:
                # For NLS/MAP, use point estimate
                insample_pred = self._predict(result, train_data, len(train_data.V_obs))
                full_pred = self._predict(result, full_data, test_end - train_start)
                oosample_pred = full_pred[train_end - train_start :]
        else:
            # In-sample only (t_test = 0)
            if model_config.estimator == "mcmc":
                pred_95 = estimator.predict(train_data, result, uncertainty=True, ci_prob=0.95)
                insample_ci = {
                    "lower": pred_95["lower"],
                    "upper": pred_95["upper"],
                }
                insample_pred = pred_95["mean"]
            else:
                insample_pred = self._predict(result, train_data, len(train_data.V_obs))

        # Compute metrics
        insample_metrics = _compute_metrics(insample_actual, insample_pred)

        metrics = {
            "insample_rmse": insample_metrics["rmse"],
            "insample_mape": insample_metrics["mape"],
            "oosample_rmse": np.nan,
            "oosample_mape": np.nan,
        }

        if has_oosample:
            oosample_metrics = _compute_metrics(oosample_actual, oosample_pred)
            metrics["oosample_rmse"] = oosample_metrics["rmse"]
            metrics["oosample_mape"] = oosample_metrics["mape"]

        return WindowResult(
            window_id=0,  # Will be set by caller
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
            train_start_date=self._get_date(train_start),
            train_end_date=self._get_date(train_end),
            test_end_date=self._get_date(test_end) if test_end < self.data_length else None,
            result=result,
            insample_pred=insample_pred,
            insample_actual=np.asarray(insample_actual),
            oosample_pred=oosample_pred,
            oosample_actual=np.asarray(oosample_actual) if oosample_actual is not None else None,
            insample_dates=self._get_dates_range(train_start, train_end),
            oosample_dates=self._get_dates_range(train_end, test_end) if has_oosample else None,
            insample_ci=insample_ci,
            oosample_ci=oosample_ci,
            metrics=metrics,
        )

    def run(self) -> list[WindowResult]:
        """Run full rolling window simulation.

        Returns:
            List of WindowResult for each window
        """
        if self._V_obs is None:
            self.load_data()

        n_jobs = self.config.simulation.n_jobs

        if n_jobs > 1:
            return self._run_parallel(n_jobs)
        else:
            return self._run_sequential()

    def _run_sequential(self) -> list[WindowResult]:
        """Run windows sequentially."""
        logger.info("Starting rolling window simulation (sequential)")
        results = []
        for window_id, train_start, train_end, test_end in self.iter_windows():
            result = self.run_window(train_start, train_end, test_end)
            result.window_id = window_id
            results.append(result)
            oos_rmse = result.metrics['oosample_rmse']
            if np.isnan(oos_rmse):
                logger.debug(f"Window {window_id} completed: IS RMSE={result.metrics['insample_rmse']:.4f} (in-sample only)")
            else:
                logger.debug(f"Window {window_id} completed: OOS RMSE={oos_rmse:.4f}")

        logger.info(f"Simulation complete: {len(results)} windows processed")
        return results

    def _run_parallel(self, n_jobs: int) -> list[WindowResult]:
        """Run windows in parallel using ProcessPoolExecutor.

        Args:
            n_jobs: Number of parallel workers

        Returns:
            List of WindowResult for each window
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        windows = list(self.iter_windows())
        n_windows = len(windows)

        logger.info(f"Starting rolling window simulation (parallel, n_jobs={n_jobs})")

        # Prepare arguments for parallel execution
        # We need to pass serializable data, not the simulator object
        config_dict = self.config.model_dump()
        V_obs = self._V_obs
        inflow = self._inflow
        z = self._z
        dates = self._dates.values if self._dates is not None else None

        results = [None] * n_windows

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_window = {}
            for window_id, train_start, train_end, test_end in windows:
                future = executor.submit(
                    _run_window_worker,
                    config_dict,
                    V_obs,
                    inflow,
                    z,
                    dates,
                    window_id,
                    train_start,
                    train_end,
                    test_end,
                )
                future_to_window[future] = window_id

            # Collect results as they complete
            for future in as_completed(future_to_window):
                window_id = future_to_window[future]
                try:
                    result = future.result()
                    results[window_id] = result
                    oos_rmse = result.metrics['oosample_rmse']
                    if np.isnan(oos_rmse):
                        logger.debug(f"Window {window_id} completed: IS RMSE={result.metrics['insample_rmse']:.4f} (in-sample only)")
                    else:
                        logger.debug(f"Window {window_id} completed: OOS RMSE={oos_rmse:.4f}")
                except Exception as e:
                    logger.error(f"Window {window_id} failed: {e}")
                    raise

        logger.info(f"Simulation complete: {len(results)} windows processed")
        return results


def _run_window_worker(
    config_dict: dict,
    V_obs: NDArray,
    inflow: NDArray,
    z: NDArray | None,
    dates: NDArray | None,
    window_id: int,
    train_start: int,
    train_end: int,
    test_end: int,
) -> WindowResult:
    """Worker function for parallel window execution.

    This function is called in a subprocess and must be picklable.
    """
    from simulation.config import Config

    # Reconstruct config and simulator
    config = Config.model_validate(config_dict)
    simulator = RollingWindowSimulator(config)

    # Set data directly (already loaded in main process)
    simulator._V_obs = V_obs
    simulator._inflow = inflow
    simulator._z = z
    if dates is not None:
        simulator._dates = pd.DatetimeIndex(dates)

    # Run single window
    result = simulator.run_window(train_start, train_end, test_end)
    result.window_id = window_id

    return result
