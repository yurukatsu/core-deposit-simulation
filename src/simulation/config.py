"""Configuration schemas using Pydantic."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CovariateConfig(BaseModel):
    """Covariate data configuration."""

    path: Path = Field(..., description="Path to the covariate data file (CSV)")
    columns: list[str] = Field(..., min_length=1, description="Column names to use as covariates")
    date_column: str = Field(default="date", description="Column name for date index (for joining)")


class DataConfig(BaseModel):
    """Data source configuration."""

    path: Path = Field(..., description="Path to the data file (CSV)")
    volume_column: str = Field(default="volume", description="Column name for deposit volume")
    inflow_column: str = Field(default="inflow", description="Column name for inflow")
    date_column: str | None = Field(default=None, description="Column name for date index")
    covariate: CovariateConfig | None = Field(default=None, description="Covariate data configuration")


class ModelConfig(BaseModel):
    """Model estimation configuration."""

    estimator: Literal["nls", "map", "mcmc"] = Field(default="nls", description="Estimator type")

    # NLS/MAP common
    fix_m: float | None = Field(default=None, description="Fixed value for m parameter")
    loss: str = Field(default="soft_l1", description="Loss function for NLS (soft_l1, huber, cauchy)")
    f_scale: float = Field(default=0.05, description="Scaling factor for robust loss functions")

    # MCMC specific
    num_warmup: int = Field(default=1000, description="MCMC warmup iterations")
    num_samples: int = Field(default=2000, description="MCMC sample iterations")
    num_chains: int = Field(default=1, description="Number of MCMC chains")
    ar_errors: bool = Field(default=False, description="Use AR(1) error model")
    likelihood: Literal["normal", "studentt"] = Field(
        default="studentt", description="Likelihood type for MCMC"
    )

    # NLS initialization for MCMC
    init_from_nls: bool = Field(default=True, description="Initialize MCMC from NLS estimates")

    # MCMC convergence and retry settings
    rhat_threshold: float = Field(default=1.1, description="R-hat threshold for convergence check")
    max_retries: int = Field(default=2, description="Maximum retry attempts for MCMC convergence")
    retry_multiplier: float = Field(default=1.5, description="Multiplier for warmup/samples on retry")


class SimulationConfig(BaseModel):
    """Rolling window simulation configuration."""

    t_train: int = Field(..., gt=0, description="Training period length")
    t_test: int = Field(..., ge=0, description="Test period length (0 = in-sample only)")
    t_gap: int = Field(default=1, gt=0, description="Sliding step size")
    max_windows: int | None = Field(default=None, gt=0, description="Maximum number of windows (None = no limit)")

    # Index-based bounds (legacy)
    start_index: int | None = Field(default=None, description="Start index (0-based)")
    end_index: int | None = Field(default=None, description="End index (exclusive)")

    # Date-based bounds (preferred)
    start_date: date | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: date | None = Field(default=None, description="End date (YYYY-MM-DD, exclusive)")

    # Parallel processing
    n_jobs: int = Field(default=1, description="Number of parallel workers (1=sequential)")

    @model_validator(mode="after")
    def validate_bounds(self) -> "SimulationConfig":
        # Check that index and date bounds are not mixed
        has_index = self.start_index is not None or self.end_index is not None
        has_date = self.start_date is not None or self.end_date is not None
        if has_index and has_date:
            raise ValueError("Cannot mix index-based and date-based bounds. Use either start_index/end_index or start_date/end_date.")

        # Validate end > start for dates
        if self.start_date is not None and self.end_date is not None:
            if self.end_date <= self.start_date:
                raise ValueError("end_date must be after start_date")

        # Validate end > start for indices
        if self.start_index is not None and self.end_index is not None:
            if self.end_index <= self.start_index:
                raise ValueError("end_index must be greater than start_index")

        return self


class MLFlowConfig(BaseModel):
    """MLFlow configuration."""

    tracking_uri: str = Field(
        default="http://192.168.1.12:15000", description="MLFlow tracking server URI"
    )
    experiment_name: str = Field(
        default="core-deposit-simulation", description="MLFlow experiment name"
    )
    run_name: str | None = Field(default=None, description="MLFlow run name")
    artifact_location: str | None = Field(
        default=None, description="Artifact storage location (e.g., s3://bucket/path)"
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Additional tags")


class Config(BaseModel):
    """Root configuration."""

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    simulation: SimulationConfig
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated Config object
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
