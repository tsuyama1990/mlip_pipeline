from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class ValidationMetric(BaseModel):
    """
    Represents a single validation metric (e.g. Bulk Modulus).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    reference: Optional[float] = None
    error: Optional[float] = None
    passed: bool
    unit: Optional[str] = None


class ValidationResult(BaseModel):
    """
    Aggregate result of a validation run.
    """

    model_config = ConfigDict(extra="forbid")

    potential_id: str
    metrics: list[ValidationMetric]
    plots: dict[str, Path]
    overall_status: Literal["PASS", "WARN", "FAIL"]


class ValidationConfig(BaseModel):
    """
    Configuration for physics validation.
    """

    model_config = ConfigDict(extra="forbid")

    phonon_tolerance: float = -0.1  # THz
    eos_vol_range: float = 0.1  # +/- 10%
    eos_n_points: int = 10
    elastic_strain_mag: float = 0.01
