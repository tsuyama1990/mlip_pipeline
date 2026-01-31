from pathlib import Path
from typing import Literal, Optional
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValidationMetric(BaseModel):
    """
    Represents a single validation metric result.

    Attributes:
        name: Name of the metric (e.g., "RMSE Energy").
        value: The calculated numerical value.
        reference: Optional reference value (e.g., DFT ground truth).
        error: Optional calculated error (e.g., |value - reference|).
        passed: Boolean indicating if the metric passed the criteria.
        message: Optional descriptive message or failure reason.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    reference: Optional[float] = None
    error: Optional[float] = None
    passed: bool
    message: Optional[str] = None

    @field_validator("value", "reference", "error", mode="before")
    @classmethod
    def check_finite(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
             if not math.isfinite(v):
                  raise ValueError("Metric values must be finite numbers")
        return v


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_id: str
    metrics: list[ValidationMetric]
    plots: dict[str, Path] = Field(default_factory=dict)
    overall_status: Literal["PASS", "WARN", "FAIL"]
