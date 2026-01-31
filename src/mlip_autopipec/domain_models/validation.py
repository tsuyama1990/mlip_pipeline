from pathlib import Path
from typing import Literal, Optional
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValidationMetric(BaseModel):
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
