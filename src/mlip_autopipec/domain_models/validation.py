from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class ValidationMetric(BaseModel):
    """A single validation metric (e.g., Bulk Modulus)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    reference: Optional[float] = None
    error: Optional[float] = None
    message: Optional[str] = None
    passed: bool


class ValidationResult(BaseModel):
    """Result of a validation run."""

    model_config = ConfigDict(extra="forbid")

    potential_id: str
    metrics: list[ValidationMetric]
    plots: dict[str, Path]
    overall_status: Literal["PASS", "WARN", "FAIL"]
