from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ValidationMetric(BaseModel):
    """Represents a single validation metric (e.g., Bulk Modulus)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    reference: Optional[float] = None
    error: Optional[float] = None
    error_message: Optional[str] = None
    passed: bool


class ValidationResult(BaseModel):
    """Aggregates results from a validation run."""

    model_config = ConfigDict(extra="forbid")

    potential_id: str
    metrics: list[ValidationMetric] = Field(default_factory=list)
    plots: dict[str, Path] = Field(default_factory=dict)
    overall_status: Literal["PASS", "WARN", "FAIL"] = "FAIL"
