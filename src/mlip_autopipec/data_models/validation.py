from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationMetric(BaseModel):
    """
    Represents a single physical property measurement and its validity.
    """

    name: str
    value: float | str | bool | list[float]
    unit: str | None = None
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ValidationResult(BaseModel):
    """
    Aggregates validation metrics for a specific physics module.
    """

    module: str = Field(..., description="Name of the validation module (e.g., phonon)")
    passed: bool
    metrics: list[ValidationMetric] = Field(default_factory=list)
    error: str | None = Field(None, description="Error message if validation failed execution")

    model_config = ConfigDict(extra="forbid")
