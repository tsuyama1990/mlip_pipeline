from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """Result of a single validation metric."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the metric")
    passed: bool = Field(..., description="Whether the metric passed")
    score: float | None = Field(None, description="Quantitative score for the metric")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ValidationResult(BaseModel):
    """Overall result of the validation phase."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Whether the overall validation passed")
    metrics: list[MetricResult] = Field(default_factory=list, description="List of metric results")
    report_path: str | None = Field(None, description="Path to the validation report")
