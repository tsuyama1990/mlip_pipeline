from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """
    Domain model representing the result of a single validation metric.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the metric")
    passed: bool = Field(..., description="Whether the metric passed the criteria")
    score: float | None = Field(None, description="Numerical score if applicable")
    details: dict[str, Any] | None = Field(None, description="Detailed results or metadata")


class ValidationResult(BaseModel):
    """
    Domain model representing the overall result of a validation process.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Overall validation status")
    metrics: list[MetricResult] = Field(
        default_factory=list, description="List of individual metric results"
    )
    report_path: str | None = Field(None, description="Path to the detailed validation report file")
