from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the metric")
    passed: bool = Field(..., description="Whether the metric passed")
    score: float = Field(..., description="Numerical score of the metric")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Overall validation status")
    metrics: list[MetricResult] = Field(default_factory=list, description="List of metric results")
    report_path: str | None = Field(None, description="Path to detailed report")
