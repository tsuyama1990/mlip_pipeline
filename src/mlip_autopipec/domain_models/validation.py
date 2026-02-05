from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """
    Result of a single validation metric.
    """
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description="Name of the metric (e.g., 'Elastic Tensor Error').")
    passed: bool = Field(..., description="Whether the metric passed the threshold.")
    score: float = Field(..., description="The calculated score or error value.")
    threshold: float | None = Field(None, description="The threshold used for pass/fail.")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details about the metric.")

class ValidationResult(BaseModel):
    """
    Result of a validation run, containing multiple metrics.
    """
    model_config = ConfigDict(extra='forbid')

    passed: bool = Field(..., description="Overall pass/fail status of the validation.")
    metrics: list[MetricResult] = Field(default_factory=list, description="List of individual metric results.")
    report_path: str | None = Field(None, description="Path to the detailed validation report (e.g., PDF/HTML).")
