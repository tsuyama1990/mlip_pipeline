from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    """Result of a single validation metric."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description="Name of the metric")
    passed: bool = Field(..., description="Whether the metric passed")
    score: float = Field(..., description="Quantitative score of the metric")
    details: str | None = Field(None, description="Detailed message or data")


class ValidationResult(BaseModel):
    """Result of a validation suite."""
    model_config = ConfigDict(extra='forbid')

    passed: bool = Field(..., description="Overall pass status")
    metrics: list[MetricResult] = Field(default_factory=list, description="List of individual metric results")
    report_path: str | None = Field(None, description="Path to the detailed validation report")
