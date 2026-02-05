
from pydantic import BaseModel, ConfigDict


class MetricResult(BaseModel):
    """
    Result of a single validation metric (e.g., Elastic Tensor match).
    """
    model_config = ConfigDict(extra='forbid')

    name: str
    passed: bool
    score: float
    details: str | None = None

class ValidationResult(BaseModel):
    """
    Aggregate result of a validation run.
    """
    model_config = ConfigDict(extra='forbid')

    passed: bool
    metrics: list[MetricResult]
    report_path: str | None = None
