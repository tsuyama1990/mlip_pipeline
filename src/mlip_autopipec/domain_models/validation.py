from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """
    Result of a validation step.
    """

    model_config = ConfigDict(extra="forbid")
    passed: bool
    metrics: dict[str, float]
    report_path: Path | None = None
    details: str | None = None
