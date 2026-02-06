from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """
    Domain model representing the validation results of a potential.
    """
    passed: bool = Field(description="Whether the validation passed")
    metrics: dict[str, float] = Field(default_factory=dict, description="Dictionary of validation metrics (e.g., RMSE, R2)")
    report_path: Path | None = Field(default=None, description="Path to the validation report file")

    def __str__(self) -> str:
        return f"ValidationResult(passed={self.passed}, metrics={self.metrics})"

    model_config = ConfigDict(extra="forbid")
