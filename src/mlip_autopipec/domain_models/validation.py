from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """
    Domain model representing the outcome of a validation run.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    report_path: Path | None = None
    details: dict[str, Any] = Field(default_factory=dict)
