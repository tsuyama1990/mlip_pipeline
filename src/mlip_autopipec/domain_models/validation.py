from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """
    Represents the result of a validation run (domain model).
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    report_path: Path | None = None
    details: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
