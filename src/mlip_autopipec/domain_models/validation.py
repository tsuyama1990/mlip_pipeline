from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    passed: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    report_path: Path | None = None
    reason: str | None = None

    model_config = ConfigDict(extra="forbid")
