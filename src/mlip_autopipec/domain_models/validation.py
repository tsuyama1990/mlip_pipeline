from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricResult(BaseModel):
    name: str
    passed: bool
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    plot_path: Path | None = None

    model_config = ConfigDict(extra="forbid")


class ValidationResult(BaseModel):
    passed: bool
    metrics: list[MetricResult] = Field(default_factory=list)
    report_path: Path | None = None
    reason: str | None = None

    model_config = ConfigDict(extra="forbid")
