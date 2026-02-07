from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """
    Result of a validation run.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] | None = None
