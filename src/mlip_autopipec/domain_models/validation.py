from typing import Any

from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """
    Result of a validation process.
    """

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    passed: bool
    details: dict[str, Any] = {}
