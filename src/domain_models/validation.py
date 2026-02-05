from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """
    Result of a potential validation process.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Whether the potential passed validation")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Validation metrics")
