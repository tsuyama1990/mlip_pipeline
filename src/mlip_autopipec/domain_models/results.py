from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationMetrics(BaseModel):
    """
    Standardized metrics returned by a Validator component.
    """
    model_config = ConfigDict(extra="forbid")

    energy_rmse: float | None = None
    force_rmse: float | None = None
    stress_rmse: float | None = None

    # Validation status
    passed: bool = Field(default=False, description="Whether the validation criteria were met.")

    # Detailed breakdown (optional)
    details: dict[str, Any] = Field(default_factory=dict)
