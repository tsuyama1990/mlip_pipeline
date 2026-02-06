from typing import Any

from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """
    Result of a validation process.
    """

    metrics: dict[str, Any]
    is_acceptable: bool

    model_config = ConfigDict(extra="forbid")
