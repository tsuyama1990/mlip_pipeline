from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Represents a trained potential model artifact (domain model).
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: str = "yace"
    metadata: dict[str, Any] = Field(default_factory=dict)
