from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Metadata for a machine learning interatomic potential.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
