from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Represents a trained machine learning potential artifact.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["ace", "mace", "gap"]
    elements: list[str]
    creation_date: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
