from datetime import datetime
from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict


class Potential(BaseModel):
    """
    Represents a trained Machine Learning Potential artifact.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["ace", "mace", "gap"]
    elements: list[str]
    creation_date: datetime
    metadata: dict[str, Any]
