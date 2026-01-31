from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Potential(BaseModel):
    """
    Represents a trained potential artifact.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["ace", "mace", "gap"]
    elements: list[str]
    creation_date: datetime
    metadata: dict
