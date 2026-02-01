from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Represents a trained potential artifact.
    Includes lineage metadata to track which iteration produced it.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: Literal["ace", "mace", "gap"]
    elements: list[str]
    creation_date: datetime = Field(default_factory=datetime.now)

    # Lineage / Metadata
    iteration: Optional[int] = None
    parent_potential_path: Optional[Path] = None
    metadata: dict = Field(default_factory=dict)
