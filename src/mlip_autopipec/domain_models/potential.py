from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    format: str = "yace"
    species: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    creation_date: datetime = Field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return f"<Potential(path={self.path}, format={self.format})>"

    def __str__(self) -> str:
        return f"Potential({self.path.name})"
