from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    format: str = "yace"
    metrics: dict[str, Any] = Field(default_factory=dict)
    creation_date: datetime = Field(default_factory=datetime.now)
