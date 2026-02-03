from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProductionManifest(BaseModel):
    version: str
    author: str
    training_set_size: int = Field(ge=0)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    creation_date: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(extra="forbid")
