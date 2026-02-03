from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExplorationMethod(str, Enum):
    MD = "molecular_dynamics"
    STATIC = "static_displacement"
    AKMC = "adaptive_kmc"


class ExplorationTask(BaseModel):
    method: ExplorationMethod
    parameters: dict[str, Any] = Field(default_factory=dict)
    modifiers: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
