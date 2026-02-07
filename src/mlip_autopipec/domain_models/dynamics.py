from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .structure import Structure


class ExplorationResult(BaseModel):
    """
    Result of a dynamics exploration run (domain model).
    """

    model_config = ConfigDict(extra="forbid")

    structures: list[Structure] = Field(default_factory=list)
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
