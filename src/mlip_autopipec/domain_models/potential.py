from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


class ExplorationResult(BaseModel):
    """
    Result of an exploration run (e.g., MD or MC).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    halted: bool
    reason: str | None = None
    structures: list[Structure] = Field(default_factory=list)


class Potential(BaseModel):
    """
    Represents a machine learning potential.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    metadata: dict[str, Any] | None = None
