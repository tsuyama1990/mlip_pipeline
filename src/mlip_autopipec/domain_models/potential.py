from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


class Potential(BaseModel):
    """
    Domain model representing a trained interatomic potential.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplorationResult(BaseModel):
    """
    Domain model representing the result of an exploration (dynamics) run.
    """

    model_config = ConfigDict(extra="forbid")

    converged: bool
    structures: list[Structure] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
