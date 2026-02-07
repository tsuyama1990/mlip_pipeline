from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.structure import Structure


class Potential(BaseModel):
    """
    Metadata for a machine learning interatomic potential.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    path: Path
    format: Literal["yace", "pb", "pt"] = "yace"

class ExplorationResult(BaseModel):
    """
    Result of a molecular dynamics or exploration run.
    """
    model_config = ConfigDict(extra="forbid")

    trajectory: list[Structure]
    status: Literal["halted", "converged", "max_steps", "failed"]
    reason: str | None = None
