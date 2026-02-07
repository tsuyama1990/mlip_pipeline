from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .structure import Structure


class Potential(BaseModel):
    """
    Domain model representing a trained interatomic potential.
    """

    model_config = ConfigDict(extra="forbid")
    path: Path


class ExplorationStatus(str, Enum):
    """
    Status of a molecular dynamics exploration run.
    """

    CONVERGED = "converged"
    HALTED = "halted"
    FAILED = "failed"


class ExplorationResult(BaseModel):
    """
    Result of a molecular dynamics exploration run.
    """

    model_config = ConfigDict(extra="forbid")
    final_structure: Structure
    trajectory_path: Path
    status: ExplorationStatus
    max_uncertainty: float | None = None
