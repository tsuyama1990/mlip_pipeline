from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.inputs import Structure


class CalculationResult(BaseModel):
    """Result of a DFT calculation."""

    model_config = ConfigDict(extra="forbid")

    structure: Structure
    energy: float
    forces: list[list[float]]
    stress: list[float]
    converged: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PotentialArtifact(BaseModel):
    """Represents a trained potential file."""

    model_config = ConfigDict(extra="forbid")

    path: str
    format: str = "yace"
    version: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ExplorationResult(BaseModel):
    """Result of a Dynamics exploration."""

    model_config = ConfigDict(extra="forbid")

    halted_structures: list[Structure] = Field(default_factory=list)
    total_steps: int
    halt_count: int
    exploration_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)
