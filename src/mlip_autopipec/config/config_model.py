from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DFTConfig(BaseModel):
    """Configuration for DFT calculations."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["qe", "vasp"] = Field(..., description="DFT code to use")
    ecutwfc: float = Field(..., gt=0, description="Wavefunction cutoff energy in Ry")
    kpoints: list[int] = Field(
        ..., min_length=3, max_length=3, description="K-points grid (nkx, nky, nkz)"
    )


class TrainingConfig(BaseModel):
    """Configuration for potential training."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["pacemaker"] = Field(..., description="Training code to use")
    cutoff: float = Field(..., gt=0, description="Interaction cutoff radius in Angstrom")


class ExplorationConfig(BaseModel):
    """Configuration for structure exploration."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random", "md", "strain"] = Field(
        default="random", description="Exploration strategy"
    )
    max_temperature: float = Field(default=300.0, ge=0, description="Maximum temperature for MD/MC")
    steps: int = Field(default=10, gt=0, description="Number of exploration steps")


class SimulationConfig(BaseModel):
    """Root configuration for the simulation."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(..., min_length=1, description="Name of the project")
    dft: DFTConfig = Field(..., description="DFT configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    exploration: ExplorationConfig = Field(
        default_factory=ExplorationConfig, description="Exploration configuration"
    )
