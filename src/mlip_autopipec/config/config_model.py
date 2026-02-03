from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal["qe", "vasp"] = Field(..., description="DFT code to use")
    ecutwfc: float = Field(..., description="Wavefunction cutoff energy in Ry")
    kpoints: list[int] = Field(..., description="K-points grid [kx, ky, kz]")
    pseudopotentials: dict[str, str] | None = Field(None, description="Path to pseudopotentials")

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal["pacemaker"] = Field(..., description="Training code to use")
    cutoff: float = Field(..., description="Cutoff radius in Angstrom")
    max_generations: int = Field(100, description="Maximum number of generations for active learning")

class ExplorationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random", "md", "mc"] = Field("random", description="Exploration strategy")
    max_temperature: float = Field(1000.0, description="Maximum temperature for exploration")
    steps: int = Field(100, description="Number of steps for exploration")

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(..., description="Name of the project")
    dft: DFTConfig = Field(..., description="DFT configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    exploration: ExplorationConfig = Field(default_factory=lambda: ExplorationConfig(), description="Exploration configuration")
