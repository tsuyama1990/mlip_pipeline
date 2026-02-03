from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DFTConfig(BaseModel):
    """Configuration for Density Functional Theory calculations."""
    model_config = ConfigDict(extra='forbid')

    code: Literal['qe', 'vasp'] = Field(..., description="DFT code to use (e.g., 'qe', 'vasp')")
    ecutwfc: float = Field(..., gt=0, description="Wavefunction cutoff energy in Ry")
    kpoints: list[int] = Field(..., min_length=3, max_length=3, description="K-points mesh [x, y, z]")


class TrainingConfig(BaseModel):
    """Configuration for ML Potential Training."""
    model_config = ConfigDict(extra='forbid')

    code: Literal['pacemaker'] = Field(..., description="Training code to use")
    cutoff: float = Field(..., gt=0, description="Radial cutoff for the potential in Angstroms")
    max_generations: int = Field(default=100, ge=1, description="Maximum number of active learning generations")


class ExplorationConfig(BaseModel):
    """Configuration for Structure Exploration."""
    model_config = ConfigDict(extra='forbid')

    strategy: Literal['random', 'md', 'mutation'] = Field(default='random', description="Exploration strategy")
    max_temperature: float = Field(default=1000.0, gt=0, description="Maximum temperature for MD exploration")
    steps: int = Field(default=1000, ge=1, description="Number of exploration steps")


class SimulationConfig(BaseModel):
    """Root configuration for the simulation."""
    model_config = ConfigDict(extra='forbid')

    project_name: str = Field(..., min_length=1, description="Name of the project")
    dft: DFTConfig = Field(..., description="DFT configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    exploration: ExplorationConfig = Field(default_factory=lambda: ExplorationConfig(), description="Exploration configuration")
