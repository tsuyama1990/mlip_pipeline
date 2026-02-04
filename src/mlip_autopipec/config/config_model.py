from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class DFTConfig(BaseModel):
    """Configuration for the DFT Oracle."""

    model_config = ConfigDict(extra="forbid")

    calculator: Literal["espresso", "vasp", "lj"] = Field(
        ..., description="DFT code or calculator to use"
    )
    kpoints_density: PositiveFloat = Field(0.04, description="K-points density in 1/A")
    encut: PositiveFloat = Field(500.0, description="Energy cutoff in eV")


class TrainingConfig(BaseModel):
    """Configuration for the Potential Trainer."""

    model_config = ConfigDict(extra="forbid")

    potential_type: Literal["ace", "mnn"] = Field("ace", description="Type of potential")
    cutoff: PositiveFloat = Field(5.0, description="Cutoff radius in Angstrom")
    max_degree: PositiveInt = Field(1, description="Max polynomial degree (ACE)")


class ExplorationConfig(BaseModel):
    """Configuration for the Structure Generator."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random", "active"] = Field("random", description="Exploration strategy")
    num_candidates: PositiveInt = Field(
        10, description="Number of candidates to generate per cycle"
    )
    supercell_size: PositiveInt = Field(2, description="Size of supercell (NxNxN)")


class GlobalConfig(BaseModel):
    """Global configuration for the MLIP pipeline."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(..., min_length=1, description="Name of the project")
    execution_mode: Literal["production", "dry_run", "mock"] = Field(
        "mock", description="Execution mode"
    )
    cycles: PositiveInt = Field(1, description="Number of active learning cycles")

    dft: DFTConfig
    training: TrainingConfig
    exploration: ExplorationConfig
