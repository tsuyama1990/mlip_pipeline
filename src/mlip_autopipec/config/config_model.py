from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplorationConfig(BaseModel):
    """Configuration for structure exploration/generation."""

    model_config = ConfigDict(extra="forbid")

    strategy_name: str = Field("random", description="Name of the exploration strategy to use")
    max_structures: int = Field(
        10, description="Maximum number of structures to generate per cycle"
    )


class DFTConfig(BaseModel):
    """Configuration for DFT calculations (Oracle)."""

    model_config = ConfigDict(extra="forbid")

    calculator: str = Field("espresso", description="Name of the DFT calculator")
    encut: float = Field(500.0, description="Energy cutoff in eV")
    kpoints: tuple[int, int, int] = Field((1, 1, 1), description="K-points grid (k_x, k_y, k_z)")


class TrainingConfig(BaseModel):
    """Configuration for potential training."""

    model_config = ConfigDict(extra="forbid")

    fitting_code: str = Field("pacemaker", description="Name of the fitting code")
    max_epochs: int = Field(100, description="Maximum number of training epochs")


class GlobalConfig(BaseModel):
    """Root configuration for the MLIP pipeline."""

    model_config = ConfigDict(extra="forbid")

    execution_mode: Literal["mock", "production"] = Field(
        "mock", description="Execution mode: 'mock' for testing, 'production' for real runs"
    )
    max_cycles: int = Field(5, description="Maximum number of active learning cycles to run")
    exploration: ExplorationConfig = Field(
        default_factory=lambda: ExplorationConfig(), description="Exploration settings"
    )
    dft: DFTConfig = Field(default_factory=lambda: DFTConfig(), description="DFT settings")
    training: TrainingConfig = Field(
        default_factory=lambda: TrainingConfig(), description="Training settings"
    )
