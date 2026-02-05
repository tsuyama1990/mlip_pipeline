from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExplorationConfig(BaseModel):
    """Configuration for the exploration (structure generation) phase."""
    model_config = ConfigDict(extra='forbid')

    strategy: str = "random"
    max_structures: int = 10
    seed_structure_path: str = "seeds/structure.xyz"

class DFTConfig(BaseModel):
    """Configuration for the DFT calculation phase (Oracle)."""
    model_config = ConfigDict(extra='forbid')

    calculator: str = "mock"
    encut: float = 500.0
    kpoints_density: float = 0.04

class TrainingConfig(BaseModel):
    """Configuration for the potential training phase."""
    model_config = ConfigDict(extra='forbid')

    trainer: str = "mock"
    epochs: int = 10
    energy_weight: float = 1.0
    force_weight: float = 0.1

class GlobalConfig(BaseModel):
    """
    Root configuration object for the MLIP pipeline.
    Enforces strict typing and forbids extra fields to prevent configuration drift.
    """
    model_config = ConfigDict(extra='forbid')

    work_dir: Path
    logging_level: str = "INFO"
    random_seed: int = 42
    max_cycles: int = 1

    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    dft: DFTConfig = Field(default_factory=DFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
