from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")

class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float = 5.0
    seed: int = 42

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v

class ExplorationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = 10000
    time_step: float = 0.001  # ps
    temperature: float = 300.0
    pressure: float = 0.0
    batch_size: int = 10  # Number of structures to generate/explore

class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncertainty_threshold: float = 5.0  # Gamma threshold
    n_candidates: int = 5  # Number of candidates to select

class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = "pw.x"
    kspacing: float = 0.04
    smearing_width: float = 0.01

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 50
    batch_size: int = 32
    energy_weight: float = 100.0
    force_weight: float = 1.0

class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy_rmse_threshold: float = 0.002  # eV/atom
    force_rmse_threshold: float = 0.05    # eV/A
    check_phonons: bool = True
    check_elasticity: bool = True

class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cycles: int = 10
    work_dir: Path = Path("active_learning")

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig

    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    dft: DFTConfig = Field(default_factory=DFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
