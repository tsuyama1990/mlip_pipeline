import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path(os.getenv("MLIP_LOG_FILENAME", "mlip_pipeline.log"))


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float
    seed: int = 42

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "Elements list cannot be empty"
            raise ValueError(msg)
        return v


class StructureGenConfig(BaseModel):
    """Configuration for Structure Generation."""
    model_config = ConfigDict(extra="forbid")
    method: str = "random"
    # Placeholder for future params


class OracleConfig(BaseModel):
    """Configuration for DFT Oracle."""
    model_config = ConfigDict(extra="forbid")
    scf_k_points: list[int] = Field(default_factory=lambda: [1, 1, 1])
    # Placeholder


class TrainerConfig(BaseModel):
    """Configuration for Potential Training."""
    model_config = ConfigDict(extra="forbid")
    max_epochs: int = 100

    @field_validator("max_epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        if v <= 0:
            msg = "Max epochs must be positive"
            raise ValueError(msg)
        return v


class DynamicsConfig(BaseModel):
    """Configuration for MD/kMC Dynamics."""
    model_config = ConfigDict(extra="forbid")
    timestep: float = 1.0  # fs
    # Placeholder


class OrchestratorConfig(BaseModel):
    """Configuration for Orchestrator."""
    model_config = ConfigDict(extra="forbid")
    max_cycles: int = 10
    state_file: Path = Path(os.getenv("MLIP_STATE_FILENAME", "workflow_state.json"))


class Config(BaseModel):
    """Root configuration object."""
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    potential: PotentialConfig
    structure_gen: StructureGenConfig = Field(default_factory=StructureGenConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        if not v.strip():
            msg = "Project name cannot be empty"
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
