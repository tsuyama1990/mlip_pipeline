import os
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
    cutoff: float
    seed: int = 42

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v


class MDParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 300.0
    pressure: float = 0.0
    timestep: float = 0.001
    n_steps: int = 100


class LammpsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(default_factory=lambda: os.getenv("LAMMPS_COMMAND", "lmp_serial"))
    cores: int = 1
    timeout: float = 3600.0
    seed: int = 42


class ExplorationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "random"
    supercell_size: list[int] = Field(default_factory=lambda: [1, 1, 1])
    rattle_amplitude: float = 0.0
    num_candidates: int = 1
    composition: str # Made mandatory to avoid hardcoding defaults

    # Fields for Cycle 02
    lattice_constant: float = 5.43
    md_params: MDParams = Field(default_factory=MDParams)


class OrchestratorConfig(BaseModel):
    """Placeholder for Orchestrator configuration."""
    model_config = ConfigDict(extra="forbid")
    # No fields defined in Cycle 01 spec yet, allowing empty or defaults if needed later.


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    exploration: ExplorationConfig # Removed default factory as composition is now mandatory
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
