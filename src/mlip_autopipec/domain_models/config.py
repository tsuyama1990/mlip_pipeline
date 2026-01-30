from pathlib import Path
from typing import Literal, Optional, Union
import os

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.dynamics import LammpsConfig, MDConfig
from mlip_autopipec.domain_models.training import TrainingConfig


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(default_factory=lambda: os.environ.get("MLIP_DEFAULT_ELEMENTS", "Si").split(","))
    cutoff: float
    seed: int = 42

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = 1
    uncertainty_threshold: float = 5.0


class BulkStructureGenConfig(BaseModel):
    """Configuration for bulk structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["bulk"] = "bulk"
    element: str
    crystal_structure: str
    lattice_constant: float
    rattle_stdev: float = 0.0
    supercell: tuple[int, int, int] = (1, 1, 1)


class SurfaceStructureGenConfig(BaseModel):
    """Configuration for surface structure generation (Placeholder)."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["surface"] = "surface"
    element: str
    facet: tuple[int, int, int]
    layers: int
    vacuum: float


# Union of all structure generation configs
StructureGenConfig = Union[BulkStructureGenConfig, SurfaceStructureGenConfig]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)

    # New configurations
    structure_gen: StructureGenConfig = Field(discriminator="strategy")
    md: MDConfig

    # Optional components
    dft: Optional[DFTConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        return cls(**data)
