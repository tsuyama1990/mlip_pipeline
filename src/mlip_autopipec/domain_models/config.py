from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.dynamics import LammpsConfig, MDConfig
from mlip_autopipec.domain_models.training import TrainingConfig


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float
    seed: int = 42

    # Hybrid Potential Settings (ACE + ZBL)
    pair_style: Literal["pace", "hybrid/overlay"] = "pace"
    zbl_inner_cutoff: float = 1.0
    zbl_outer_cutoff: float = 2.0

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v

    @field_validator("zbl_inner_cutoff", "zbl_outer_cutoff")
    @classmethod
    def validate_zbl_values(cls, v: float) -> float:
        if v < 0:
            raise ValueError("ZBL cutoffs must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_hybrid_params(self) -> "PotentialConfig":
        if self.pair_style == "hybrid/overlay":
            # Just ensure cutoffs are sensible if hybrid is used.
            # They have defaults, but we can check relative order
            if self.zbl_inner_cutoff > self.zbl_outer_cutoff:
                raise ValueError("zbl_inner_cutoff must be <= zbl_outer_cutoff")
        return self


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = 1
    uncertainty_threshold: float = 5.0

    # Active Set Selection
    active_set_optimization: bool = True
    max_active_set_size: int = 1000


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


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon_tolerance: float = -0.1
    phonon_supercell: tuple[int, int, int] = (2, 2, 2)
    eos_vol_range: float = 0.1
    eos_n_points: int = 10
    elastic_strain: float = 0.01
    elastic_stability_tolerance: float = 0.0


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
    validation: Optional[ValidationConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        return cls(**data)
