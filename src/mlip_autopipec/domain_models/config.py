from pathlib import Path
from typing import Literal, Optional, Union

import ase.data
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
    lammps_command: str = "lmp"

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

    @model_validator(mode="after")
    def check_delta_learning(self) -> "PotentialConfig":
        """
        Enforce Delta Learning (hybrid/overlay) for physical elements (Z >= 2).
        Spec Section 3.3.
        """
        # Check if any element has Z >= 2 (He and above)
        # H (Z=1) might be exempt, but typically we want it for all core repulsions.
        # Feedback explicitly said "nuclear charge >= 2".
        has_heavy_atoms = False
        for el in self.elements:
            try:
                z = ase.data.atomic_numbers[el]
                if z >= 2:
                    has_heavy_atoms = True
                    break
            except KeyError:
                # If element is not standard, we skip check or assume it's custom
                pass

        if has_heavy_atoms and self.pair_style != "hybrid/overlay":
             raise ValueError(
                 "Elements with Z >= 2 found. You MUST use 'hybrid/overlay' pair_style "
                 "to enforce physical core repulsion (Delta Learning). "
                 "See SPEC.md Section 3.3."
             )

        return self


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = 1
    uncertainty_threshold: float = 5.0
    halt_threshold: int = 5
    validation_frequency: int = 1

    # Active Set Selection
    active_set_optimization: bool = True
    max_active_set_size: int = 1000

    @field_validator("max_active_set_size")
    @classmethod
    def validate_max_active_set_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_active_set_size must be positive")
        return v


class BulkStructureGenConfig(BaseModel):
    """Configuration for bulk structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["bulk"] = "bulk"
    element: str
    crystal_structure: str
    lattice_constant: float
    rattle_stdev: float = 0.0
    supercell: tuple[int, int, int] = (1, 1, 1)


class RandomSliceStructureGenConfig(BaseModel):
    """Configuration for random slice structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random_slice"] = "random_slice"
    element: str
    crystal_structure: str
    lattice_constant: float
    supercell: tuple[int, int, int] = (2, 2, 2)
    vacuum: float = 10.0
    n_structures: int = 5


class DefectStructureGenConfig(BaseModel):
    """Configuration for defect structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["defect"] = "defect"
    base_structure_path: Path
    defect_type: Literal["vacancy", "interstitial", "substitution"]
    concentration: float = 0.01


class StrainStructureGenConfig(BaseModel):
    """Configuration for strained structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["strain"] = "strain"
    base_structure_path: Path
    strain_range: float = 0.1  # +/- 10%
    n_points: int = 10


# Union of all structure generation configs
StructureGenConfig = Union[
    BulkStructureGenConfig,
    RandomSliceStructureGenConfig,
    DefectStructureGenConfig,
    StrainStructureGenConfig
]


class ValidationConfig(BaseModel):
    """Configuration for the Validation Framework."""

    model_config = ConfigDict(extra="forbid")

    # Phonon
    phonon_tolerance: float = -0.05  # THz, strictly negative to allow numeric noise
    phonon_supercell: tuple[int, int, int] = (2, 2, 2)

    # Elastic
    elastic_stability_tolerance: float = 1e-4  # GPa

    # EOS
    eos_vol_range: float = 0.1  # +/- 10%
    eos_n_points: int = 10

    # Reporting
    report_path: Path = Path("validation_report.html")
    template_dir: Optional[Path] = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

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
