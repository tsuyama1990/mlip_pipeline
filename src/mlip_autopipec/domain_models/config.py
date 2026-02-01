from pathlib import Path
from typing import Literal, Optional, Union, List

import ase.data
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.dynamics import LammpsConfig, MDConfig
from mlip_autopipec.domain_models.training import TrainingConfig


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class ACEConfig(BaseModel):
    """Configuration for ACE (Pacemaker) basis set."""
    model_config = ConfigDict(extra="forbid")

    npot: str = Field(default="FinnisSinclair", description="Potential type (e.g. FinnisSinclair)")
    fs_parameters: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 0.5], description="Finnis-Sinclair parameters")
    ndensity: int = Field(default=2, description="Density of basis functions")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(default_factory=list)
    cutoff: float = 5.0
    seed: int = 42
    lammps_command: str = "lmp"

    # Hybrid Potential Settings (ACE + ZBL)
    pair_style: Literal["pace", "hybrid/overlay"] = "hybrid/overlay"
    zbl_inner_cutoff: float = 1.0
    zbl_outer_cutoff: float = 2.0

    # Pacemaker / ACE Basis Parameters
    ace_params: ACEConfig = Field(default_factory=ACEConfig, description="ACE basis parameters")

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
        self._validate_hybrid_style()
        return self

    def _validate_hybrid_style(self):
        """Internal helper to validate hybrid style requirements."""
        if not self.elements:
            # If elements list is empty, we can't validate yet, maybe allow it?
            # Or fail? The field allows default_factory=list, so empty is possible initially.
            return

        has_heavy_atoms = False
        for el in self.elements:
            if el not in ase.data.atomic_numbers:
                raise ValueError(f"Invalid element symbol: {el}")

            z = ase.data.atomic_numbers[el]
            if z >= 2:
                has_heavy_atoms = True

        if has_heavy_atoms and self.pair_style != "hybrid/overlay":
             raise ValueError(
                 "Elements with Z >= 2 found. You MUST use 'hybrid/overlay' pair_style "
                 "to enforce physical core repulsion (Delta Learning). "
                 "See SPEC.md Section 3.3."
             )


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = 1
    uncertainty_threshold: float = 5.0
    halt_threshold: int = 5
    validation_frequency: int = 1

    # Sampling & Batching
    trajectory_sampling_stride: int = 1
    dft_batch_size: int = 10

    # Paths
    data_dir: Path = Path("data")

    @field_validator("trajectory_sampling_stride", "dft_batch_size")
    @classmethod
    def validate_positive_ints(cls, v: int, info: ValidationInfo) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v


class BulkStructureGenConfig(BaseModel):
    """Configuration for bulk structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["bulk"] = "bulk"
    element: str = "Si"
    crystal_structure: str = "diamond"
    lattice_constant: float = 5.43
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

    # Structure Prep
    validation_rattle_stdev: float = 0.0

    @field_validator("validation_rattle_stdev")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("validation_rattle_stdev must be non-negative")
        return v


class PolicyConfig(BaseModel):
    """Configuration for the Adaptive Exploration Policy."""
    model_config = ConfigDict(extra="forbid")

    is_metal: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = "MyMLIPProject"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig = Field(default_factory=PotentialConfig)
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    # New configurations
    structure_gen: StructureGenConfig = Field(default_factory=BulkStructureGenConfig, discriminator="strategy")
    md: MDConfig = Field(default_factory=lambda: MDConfig(temperature=300.0, n_steps=1000, ensemble="NVT"))

    # Optional components
    dft: Optional[DFTConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        return cls(**data)
