from pathlib import Path
from typing import Literal, Optional, Union, List

import ase.data
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.dynamics import LammpsConfig, MDConfig, EonConfig
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec import defaults


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = defaults.DEFAULT_LOG_LEVEL  # type: ignore
    file_path: Path = Path(defaults.DEFAULT_LOG_FILENAME)


class ACEConfig(BaseModel):
    """Configuration for ACE (Pacemaker) basis set."""
    model_config = ConfigDict(extra="forbid")

    npot: str = Field(default=defaults.DEFAULT_ACE_NPOT, description="Potential type (e.g. FinnisSinclair)")
    fs_parameters: List[float] = Field(default_factory=lambda: defaults.DEFAULT_ACE_FS_PARAMS, description="Finnis-Sinclair parameters")
    ndensity: int = Field(default=defaults.DEFAULT_ACE_NDENSITY, description="Density of basis functions")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(default_factory=lambda: defaults.DEFAULT_ELEMENTS)
    cutoff: float = defaults.DEFAULT_CUTOFF
    seed: int = defaults.DEFAULT_SEED
    lammps_command: str = defaults.DEFAULT_LAMMPS_COMMAND

    # Hybrid Potential Settings (ACE + ZBL)
    pair_style: Literal["pace", "hybrid/overlay"] = defaults.DEFAULT_PAIR_STYLE  # type: ignore
    zbl_inner_cutoff: float = defaults.DEFAULT_ZBL_INNER
    zbl_outer_cutoff: float = defaults.DEFAULT_ZBL_OUTER

    # Pacemaker / ACE Basis Parameters
    ace_params: ACEConfig = Field(default_factory=ACEConfig, description="ACE basis parameters")

    # Fitting Parameters (Moved from hardcoded values)
    delta_spline_bins: float = defaults.DEFAULT_PACEMAKER_DELTA_SPLINE_BINS
    loss_weight_energy: float = defaults.DEFAULT_PACEMAKER_LOSS_WEIGHT_ENERGY
    loss_weight_forces: float = defaults.DEFAULT_PACEMAKER_LOSS_WEIGHT_FORCES
    loss_weight_stress: float = defaults.DEFAULT_PACEMAKER_LOSS_WEIGHT_STRESS

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "PotentialConfig":
        """
        Validate the configuration for physical consistency.
        """
        self._validate_elements()
        self._validate_hybrid_style()
        return self

    def _validate_elements(self):
        """Internal helper to validate elements list."""
        if not self.elements:
            raise ValueError("Elements list cannot be empty.")

        for el in self.elements:
            if el not in ase.data.atomic_numbers:
                raise ValueError(f"Invalid element symbol: {el}")

    def _validate_hybrid_style(self):
        """Internal helper to validate hybrid style requirements."""
        has_heavy_atoms = any(ase.data.atomic_numbers[el] >= 2 for el in self.elements)

        if has_heavy_atoms and self.pair_style != "hybrid/overlay":
             raise ValueError(
                 "Elements with Z >= 2 found. You MUST use 'hybrid/overlay' pair_style "
                 "to enforce physical core repulsion (Delta Learning). "
                 "See SPEC.md Section 3.3."
             )

        if self.pair_style == "hybrid/overlay":
            if self.zbl_inner_cutoff >= self.zbl_outer_cutoff:
                raise ValueError(
                    f"zbl_inner_cutoff ({self.zbl_inner_cutoff}) must be less than "
                    f"zbl_outer_cutoff ({self.zbl_outer_cutoff})."
                )
            if self.zbl_inner_cutoff <= 0:
                raise ValueError("zbl_inner_cutoff must be positive.")


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = defaults.DEFAULT_MAX_ITERATIONS
    uncertainty_threshold: float = defaults.DEFAULT_UNCERTAINTY_THRESHOLD
    halt_threshold: int = defaults.DEFAULT_HALT_THRESHOLD
    validation_frequency: int = defaults.DEFAULT_VALIDATION_FREQUENCY

    # Sampling & Batching
    trajectory_sampling_stride: int = defaults.DEFAULT_TRAJECTORY_STRIDE
    dft_batch_size: int = defaults.DEFAULT_DFT_BATCH_SIZE

    # Paths
    data_dir: Path = Path(defaults.DEFAULT_DATA_DIR)

    # File Naming Conventions
    trajectory_file_lammps: str = defaults.DEFAULT_TRAJ_FILE_LAMMPS
    trajectory_file_extxyz: str = defaults.DEFAULT_TRAJ_FILE_EXTXYZ
    lammps_log_file: str = defaults.DEFAULT_LOG_FILE_LAMMPS
    lammps_input_file: str = defaults.DEFAULT_INPUT_FILE_LAMMPS
    lammps_data_file: str = defaults.DEFAULT_DATA_FILE_LAMMPS
    stdout_log_file: str = defaults.DEFAULT_STDOUT_FILE
    stderr_log_file: str = defaults.DEFAULT_STDERR_FILE

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
    element: str = defaults.DEFAULT_BULK_ELEMENT
    crystal_structure: str = defaults.DEFAULT_BULK_CRYSTAL
    lattice_constant: float = defaults.DEFAULT_BULK_LATTICE
    rattle_stdev: float = defaults.DEFAULT_BULK_RATTLE
    supercell: tuple[int, int, int] = defaults.DEFAULT_BULK_SUPERCELL


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

    # Production
    package_name_format: str = "mlip_package_{version}.zip"

    # Phonon
    phonon_tolerance: float = defaults.DEFAULT_PHONON_TOL  # THz, strictly negative to allow numeric noise
    phonon_supercell: tuple[int, int, int] = defaults.DEFAULT_PHONON_SUPERCELL

    # Elastic
    elastic_stability_tolerance: float = defaults.DEFAULT_ELASTIC_TOL  # GPa

    # EOS
    eos_vol_range: float = defaults.DEFAULT_EOS_RANGE  # +/- 10%
    eos_n_points: int = defaults.DEFAULT_EOS_POINTS

    # Reporting
    report_path: Path = Path(defaults.DEFAULT_REPORT_PATH)
    template_dir: Optional[Path] = None

    # Structure Prep
    validation_rattle_stdev: float = defaults.DEFAULT_VAL_RATTLE

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

    project_name: str = defaults.DEFAULT_PROJECT_NAME
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig = Field(default_factory=PotentialConfig)
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    eon: EonConfig = Field(default_factory=EonConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    # New configurations
    structure_gen: StructureGenConfig = Field(default_factory=BulkStructureGenConfig, discriminator="strategy")
    md: MDConfig = Field(default_factory=lambda: MDConfig(temperature=defaults.DEFAULT_MD_TEMP, n_steps=defaults.DEFAULT_MD_STEPS, ensemble=defaults.DEFAULT_MD_ENSEMBLE))  # type: ignore

    # Optional components
    dft: Optional[DFTConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        # Pydantic validation happens here automatically
        return cls(**data)
