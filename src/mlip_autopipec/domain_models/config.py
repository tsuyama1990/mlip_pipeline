from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)

# Constants for Structure validation
MAX_ATOMIC_NUMBER = 118
MAX_FORCE_MAGNITUDE = 1000.0  # eV/A
MAX_ENERGY_MAGNITUDE = 1e6  # eV

# Constants for Dataset
DEFAULT_BUFFER_SIZE = 1000

# Constants for Oracle
MAX_VACUUM_SIZE = 50.0  # Angstroms
HEALER_MIXING_BETA_TARGET = 0.3
HEALER_DEGAUSS_TARGET = 0.02


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


# --- Generator Configs ---


class BaseGeneratorConfig(ComponentConfig):
    n_structures: int = Field(default=10, gt=0)


class MockGeneratorConfig(BaseGeneratorConfig):
    name: Literal[GeneratorType.MOCK] = GeneratorType.MOCK
    cell_size: float = Field(..., gt=0)
    n_atoms: int = Field(..., gt=0)
    atomic_numbers: list[int] = Field(..., min_length=1)


class AdaptiveGeneratorConfig(BaseGeneratorConfig):
    name: Literal[GeneratorType.ADAPTIVE] = GeneratorType.ADAPTIVE
    element: str
    crystal_structure: str
    strain_range: float = 0.05
    rattle_strength: float = 0.01
    surface_indices: list[list[int]] = Field(
        default_factory=lambda: [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    )
    vacuum: float = 10.0
    supercell_dim: int = 2
    policy_ratios: dict[str, float] = Field(
        default_factory=lambda: {"cycle0_bulk": 0.6, "cycle0_surface": 0.4}
    )


GeneratorConfig = MockGeneratorConfig | AdaptiveGeneratorConfig


# --- Oracle Configs ---


class BaseOracleConfig(ComponentConfig):
    pass


class MockOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.MOCK] = OracleType.MOCK


class QEOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.QE] = OracleType.QE
    kspacing: float = Field(default=0.05, ge=0.01, le=2.0)
    mixing_beta: float = 0.7
    smearing: str = "mv"
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    ecutwfc: float = 60.0
    ecutrho: float = 360.0
    batch_size: int = Field(default=10, gt=0)  # Configurable batch size


class VASPOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.VASP] = OracleType.VASP
    encut: float = 520.0
    kspacing: float = Field(default=0.05, ge=0.01, le=2.0)
    ediff: float = 1e-6


OracleConfig = MockOracleConfig | QEOracleConfig | VASPOracleConfig


# --- Trainer Configs ---


class BaseTrainerConfig(ComponentConfig):
    max_num_epochs: int = 50


class MockTrainerConfig(BaseTrainerConfig):
    name: Literal[TrainerType.MOCK] = TrainerType.MOCK
    energy_rmse_threshold: float | None = None
    force_rmse_threshold: float | None = None


class PacemakerTrainerConfig(BaseTrainerConfig):
    name: Literal[TrainerType.PACEMAKER] = TrainerType.PACEMAKER
    basis_size: int = 1000
    cutoff: float = 5.0
    regularization: list[float] = Field(default_factory=lambda: [1e-3, 1e-4])
    ladder_step: list[int] = Field(default_factory=lambda: [1, 2, 3])
    batch_size: int = 32


TrainerConfig = MockTrainerConfig | PacemakerTrainerConfig


# --- Dynamics Configs ---


class BaseDynamicsConfig(ComponentConfig):
    uncertainty_threshold: float = 5.0


class MockDynamicsConfig(BaseDynamicsConfig):
    name: Literal[DynamicsType.MOCK] = DynamicsType.MOCK
    selection_rate: float = Field(..., ge=0.0, le=1.0)
    seed: int | None = None
    simulated_uncertainty: float = 1.0


class LAMMPSDynamicsConfig(BaseDynamicsConfig):
    name: Literal[DynamicsType.LAMMPS] = DynamicsType.LAMMPS
    timestep: float = 0.001
    n_steps: int = 10000
    temperature: float = 300.0
    pressure: float = 0.0
    thermo_freq: int = 100


DynamicsConfig = MockDynamicsConfig | LAMMPSDynamicsConfig


# --- Validator Configs ---


class BaseValidatorConfig(ComponentConfig):
    test_set_ratio: float = 0.1


class MockValidatorConfig(BaseValidatorConfig):
    name: Literal[ValidatorType.MOCK] = ValidatorType.MOCK


class StandardValidatorConfig(BaseValidatorConfig):
    name: Literal[ValidatorType.STANDARD] = ValidatorType.STANDARD
    phonon_supercell: list[int] = Field(default_factory=lambda: [4, 4, 4])
    eos_strain_range: float = 0.2
    elastic_strain_magnitude: float = 0.01


ValidatorConfig = MockValidatorConfig | StandardValidatorConfig


# --- Global Config ---


class ComponentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: GeneratorConfig = Field(discriminator="name")
    oracle: OracleConfig = Field(discriminator="name")
    trainer: TrainerConfig = Field(discriminator="name")
    dynamics: DynamicsConfig = Field(discriminator="name")
    validator: ValidatorConfig = Field(discriminator="name")


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(gt=0)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    components: ComponentsConfig

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, v: Path) -> Path:
        """Ensure workdir is a valid path and its parent exists."""
        try:
            resolved = v.resolve()
            if not resolved.parent.exists():
                msg = f"Workdir parent directory does not exist: {resolved.parent}"
                raise ValueError(msg)
        except OSError as e:
            msg = f"Invalid workdir path: {v}"
            raise ValueError(msg) from e
        return v
