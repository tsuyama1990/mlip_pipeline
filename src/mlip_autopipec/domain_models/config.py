from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.constants import (
    DEFAULT_BUFFER_SIZE,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


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

    @field_validator("policy_ratios")
    @classmethod
    def validate_ratios(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            msg = f"Policy ratios must sum to 1.0, got {total}"
            raise ValueError(msg)
        return v


GeneratorConfig = MockGeneratorConfig | AdaptiveGeneratorConfig


# --- Oracle Configs ---


class BaseOracleConfig(ComponentConfig):
    pass


class MockOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.MOCK] = OracleType.MOCK


class QEOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.QE] = OracleType.QE
    kspacing: float = Field(default=0.05, ge=0.01, le=2.0)
    mixing_beta: float = Field(default=0.7, ge=0.0, le=1.0)
    smearing: str = "mv"
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    # Require explicit cutoffs to avoid magic numbers
    ecutwfc: float = Field(..., gt=0)
    ecutrho: float = Field(..., gt=0)
    batch_size: int = Field(default=10, gt=0, le=1000)
    max_workers: int = Field(default=4, gt=0)

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        import os

        cpu_count = os.cpu_count() or 1
        if v > cpu_count * 2:
            msg = f"max_workers {v} seems too high for {cpu_count} CPUs"
            raise ValueError(msg)
        return v


class VASPOracleConfig(BaseOracleConfig):
    name: Literal[OracleType.VASP] = OracleType.VASP
    encut: float = 520.0
    kspacing: float = Field(default=0.05, ge=0.01, le=2.0)
    ediff: float = 1e-6


OracleConfig = MockOracleConfig | QEOracleConfig | VASPOracleConfig


# --- Physics Baseline Config ---


class PhysicsBaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lj", "zbl"]
    params: dict[str, float] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<PhysicsBaselineConfig(type={self.type})>"

    def __str__(self) -> str:
        return f"PhysicsBaselineConfig(type={self.type})"


class PacemakerInputConfig(BaseModel):
    """
    Configuration model for Pacemaker input.yaml file.
    Only includes fields we control dynamically.
    """

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields to support custom Pacemaker options

    cutoff: float
    data: dict[str, str]
    fitting: dict[str, float]
    backend: dict[str, str]
    b_basis: dict[str, int]
    physics_baseline: dict[str, Any] | None = None
    initial_potential: str | None = None

    def __repr__(self) -> str:
        return f"<PacemakerInputConfig(cutoff={self.cutoff})>"

    def __str__(self) -> str:
        return f"PacemakerInputConfig(cutoff={self.cutoff})"


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
    fitting_weight_energy: float = 1.0
    fitting_weight_force: float = 1.0
    backend_evaluator: str = "tensorpot"
    active_set_selection: bool = True
    active_set_limit: int = 1000
    initial_potential: str | Path | None = None
    physics_baseline: PhysicsBaselineConfig | None = None

    # Defaults defined directly here for isolation
    input_filename: str = "input.yaml"
    dataset_filename: str = "dataset.pckl.gzip"
    potential_filename: str = "output_potential.yace"
    activeset_filename: str = "dataset_activeset.pckl.gzip"

    data_format: Literal["extxyz", "pckl.gzip"] = "extxyz"
    pacemaker_options: dict[str, Any] = Field(default_factory=dict)


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


class EONDynamicsConfig(BaseDynamicsConfig):
    name: Literal[DynamicsType.EON] = DynamicsType.EON
    temperature: float = 300.0
    time_step: float = (
        1.0  # Time per step in seconds (approx) or similar EON param? EON calculates time.
    )
    # EON usually runs for a number of events or time.
    n_events: int = 1000
    supercell: list[int] = Field(default_factory=lambda: [1, 1, 1])
    # EON specific params
    prefactor: float = 1e12
    seed: int | None = None


DynamicsConfig = MockDynamicsConfig | LAMMPSDynamicsConfig | EONDynamicsConfig


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

    def __repr__(self) -> str:
        return "<ComponentsConfig>"

    def __str__(self) -> str:
        return "ComponentsConfig"


class OrchestratorConfig(BaseModel):
    """Configuration for Orchestrator paths and behavior."""

    model_config = ConfigDict(extra="forbid")

    dataset_filename: str = "dataset.jsonl"
    state_filename: str = "workflow_state.json"
    cycle_dir_pattern: str = "cycle_{cycle:02d}"
    potential_filename: str = "potential.yace"
    default_buffer_size: int = Field(default=DEFAULT_BUFFER_SIZE)

    def __repr__(self) -> str:
        return "<OrchestratorConfig>"

    def __str__(self) -> str:
        return "OrchestratorConfig"


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(gt=0)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    physics_baseline: PhysicsBaselineConfig | None = None

    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
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

    def __repr__(self) -> str:
        return f"<GlobalConfig(workdir={self.workdir})>"

    def __str__(self) -> str:
        return f"GlobalConfig(workdir={self.workdir})"
