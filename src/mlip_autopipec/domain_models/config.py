from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DEFECT_DENSITY,
    DEFAULT_DYNAMICS_STEPS,
    DEFAULT_DYNAMICS_TEMP,
    DEFAULT_ELASTIC_TOLERANCE,
    DEFAULT_ENCUT,
    DEFAULT_EON_TIME_STEP,
    DEFAULT_HEALER_DEGAUSS,
    DEFAULT_HEALER_MIXING_BETA,
    DEFAULT_KSPACING,
    DEFAULT_LAMMPS_TEMPLATE,
    DEFAULT_LOCAL_CANDIDATES,
    DEFAULT_LOCAL_SAMPLING_METHOD,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_CYCLES,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_GAMMA,
    DEFAULT_MAX_VACUUM,
    DEFAULT_MC_SWAP_PROB,
    DEFAULT_MD_STEPS,
    DEFAULT_MIXING_BETA,
    DEFAULT_MOCK_COUNT,
    DEFAULT_MOCK_FRAMES,
    DEFAULT_N_ACTIVE_SET,
    DEFAULT_N_WORKERS,
    DEFAULT_RATIO_AB_INITIO,
    DEFAULT_SELECTION_RATIO,
    DEFAULT_SMEARING_WIDTH,
    DEFAULT_STRAIN_RANGE,
    DEFAULT_SYMMETRY_PRECISION,
    DEFAULT_TEMPERATURE_SCHEDULE,
)
from mlip_autopipec.domain_models.enums import (
    ActiveSetMethod,
    DFTCode,
    DynamicsType,
    ExecutionMode,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


class BaseComponentConfig(BaseModel):
    # Component type identifier (e.g., 'random', 'm3gnet', 'mock')
    type: str
    model_config = ConfigDict(extra="forbid")


class ExplorationPolicyConfig(BaseModel):
    strategy: GeneratorType = Field(
        default=GeneratorType.ADAPTIVE, description="Exploration strategy to use"
    )
    temperature_schedule: list[float] = Field(
        default_factory=lambda: DEFAULT_TEMPERATURE_SCHEDULE,
        description="Temperature schedule per cycle",
    )
    md_steps: int = Field(
        default=DEFAULT_MD_STEPS, ge=1, description="Number of MD steps per exploration"
    )
    mc_swap_prob: float = Field(
        default=DEFAULT_MC_SWAP_PROB, ge=0.0, le=1.0, description="Probability of MC swap"
    )
    defect_density: float = Field(
        default=DEFAULT_DEFECT_DENSITY, ge=0.0, description="Defect density"
    )
    strain_range: float = Field(
        default=DEFAULT_STRAIN_RANGE, ge=0.0, description="Strain range for random generation"
    )
    # OTF Local Generation
    n_local_candidates: int = Field(
        default=DEFAULT_LOCAL_CANDIDATES,
        ge=1,
        description="Number of local candidates to generate on halt",
    )
    local_sampling_method: str = Field(
        default=DEFAULT_LOCAL_SAMPLING_METHOD, description="Method for local candidate generation"
    )
    lammps_template: str = Field(
        default=DEFAULT_LAMMPS_TEMPLATE, description="Template for LAMMPS input script"
    )

    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseComponentConfig):
    type: GeneratorType = GeneratorType.MOCK
    ratio_ab_initio: float = Field(default=DEFAULT_RATIO_AB_INITIO, ge=0.0, le=1.0)
    mock_count: int = Field(
        default=DEFAULT_MOCK_COUNT,
        ge=1,
        description="Number of structures to generate in Mock mode",
    )
    policy: ExplorationPolicyConfig = Field(default_factory=ExplorationPolicyConfig)
    seed_structure_path: Path | None = Field(
        default=None, description="Path to initial seed structure"
    )


class OracleConfig(BaseComponentConfig):
    type: OracleType = OracleType.MOCK
    dft_code: DFTCode | None = None
    command: str | None = Field(
        default=None, description="MPI command for DFT execution. Required for DFT mode."
    )
    # DFT Parameters
    kspacing: float = Field(
        default=DEFAULT_KSPACING, gt=0.0, description="K-point spacing in inverse Angstroms"
    )
    encut: float = Field(default=DEFAULT_ENCUT, gt=0.0, description="Energy cutoff in Ry")
    pseudo_dir: Path | None = Field(default=None, description="Path to pseudopotentials directory")
    pseudos: dict[str, str] = Field(
        default_factory=dict, description="Map of element to pseudopotential filename"
    )
    mixing_beta: float = Field(
        default=DEFAULT_MIXING_BETA, gt=0.0, le=1.0, description="Mixing beta for SCF"
    )
    smearing_width: float = Field(
        default=DEFAULT_SMEARING_WIDTH, ge=0.0, description="Smearing width in Ry"
    )
    symmetry_precision: float = Field(
        default=DEFAULT_SYMMETRY_PRECISION,
        gt=0.0,
        description="Symmetry precision for spglib/Phonopy",
    )
    n_workers: int = Field(
        default=DEFAULT_N_WORKERS, ge=1, description="Number of parallel DFT calculations"
    )


class TrainerConfig(BaseComponentConfig):
    type: TrainerType = TrainerType.MOCK
    max_epochs: int = Field(default=DEFAULT_MAX_EPOCHS, ge=1)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)
    # Active Set
    active_set_method: ActiveSetMethod = Field(
        default=ActiveSetMethod.NONE, description="Method for active set selection"
    )
    selection_ratio: float = Field(
        default=DEFAULT_SELECTION_RATIO, ge=0.0, le=1.0, description="Ratio of candidates to select"
    )
    n_active_set_per_halt: int = Field(
        default=DEFAULT_N_ACTIVE_SET,
        ge=1,
        description="Number of structures to select per halt event",
    )


class DynamicsConfig(BaseComponentConfig):
    type: DynamicsType = DynamicsType.MOCK
    temperature: float = Field(default=DEFAULT_DYNAMICS_TEMP, gt=0.0)
    steps: int = Field(default=DEFAULT_DYNAMICS_STEPS, ge=1)
    # Mock specific
    mock_frames: int = Field(
        default=DEFAULT_MOCK_FRAMES, ge=1, description="Number of frames to generate in mock mode"
    )
    # Halt / OTF
    halt_on_uncertainty: bool = Field(
        default=True, description="Whether to stop on high uncertainty"
    )
    max_gamma_threshold: float = Field(
        default=DEFAULT_MAX_GAMMA, gt=0.0, description="Threshold for extrapolation grade"
    )


class ValidatorConfig(BaseComponentConfig):
    type: ValidatorType = ValidatorType.MOCK
    elastic_tolerance: float = Field(default=DEFAULT_ELASTIC_TOLERANCE, gt=0.0)
    phonon_stability: bool = True


class OrchestratorConfig(BaseModel):
    max_cycles: int = Field(
        default=DEFAULT_MAX_CYCLES, ge=1, description="Maximum number of active learning cycles"
    )
    # Remove default to satisfy "NO hardcoded paths". Must be provided in config.
    work_dir: Path = Field(..., description="Root directory for outputs")
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.MOCK, description="Mode of operation"
    )
    cleanup_on_exit: bool = Field(default=False, description="Whether to remove temporary files")
    max_candidates: int = Field(
        default=DEFAULT_MAX_CANDIDATES,
        ge=1,
        description="Maximum number of candidates to process per cycle",
    )

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    max_vacuum_size: float = Field(
        default=DEFAULT_MAX_VACUUM, description="Max vacuum size for embedding (Angstroms)"
    )
    healer_mixing_beta_target: float = Field(
        default=DEFAULT_HEALER_MIXING_BETA, description="Target mixing beta for SCF healing"
    )
    healer_degauss_target: float = Field(
        default=DEFAULT_HEALER_DEGAUSS, description="Target degauss for SCF healing (Ryd)"
    )
    default_buffer_size: int = Field(
        default=DEFAULT_BUFFER_SIZE, description="Streaming buffer size"
    )
    eon_default_time_step: float = Field(
        default=DEFAULT_EON_TIME_STEP, description="Default time step for EON (fs)"
    )
    log_file: str = Field(default=DEFAULT_LOG_FILE, description="Filename for logging")

    model_config = ConfigDict(extra="forbid")


class GlobalConfig(BaseModel):
    orchestrator: OrchestratorConfig
    system: SystemConfig = Field(default_factory=SystemConfig)
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    validator: ValidatorConfig

    model_config = ConfigDict(extra="forbid")
