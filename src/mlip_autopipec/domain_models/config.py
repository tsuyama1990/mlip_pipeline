from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.enums import (
    ActiveSetMethod,
    DFTCode,
    DynamicsType,
    ExecutionMode,
    GeneratorType,
    HybridPotentialType,
    OracleType,
    TrainerType,
    ValidatorType,
)


# Lazy loader for LAMMPS template
def get_default_lammps_template() -> str:
    template_path = Path(__file__).parents[1] / "templates" / "default_lammps.in"
    if template_path.exists():
        return template_path.read_text()
    return ""


class BaseComponentConfig(BaseModel):
    # Component type identifier (e.g., 'random', 'm3gnet', 'mock')
    # Allow Any to support Enums in subclasses
    type: Any
    model_config = ConfigDict(extra="forbid")


class ExplorationPolicyConfig(BaseModel):
    strategy: GeneratorType = Field(
        default=GeneratorType.ADAPTIVE, description="Exploration strategy to use"
    )
    temperature_schedule: list[float] = Field(
        default_factory=lambda: [300.0, 600.0, 1200.0],
        description="Temperature schedule per cycle",
    )
    md_steps: int = Field(default=1000, ge=1, description="Number of MD steps per exploration")
    mc_swap_prob: float = Field(default=0.1, ge=0.0, le=1.0, description="Probability of MC swap")
    defect_density: float = Field(default=0.0, ge=0.0, description="Defect density")
    strain_range: float = Field(
        default=0.05, ge=0.0, description="Strain range for random generation"
    )
    # OTF Local Generation
    n_local_candidates: int = Field(
        default=20,
        ge=1,
        description="Number of local candidates to generate on halt",
    )
    local_sampling_method: str = Field(
        default="perturbation", description="Method for local candidate generation"
    )
    lammps_template: str = Field(
        default_factory=get_default_lammps_template, description="Template for LAMMPS input script"
    )

    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseComponentConfig):
    type: GeneratorType = GeneratorType.MOCK
    ratio_ab_initio: float = Field(default=0.1, ge=0.0, le=1.0)
    mock_count: int = Field(
        default=2,
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
        default=0.04, gt=0.0, description="K-point spacing in inverse Angstroms"
    )
    encut: float = Field(default=60.0, gt=0.0, description="Energy cutoff in Ry")
    pseudo_dir: Path | None = Field(default=None, description="Path to pseudopotentials directory")
    pseudos: dict[str, str] = Field(
        default_factory=dict, description="Map of element to pseudopotential filename"
    )
    mixing_beta: float = Field(default=0.7, gt=0.0, le=1.0, description="Mixing beta for SCF")
    smearing_width: float = Field(default=0.01, ge=0.0, description="Smearing width in Ry")
    symmetry_precision: float = Field(
        default=1e-5,
        gt=0.0,
        description="Symmetry precision for spglib/Phonopy",
    )
    n_workers: int = Field(default=1, ge=1, description="Number of parallel DFT calculations")


class TrainerConfig(BaseComponentConfig):
    type: TrainerType = TrainerType.MOCK
    max_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    mock_potential_content: str = Field(
        default="MOCK POTENTIAL FILE CONTENT", description="Content for mock potential file"
    )
    # Active Set
    active_set_method: ActiveSetMethod = Field(
        default=ActiveSetMethod.NONE, description="Method for active set selection"
    )
    selection_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Ratio of candidates to select"
    )
    n_active_set_per_halt: int = Field(
        default=5,
        ge=1,
        description="Number of structures to select per halt event",
    )
    # Pacemaker parameters
    cutoff: float = Field(default=5.0, gt=0.0, description="Radial cutoff")
    order: int = Field(default=2, ge=1, description="Body order")
    basis_size: int = Field(default=500, ge=1, description="Number of basis functions")
    delta_learning: str | None = Field(
        default=None, description="Delta learning baseline (e.g., 'zbl', 'lj')"
    )


class DynamicsConfig(BaseComponentConfig):
    type: DynamicsType = DynamicsType.MOCK
    temperature: float = Field(default=300.0, gt=0.0)
    steps: int = Field(default=1000, ge=1)
    timestep: float = Field(default=0.001, gt=0.0, description="MD timestep in ps")
    n_thermo: int = Field(default=10, ge=1, description="Thermodynamic output interval")
    n_dump: int = Field(default=100, ge=1, description="Trajectory dump interval")
    max_frames: int = Field(default=1000, ge=1, description="Max frames to read per simulation")
    lammps_command: str = Field(default="lmp", description="LAMMPS executable command")

    # Hybrid Potential (Baseline)
    hybrid_potential: HybridPotentialType | None = Field(
        default=None, description="Baseline potential for hybrid/overlay"
    )
    zbl_cut_inner: float = Field(default=0.5, gt=0.0, description="ZBL inner cutoff")
    zbl_cut_outer: float = Field(default=1.2, gt=0.0, description="ZBL outer cutoff")
    lj_epsilon: float = Field(default=1.0, gt=0.0, description="LJ epsilon")
    lj_sigma: float = Field(default=1.0, gt=0.0, description="LJ sigma")
    lj_cutoff: float = Field(default=2.5, gt=0.0, description="LJ cutoff")

    # Mock specific
    mock_frames: int = Field(
        default=5, ge=1, description="Number of frames to generate in mock mode"
    )
    # Halt / OTF
    halt_on_uncertainty: bool = Field(
        default=True, description="Whether to stop on high uncertainty"
    )
    max_gamma_threshold: float = Field(
        default=5.0, gt=0.0, description="Threshold for extrapolation grade"
    )


class ValidatorConfig(BaseComponentConfig):
    type: ValidatorType = ValidatorType.MOCK
    elastic_tolerance: float = Field(default=0.15, gt=0.0)
    phonon_stability: bool = True


class OrchestratorConfig(BaseModel):
    max_cycles: int = Field(default=1, ge=1, description="Maximum number of active learning cycles")
    # Remove default to satisfy "NO hardcoded paths". Must be provided in config.
    work_dir: Path = Field(..., description="Root directory for outputs")
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.MOCK, description="Mode of operation"
    )
    cleanup_on_exit: bool = Field(default=False, description="Whether to remove temporary files")
    max_candidates: int = Field(
        default=50,
        ge=1,
        description="Maximum number of candidates to process per cycle",
    )

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    max_vacuum_size: float = Field(
        default=50.0, description="Max vacuum size for embedding (Angstroms)"
    )
    healer_mixing_beta_target: float = Field(
        default=0.3, description="Target mixing beta for SCF healing"
    )
    healer_degauss_target: float = Field(
        default=0.02, description="Target degauss for SCF healing (Ryd)"
    )
    default_buffer_size: int = Field(default=1000, description="Streaming buffer size")
    eon_default_time_step: float = Field(default=1.0, description="Default time step for EON (fs)")
    log_file: str = Field(default="mlip_pipeline.log", description="Filename for logging")

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
