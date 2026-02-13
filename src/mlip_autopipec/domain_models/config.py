from pathlib import Path
from typing import Literal

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


# Minimal safe default template
def get_default_lammps_template() -> str:
    return """
# Default LAMMPS input script (Minimal)
# Please configure 'lammps_template' in GeneratorConfig.
"""


class BaseComponentConfig(BaseModel):
    """Base configuration for pipeline components."""
    # Subclasses must define specific type Enum
    model_config = ConfigDict(extra="forbid")


class ExplorationPolicyConfig(BaseModel):
    """Configuration for adaptive exploration policy."""
    strategy: GeneratorType = Field(
        default=GeneratorType.ADAPTIVE, description="Exploration strategy to use"
    )
    temperature_schedule: list[float] = Field(
        default_factory=lambda: [300.0, 600.0, 1200.0],
        description="Temperature schedule per cycle (Kelvin)",
    )
    md_steps: int = Field(default=1000, ge=1, description="Number of MD steps per exploration")
    mc_swap_prob: float = Field(default=0.1, ge=0.0, le=1.0, description="Probability of MC swap")
    defect_density: float = Field(default=0.0, ge=0.0, description="Defect density per supercell")
    strain_range: float = Field(
        default=0.05, ge=0.0, description="Strain range (fraction) for random generation"
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
    """Configuration for Structure Generator."""
    type: GeneratorType = GeneratorType.MOCK
    ratio_ab_initio: float = Field(default=0.1, ge=0.0, le=1.0, description="Ratio of ab initio structures")
    mock_count: int = Field(
        default=2,
        ge=1,
        description="Number of structures to generate in Mock mode",
    )
    policy: ExplorationPolicyConfig = Field(default_factory=ExplorationPolicyConfig)
    seed_structure_path: Path | None = Field(
        default=None, description="Path to initial seed structure file"
    )


class OracleConfig(BaseComponentConfig):
    """Configuration for Oracle (DFT Engine)."""
    type: OracleType = OracleType.MOCK
    dft_code: DFTCode | None = Field(default=None, description="DFT code to use")
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
    """Configuration for Potential Trainer."""
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
    cutoff: float = Field(default=5.0, gt=0.0, description="Radial cutoff (Angstrom)")
    order: int = Field(default=2, ge=1, description="Body order")
    basis_size: int = Field(default=500, ge=1, description="Number of basis functions")
    delta_learning: str | None = Field(
        default=None, description="Delta learning baseline (e.g., 'zbl', 'lj')"
    )


class DynamicsConfig(BaseComponentConfig):
    """Configuration for Dynamics Engine."""
    type: DynamicsType = DynamicsType.MOCK
    temperature: float = Field(default=300.0, gt=0.0, description="Simulation temperature (K)")
    steps: int = Field(default=1000, ge=1, description="Number of steps")
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
    """Configuration for Validator."""
    type: ValidatorType = ValidatorType.MOCK
    elastic_tolerance: float = Field(default=0.15, gt=0.0, description="Tolerance for elastic constants")
    phonon_stability: bool = Field(default=True, description="Check phonon stability")


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator."""
    max_cycles: int = Field(default=1, ge=1, description="Maximum number of active learning cycles")
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
    """System-wide configuration."""
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


class ActiveLearningConfig(BaseModel):
    """Configuration for Active Learning (Local Learning Loop)."""
    perturbation_magnitude: float = Field(
        default=0.1, gt=0.0, description="Magnitude of displacement for local candidates (Angstrom)"
    )
    n_candidates: int = Field(
        default=20, ge=1, description="Number of candidates to generate per halt"
    )
    sampling_method: Literal["perturbation", "md_burst"] = Field(
        default="perturbation", description="Method for local candidate generation"
    )
    max_retries: int = Field(
        default=3, ge=0, description="Max retries for the local loop if validation fails"
    )

    model_config = ConfigDict(extra="forbid")


class GlobalConfig(BaseModel):
    """Global Configuration Root."""
    orchestrator: OrchestratorConfig
    system: SystemConfig = Field(default_factory=SystemConfig)
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    validator: ValidatorConfig
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)

    model_config = ConfigDict(extra="forbid")
