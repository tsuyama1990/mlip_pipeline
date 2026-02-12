from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.enums import (
    DFTCode,
    DynamicsType,
    ExecutionMode,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


class BaseComponentConfig(BaseModel):
    type: str
    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseComponentConfig):
    type: GeneratorType = GeneratorType.MOCK
    # Add specific fields for other types later
    ratio_ab_initio: float = Field(default=0.1, ge=0.0, le=1.0)
    mock_count: int = Field(default=2, ge=1, description="Number of structures to generate in Mock mode")


class OracleConfig(BaseComponentConfig):
    type: OracleType = OracleType.MOCK
    dft_code: DFTCode | None = None
    command: str = "mpirun -np 4 pw.x"


class TrainerConfig(BaseComponentConfig):
    type: TrainerType = TrainerType.MOCK
    max_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)


class DynamicsConfig(BaseComponentConfig):
    type: DynamicsType = DynamicsType.MOCK
    temperature: float = Field(default=300.0, gt=0.0)
    steps: int = Field(default=1000, ge=1)


class ValidatorConfig(BaseComponentConfig):
    type: ValidatorType = ValidatorType.MOCK
    elastic_tolerance: float = Field(default=0.15, gt=0.0)
    phonon_stability: bool = True


class OrchestratorConfig(BaseModel):
    max_cycles: int = Field(default=1, ge=1, description="Maximum number of active learning cycles")
    work_dir: Path = Field(default=Path("./experiments"), description="Root directory for outputs")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.MOCK, description="Mode of operation")
    cleanup_on_exit: bool = Field(default=False, description="Whether to remove temporary files")

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    max_vacuum_size: float = Field(default=50.0, description="Max vacuum size for embedding (Angstroms)")
    healer_mixing_beta_target: float = Field(default=0.3, description="Target mixing beta for SCF healing")
    healer_degauss_target: float = Field(default=0.02, description="Target degauss for SCF healing (Ryd)")
    default_buffer_size: int = Field(default=1000, description="Streaming buffer size")
    eon_default_time_step: float = Field(default=1.0, description="Default time step for EON (fs)")

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
