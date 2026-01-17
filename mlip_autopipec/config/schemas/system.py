"""
System-level configuration schemas.
These models represent the fully hydrated state of the application.
"""
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import MinimalConfig
from .dft import DFTConfig
from .exploration import ExplorerConfig, ExplorerParams, GeneratorParams
from .inference import InferenceConfig
from .training import TrainingConfig, TrainingRunMetrics


class WorkflowConfig(BaseModel):
    """
    Configuration for workflow execution parameters.
    """
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")

class DaskConfig(BaseModel):
    """
    Configuration for Dask distributed execution.
    """
    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(None)

class SystemConfig(BaseModel):
    """
    Comprehensive configuration for the MLIP-AutoPipe system.
    This object is created by the ConfigFactory and serves as the single source of truth.
    It contains the user input (MinimalConfig) plus resolved runtime paths.
    """
    minimal: MinimalConfig
    working_dir: Path
    db_path: Path
    log_path: Path

    # Future fields (Cycle 2+)
    # dft_config: DFTConfig | None = None
    # explorer_config: ExplorerConfig | None = None
    # training_config: TrainingConfig | None = None
    # inference_config: InferenceConfig | None = None

    model_config = ConfigDict(frozen=True)

class CheckpointState(BaseModel):
    """
    Represents the state of the workflow for checkpointing and recovery.
    """
    run_uuid: UUID
    system_config: SystemConfig
    active_learning_generation: int = Field(0, ge=0)
    current_potential_path: Path | None = None
    pending_job_ids: list[UUID] = Field(default_factory=list)
    # job_submission_args stores arguments like (atoms,), where atoms is an ase.Atoms object.
    # ase.Atoms is not pydantic-validatable easily, so we allow arbitrary types.
    job_submission_args: dict[UUID, Any] = Field(default_factory=dict)
    training_history: list[TrainingRunMetrics] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

class CalculationMetadata(BaseModel):
    """
    Metadata attached to calculations in the database.
    """
    model_config = ConfigDict(extra="forbid")
    stage: str
    uuid: str
    force_mask: list[list[float]] | None = None
