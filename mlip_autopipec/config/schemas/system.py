from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import MinimalConfig, TargetSystem
from .dft import DFTConfig
from .exploration import ExplorerConfig, ExplorerParams, GeneratorParams
from .generator import GeneratorConfig
from .inference import InferenceConfig
from .training import TrainingConfig, TrainingRunMetrics


class WorkflowConfig(BaseModel):
    checkpoint_file_path: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")


class DaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(None)


class SystemConfig(BaseModel):
    """
    Comprehensive configuration for the MLIP-AutoPipe system.
    Cycle 01 strict schema + Optional future fields for compatibility.
    """

    minimal: MinimalConfig
    working_dir: Path
    db_path: Path
    log_path: Path

    # Optional Fields for future Cycles
    project_name: str | None = None  # Duplicate of minimal.project_name but kept for compatibility
    run_uuid: UUID | None = None
    workflow_config: WorkflowConfig | None = None

    # Optional Module Configurations
    target_system: TargetSystem | None = None  # Duplicate of minimal.target_system
    dft_config: DFTConfig | None = None
    explorer_config: ExplorerConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None
    generator_config: GeneratorConfig | None = None

    # Legacy / Transition Fields (Deprecated)
    generator: GeneratorParams | None = None
    explorer: ExplorerParams | None = None
    dask: DaskConfig | None = None
    dft: DFTConfig | None = None  # Legacy alias

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: Path) -> Path:
        # Check if absolute path or safe relative path logic if needed
        # Cycle 01 spec ensures paths are absolute from factory.
        return v

    @model_validator(mode="before")
    @classmethod
    def populate_fields_from_minimal(cls, data: Any) -> Any:
        if isinstance(data, dict):
            minimal = data.get("minimal")
            if minimal:
                # If minimal is a dict
                if isinstance(minimal, dict):
                    if data.get("target_system") is None:
                        data["target_system"] = minimal.get("target_system")
                    if data.get("project_name") is None:
                        data["project_name"] = minimal.get("project_name")
                # If minimal is an object (MinimalConfig)
                elif hasattr(minimal, "target_system"):
                    if data.get("target_system") is None:
                        data["target_system"] = minimal.target_system
                    if data.get("project_name") is None:
                        data["project_name"] = minimal.project_name
        return data


class CheckpointState(BaseModel):
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
