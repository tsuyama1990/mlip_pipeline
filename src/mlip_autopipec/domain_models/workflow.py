from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import TaskType
from mlip_autopipec.domain_models.paths import validate_path_safety


class WorkflowState(BaseModel):
    """
    Captures the current state of the Orchestrator workflow for persistence.
    """
    current_cycle: int = Field(default=0, ge=0)
    current_step: TaskType = Field(default=TaskType.EXPLORATION)
    active_potential_path: Path | None = None
    dataset_path: Path | None = None
    iteration: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("active_potential_path", "dataset_path")
    @classmethod
    def validate_paths(cls, v: Path | None) -> Path | None:
        return validate_path_safety(v)


class ValidationResult(BaseModel):
    """
    Result of a validation run.
    """
    passed: bool = Field(..., description="Whether the validation passed")
    metrics: dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g. status details)")
    report_path: Path | None = Field(None, description="Path to detailed report")

    model_config = ConfigDict(extra="forbid")

    @field_validator("report_path")
    @classmethod
    def validate_path(cls, v: Path | None) -> Path | None:
        return validate_path_safety(v)
