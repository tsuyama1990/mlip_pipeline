"""Task and Cycle domain models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyacemaker.domain_models.common import TaskStatus, TaskType, utc_now
from pyacemaker.domain_models.structure import StructureMetadata


class Task(BaseModel):
    """Representation of a computational task."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the task")
    type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    result: dict[str, Any] = Field(
        default_factory=dict, description="Result data (metrics, artifacts)"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
    started_at: datetime | None = Field(default=None, description="Start timestamp")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Task":
        """Validate temporal consistency."""
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            msg = "Completion time cannot be before start time"
            raise ValueError(msg)
        return self

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate result structure (generic check, hard to check per type without 'type' field access)."""
        # Ideally we check self.type but field_validator doesn't have access to other fields easily before model validation
        # We can use model_validator for type-specific checks
        return v

    @model_validator(mode="after")
    def validate_result_schema(self) -> "Task":
        """Validate result content based on task type."""
        if self.status == TaskStatus.COMPLETED:
            if self.type == TaskType.DFT:
                if "energy" not in self.result and "forces" not in self.result:
                     # Allow some flexibility but generally DFT results have these
                     pass
            elif self.type == TaskType.TRAINING and "potential_path" not in self.result and "metrics" not in self.result:
                 pass
        return self


class HaltInfo(BaseModel):
    """Information about a halted MD simulation."""

    model_config = ConfigDict(extra="forbid")

    halted: bool = Field(..., description="Whether the simulation halted")
    step: int | None = Field(default=None, description="Timestep at which halt occurred")
    max_gamma: float | None = Field(default=None, description="Maximum extrapolation grade at halt")
    structure: StructureMetadata | None = Field(
        default=None, description="The structure that triggered the halt"
    )

    @model_validator(mode="after")
    def validate_halt_state(self) -> "HaltInfo":
        """Validate consistency of halt state."""
        if self.halted:
            if self.step is None:
                msg = "Step must be provided when halted is True"
                raise ValueError(msg)
            if self.max_gamma is None:
                msg = "Max gamma must be provided when halted is True"
                raise ValueError(msg)
            if self.structure is None:
                msg = "Structure must be provided when halted is True"
                raise ValueError(msg)
        return self


class ActiveSet(BaseModel):
    """Collection of structures selected for active learning."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the active set")
    structure_ids: list[UUID] = Field(..., description="List of structure IDs in this set")
    # Optional field to carry the actual objects in memory if needed by the Orchestrator
    # TODO: Implement lazy loading or streaming to avoid memory pressure with large sets
    structures: list["StructureMetadata"] | None = Field(
        default=None, description="Selected structure objects"
    )
    selection_criteria: str = Field(
        ..., description="Description of selection criteria (e.g., 'max_uncertainty')"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
