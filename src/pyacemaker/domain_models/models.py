"""Domain models for PYACEMAKER."""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class StructureStatus(str, Enum):
    """Status of a structure in the active learning cycle."""

    NEW = "NEW"
    CALCULATED = "CALCULATED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class StructureMetadata(BaseModel):
    """Metadata for a structure."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the structure")
    features: dict[str, Any] = Field(
        default_factory=dict, description="Extracted features (e.g., composition)"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags (e.g., 'initial', 'high_uncertainty')"
    )
    status: StructureStatus = Field(
        default=StructureStatus.NEW, description="Processing status of the structure"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=utc_now, description="Last update timestamp")


class PotentialType(str, Enum):
    """Type of interatomic potential."""

    PACE = "PACE"
    M3GNET = "M3GNET"
    LJ = "LJ"
    EAM = "EAM"


class Potential(BaseModel):
    """Representation of an interatomic potential."""

    model_config = ConfigDict(extra="forbid")

    path: Path = Field(..., description="Path to the potential file")
    type: PotentialType = Field(..., description="Type of the potential")
    version: str = Field(..., description="Version identifier")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Validation metrics (RMSE, etc.)"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")


class TaskType(str, Enum):
    """Type of computational task."""

    DFT = "DFT"
    TRAINING = "TRAINING"
    MD = "MD"
    KMC = "KMC"


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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


class CycleStatus(str, Enum):
    """Status of the active learning cycle."""

    EXPLORATION = "EXPLORATION"
    LABELING = "LABELING"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    CONVERGED = "CONVERGED"
    FAILED = "FAILED"


class ActiveSet(BaseModel):
    """Collection of structures selected for active learning."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the active set")
    structure_ids: list[UUID] = Field(..., description="List of structure IDs in this set")
    selection_criteria: str = Field(
        ..., description="Description of selection criteria (e.g., 'max_uncertainty')"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
