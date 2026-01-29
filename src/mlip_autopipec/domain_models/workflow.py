"""Domain models for workflow state management."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Candidate


class WorkflowPhase(str, Enum):
    """Enumeration of workflow phases."""
    INITIALIZATION = "INITIALIZATION"
    EXPLORATION = "EXPLORATION"
    SELECTION = "SELECTION"
    ORACLE = "ORACLE"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    COMPLETED = "COMPLETED"


class WorkflowState(BaseModel):
    """Represents the current state of the active learning loop."""

    model_config = ConfigDict(extra="forbid")

    cycle_index: int = Field(default=0, ge=0, description="Current active learning cycle index")
    current_phase: WorkflowPhase = Field(default=WorkflowPhase.INITIALIZATION)
    dataset_stats: dict[str, Any] = Field(default_factory=dict, description="Statistics of the dataset")
    candidates: list[Candidate] = Field(default_factory=list, description="Generated candidates for the current cycle")
    is_halted: bool = Field(default=False, description="Whether the workflow is halted (e.g. error)")
    halt_reason: str = Field(default="", description="Reason for halt")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")
