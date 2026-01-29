from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowPhase(str, Enum):
    INITIALIZATION = "INITIALIZATION"
    EXPLORATION = "EXPLORATION"
    ORACLE = "ORACLE"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    SELECTION = "SELECTION"


class WorkflowState(BaseModel):
    """
    Represents the current state of the active learning workflow.
    """
    model_config = ConfigDict(extra="forbid")

    cycle_index: int = 0
    current_phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    dataset_stats: dict[str, Any] = Field(default_factory=dict)
    is_halted: bool = False
    halt_reason: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
