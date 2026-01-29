from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WorkflowPhase(str, Enum):
    EXPLORATION = "EXPLORATION"
    SELECTION = "SELECTION"
    CALCULATION = "CALCULATION"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    COMPLETED = "COMPLETED"


class WorkflowState(BaseModel):
    """
    Tracks the state of the Active Learning Cycle.
    """
    model_config = ConfigDict(extra="forbid")

    cycle_index: int = 0
    current_phase: WorkflowPhase = WorkflowPhase.EXPLORATION

    # Paths to critical artifacts
    dataset_path: Path | None = None
    current_potential_path: Path | None = None

    # Status flags
    is_halted: bool = False
    halt_reason: str | None = None

    # Metadata for the current cycle
    meta: dict[str, str | int | float | bool] = Field(default_factory=dict)
