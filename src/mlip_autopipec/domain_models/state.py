from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WorkflowPhase(str, Enum):
    """
    Enum representing the current phase of the active learning workflow.
    """

    EXPLORATION = "Exploration"
    SELECTION = "Selection"
    CALCULATION = "Calculation"
    TRAINING = "Training"


class WorkflowState(BaseModel):
    """
    Tracks the current state of the automated workflow.
    """

    cycle_index: int = Field(
        default=0, description="The current active learning cycle index (0-based).", ge=0
    )
    current_phase: WorkflowPhase = Field(
        default=WorkflowPhase.EXPLORATION, description="Current phase of the workflow."
    )
    latest_potential_path: Path | None = Field(
        default=None, description="Path to the latest potential file."
    )
    dataset_path: Path | None = Field(
        default=None, description="Path to the current training dataset (pickle)."
    )
    active_tasks: list[str] = Field(
        default_factory=list, description="List of IDs for currently active tasks."
    )
    halted_structures: list[Path] = Field(
        default_factory=list, description="List of halted structure dump files pending processing."
    )

    model_config = ConfigDict(extra="forbid")
