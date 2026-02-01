from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult


class WorkflowPhase(str, Enum):
    """
    Enum representing the current phase of the Active Learning Cycle.
    """
    EXPLORATION = "EXPLORATION"
    SELECTION = "SELECTION"
    CALCULATION = "CALCULATION"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"


class CandidateStatus(str, Enum):
    """
    Enum representing the processing status of a candidate structure.
    """
    PENDING = "PENDING"
    CALCULATING = "CALCULATING"
    DONE = "DONE"
    FAILED = "FAILED"


class CandidateStructure(BaseModel):
    """
    Represents a structure identified for calculation (e.g., from MD halt).
    """
    model_config = ConfigDict(extra="forbid")

    structure: Structure
    origin: str = Field(..., description="Origin of the candidate (e.g., 'MD_halt_cycle_3')")
    uncertainty_score: float = Field(..., description="The uncertainty value (gamma) that triggered selection")
    status: CandidateStatus = Field(default=CandidateStatus.PENDING, description="Current processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if status is FAILED")


class WorkflowState(BaseModel):
    """
    Tracks the state of the Active Learning Loop.
    Persisted to 'workflow_state.json' to enable resumability.
    """
    model_config = ConfigDict(extra="forbid")

    project_name: str
    generation: int = Field(default=0, ge=0)
    current_phase: WorkflowPhase = Field(default=WorkflowPhase.EXPLORATION)
    latest_potential_path: Optional[Path] = Field(default=None, description="Path to the current best potential")
    dataset_path: Path = Field(..., description="Path to the accumulated training dataset")
    candidates: List[CandidateStructure] = Field(default_factory=list, description="List of candidates for the current generation")
    validation_history: Dict[int, ValidationResult] = Field(default_factory=dict, description="Validation results by generation")
