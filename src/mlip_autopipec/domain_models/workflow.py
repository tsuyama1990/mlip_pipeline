from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.validation import ValidationResult


class HistoryEntry(BaseModel):
    iteration: int
    potential: str
    status: str
    candidates_count: int
    new_data_count: int
    validation_result: ValidationResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class WorkflowState(BaseModel):
    iteration: int = 0
    current_potential_path: Path | None = None
    history: list[HistoryEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
