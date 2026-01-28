from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.atoms import ASEAtoms


class CandidateData(BaseModel):
    """
    Schema for candidate structures metadata and payload.
    """

    atoms: ASEAtoms = Field(..., description="ASE Atoms object")
    config_type: str = Field(..., description="Tag indicating origin")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    # Updated status pattern to include 'screening' and 'rejected' for Active Learning
    status: str = Field("pending", pattern="^(pending|training|failed|screening|rejected)$")
    generation: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")


class SelectionResult(BaseModel):
    """
    Result of the active set selection process.
    """
    selected_indices: list[int] = Field(..., description="Indices of selected candidates")
    total_candidates: int = Field(..., description="Total number of candidates considered")

    model_config = ConfigDict(extra="forbid")
