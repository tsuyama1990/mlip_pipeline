from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.data_models.types import ASEAtoms


class CandidateData(BaseModel):
    """
    Schema for candidate structures metadata and payload.
    """

    atoms: ASEAtoms = Field(..., description="ASE Atoms object")
    config_type: str = Field(..., description="Tag indicating origin")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    status: str = Field("pending", pattern="^(pending|training|failed)$")
    generation: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")
