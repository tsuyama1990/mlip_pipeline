from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateData(BaseModel):
    """
    Schema for candidate structures metadata and payload.
    """

    atoms: Any = Field(..., description="ASE Atoms object")
    config_type: str = Field(..., description="Tag indicating origin")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    status: str = Field("pending", pattern="^(pending|training|failed)$")
    generation: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("atoms")
    @classmethod
    def validate_atoms(cls, v: Any) -> Any:
        # Check if it has 'get_positions' and 'get_cell' methods (Duck typing for ASE Atoms)
        if not (hasattr(v, "get_positions") and hasattr(v, "get_cell")):
             raise ValueError("Field 'atoms' must be an ASE Atoms object")
        return v
