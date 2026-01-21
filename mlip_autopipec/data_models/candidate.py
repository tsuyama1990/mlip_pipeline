from pydantic import BaseModel, ConfigDict, Field


class CandidateData(BaseModel):
    """
    Schema for candidate structures metadata before DFT.
    """
    status: str = Field(..., pattern="^(pending|training|failed)$")
    generation: int = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid")
