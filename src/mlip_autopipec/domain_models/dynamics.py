from pydantic import BaseModel, ConfigDict, Field


class MDState(BaseModel):
    """Snapshot of an MD simulation state."""

    model_config = ConfigDict(extra="forbid")

    current_step: int = Field(..., ge=0, description="Current MD step")
    temperature: float = Field(..., ge=0, description="Current temperature")
    halt_detected: bool = Field(default=False, description="Whether a halt event was detected")
    halt_reason: str | None = Field(None, description="Reason for the halt if detected")
