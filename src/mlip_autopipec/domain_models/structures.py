from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    uncertainty: float | None = None
    source: str = "unknown"
    model_config = ConfigDict(extra="allow")


class CandidateStructure(BaseModel):
    # In a real scenario, this might hold the actual atoms object or a path to it.
    # For now, we use a path to a file representing the structure (e.g., .xyz, .cif)
    # or an identifier.
    structure_path: Path
    metadata: StructureMetadata = Field(default_factory=StructureMetadata)

    model_config = ConfigDict(extra="forbid")
