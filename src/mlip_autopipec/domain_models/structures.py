from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    uncertainty: float | None = None
    source: str = "unknown"
    parent_structure_id: str | None = None
    generation_method: str = "unknown"
    selection_score: float | None = None  # Added for Active Set Selection

    model_config = ConfigDict(extra="allow")


class CandidateStructure(BaseModel):
    # In a real scenario, this might hold the actual atoms object or a path to it.
    # For now, we use a path to a file representing the structure (e.g., .xyz, .cif)
    # or an identifier.
    structure_path: Path
    metadata: StructureMetadata = Field(default_factory=StructureMetadata)
    energy: float | None = None
    forces: list[list[float]] | None = None
    stress: list[list[float]] | None = None

    model_config = ConfigDict(extra="forbid")
