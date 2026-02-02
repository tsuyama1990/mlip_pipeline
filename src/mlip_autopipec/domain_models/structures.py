from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """Metadata for an atomic structure."""
    source: str
    creation_method: str
    tags: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class CandidateStructure(BaseModel):
    """A structure candidate for calculation."""
    filepath: Path
    metadata: StructureMetadata

    model_config = ConfigDict(extra="forbid")
