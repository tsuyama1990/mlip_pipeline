from typing import Any

import ase
from pydantic import BaseModel, ConfigDict, field_validator


class StructureMetadata(BaseModel):
    """
    Domain model representing an atomic structure with associated metadata.
    Enforces strict validation to ensure data integrity.
    """
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    structure: ase.Atoms
    source: str
    generation_method: str
    parent_structure_id: str | None = None
    uncertainty: float | None = None
    selection_score: float | None = None
    filepath: str | None = None

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, v: Any) -> ase.Atoms:
        """Strictly validate that the structure is a valid ase.Atoms object."""
        if not isinstance(v, ase.Atoms):
            msg = f"structure must be an ase.Atoms instance, got {type(v)}"
            raise TypeError(msg)
        return v
