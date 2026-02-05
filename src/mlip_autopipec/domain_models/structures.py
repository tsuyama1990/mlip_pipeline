from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, field_validator


class StructureMetadata(BaseModel):
    """
    Data model representing an atomic structure with associated metadata.
    Wraps an ase.Atoms object.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    structure: Any  # Validated as ase.Atoms
    source: str
    generation_method: str

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, v: Any) -> Any:
        if not isinstance(v, Atoms):
            msg = f"structure must be an ase.Atoms object, got {type(v)}"
            raise TypeError(msg)
        return v
