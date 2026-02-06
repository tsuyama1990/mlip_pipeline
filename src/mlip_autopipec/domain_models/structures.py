from typing import Any, Literal

from ase import Atoms
from pydantic import BaseModel, ConfigDict, field_validator


class StructureMetadata(BaseModel):
    """
    Metadata wrapper for atomic structures.

    Attributes:
        structure: The atomic structure (ase.Atoms).
        source: The origin of the structure (e.g., 'initial', 'active_learning', 'validation').
        generation_method: The method used to generate the structure (e.g., 'random', 'mock', 'md').
    """

    structure: Any
    source: Literal["initial", "active_learning", "validation"]
    generation_method: Literal["random", "mock", "md", "perturbation"]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, v: Any) -> Atoms:
        if not isinstance(v, Atoms):
            msg = "structure must be an ase.Atoms object"
            raise TypeError(msg)
        return v
