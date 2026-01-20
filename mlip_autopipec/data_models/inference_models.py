from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedStructure(BaseModel):
    """
    Data model for an extracted local environment (cluster-in-box).

    Attributes:
        atoms: The ASE Atoms object representing the extracted cluster.
        origin_uuid: The UUID of the original MD frame.
        origin_index: The index of the focal atom in the original frame.
        mask_radius: The radius used for force masking (core radius).
    """
    atoms: Any = Field(..., description="The ASE Atoms object")
    origin_uuid: str = Field(..., description="UUID of the original MD frame")
    origin_index: int = Field(..., description="Index of the focal atom in original frame")
    mask_radius: float = Field(..., description="Radius used for force masking")

    model_config = ConfigDict(extra="forbid")

    @field_validator("atoms")
    @classmethod
    def validate_atoms(cls, v: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError:
            raise ValueError("ASE not installed.")

        if not isinstance(v, Atoms):
            raise TypeError("Field 'atoms' must be an ase.Atoms object.")
        return v
