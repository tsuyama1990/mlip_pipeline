from pathlib import Path
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
        except ImportError as e:
            msg = "ASE not installed."
            raise ValueError(msg) from e

        if not isinstance(v, Atoms):
            msg = "Field 'atoms' must be an ase.Atoms object."
            raise TypeError(msg)
        return v


class InferenceResult(BaseModel):
    """
    Result from an inference run (e.g. LAMMPS MD).
    """
    succeeded: bool = Field(..., description="Whether the simulation completed successfully")
    max_gamma_observed: float = Field(0.0, description="Maximum extrapolation grade (gamma) observed")
    uncertain_structures: list[Path] = Field(default_factory=list, description="List of paths to dump files containing uncertain structures")

    model_config = ConfigDict(extra="forbid")
