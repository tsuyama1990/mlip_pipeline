"""
This module defines data models for encapsulating domain objects like atomic structures.
This ensures type safety and validation at boundaries where raw objects (like ase.Atoms)
are passed between modules.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from ase import Atoms


class TrainingBatch(BaseModel):
    """
    A container for a batch of atomic structures to be used for training.
    Wraps a list of ase.Atoms objects to ensure they are valid.
    """

    # Use TYPE_CHECKING string forward reference for Atoms
    atoms_list: list["Atoms"] = Field(..., description="List of ase.Atoms objects.")
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms_list")
    @classmethod
    def validate_atoms_list(cls, v: list[Any]) -> list[Any]:
        try:
            from ase import Atoms
        except ImportError as e:
            msg = "ASE is required for validation."
            raise ImportError(msg) from e

        if not isinstance(v, list):
            msg = "atoms_list must be a list."
            raise TypeError(msg)

        for i, atom in enumerate(v):
            if not isinstance(atom, Atoms):
                msg = f"Item at index {i} is not an ase.Atoms object."
                raise TypeError(msg)

        return v


class TrainingData(BaseModel):
    """
    Data model representing a single training point (Structure + Labels).
    This is what we write to disk (e.g. extxyz) or send to Pacemaker.
    """
    structure_uid: str
    energy: float | None = None
    forces: list[list[float]] | None = None
    virial: list[list[float]] | None = None
    stress: list[list[float]] | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is not None:
            for i, f in enumerate(v):
                if len(f) != 3:
                     raise ValueError(f"Force vector at index {i} must be size 3.")
        return v
