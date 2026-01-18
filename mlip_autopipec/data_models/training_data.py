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

    # Use Any because ase.Atoms is not a Pydantic model.
    # Ideally we would use list[Atoms] but that requires arbitrary types allowed
    # and might cause runtime issues if imports are circular or missing.
    # However, strict validation is done via validator.
    # We hint as list[Any] to satisfy Pydantic V2 without errors, but logical type is list[Atoms].
    atoms_list: list[Any] = Field(..., description="List of ase.Atoms objects.")
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
