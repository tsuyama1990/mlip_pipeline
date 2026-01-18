"""
This module defines data models for encapsulating domain objects like atomic structures.
This ensures type safety and validation at boundaries where raw objects (like ase.Atoms)
are passed between modules.
"""

from typing import TYPE_CHECKING, Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, BeforeValidator

if TYPE_CHECKING:
    from ase import Atoms

def check_3d_vector(v: list[float]) -> list[float]:
    if len(v) != 3:
        raise ValueError("Vector must be size 3.")
    return v

# Type alias for a 3D vector
Vector3D = Annotated[list[float], BeforeValidator(check_3d_vector)]

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
    # Use nested annotation for list of 3D vectors
    forces: list[Vector3D] | None = None
    virial: list[Vector3D] | None = None # Virial typically 3x3? But sometimes contracted. Spec says list[list[float]].
    # Wait, stress is 3x3. Virial is also tensor.
    # The previous code assumed list[list[float]].
    # Let's keep it consistent.
    stress: list[Vector3D] | None = None

    model_config = ConfigDict(extra="forbid")

    # We replaced the explicit validator with the Type Annotation 'Vector3D'
