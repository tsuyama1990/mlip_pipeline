# This module defines the schemas for Training Data.
import math
from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, field_validator

# Type alias for Matrix3x3 (List of 3 Lists of 3 floats)
Vector3D = list[float]
Matrix3x3 = list[Vector3D]


class TrainingData(BaseModel):
    """
    Schema for validated training data.
    Enforces shape and finite values for energy and forces.
    """

    energy: float
    forces: list[Vector3D]
    stress: Matrix3x3 | Vector3D | None = None
    virial: Matrix3x3 | Vector3D | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("energy")
    @classmethod
    def check_energy_finite(cls, v: float) -> float:
        if not isinstance(v, float):
            msg = f"Energy must be a float, got {type(v)}"
            raise ValueError(msg)
        if not math.isfinite(v):
            msg = f"Energy value {v} is not finite."
            raise ValueError(msg)
        return v

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, v: list[Vector3D]) -> list[Vector3D]:
        """
        Validates that the forces array is a list of 3-component vectors.
        """
        if not v:
            return v

        # Check shape of each vector
        for i, row in enumerate(v):
            if not isinstance(row, list):
                msg = f"Force vector at index {i} must be a list."
                raise ValueError(msg)
            if len(row) != 3:
                msg = f"Force vector at index {i} must have exactly 3 components (got {len(row)})."
                raise ValueError(msg)

        # Check for NaN or Infinity
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    msg = f"Force value at [{i}, {j}] must be a number."
                    raise ValueError(msg)
                if not math.isfinite(val):
                    msg = f"Force value {val} at [{i}, {j}] is not finite."
                    raise ValueError(msg)
        return v

    @field_validator("stress", "virial")
    @classmethod
    def check_matrix_shape(cls, v: Matrix3x3 | Vector3D | None) -> Matrix3x3 | Vector3D | None:
        if v is None:
            return v

        # 3x3 Matrix
        if len(v) == 3:
            if not all(len(row) == 3 for row in v):  # type: ignore
                msg = "Stress/Virial matrix must be 3x3."
                raise ValueError(msg)
        # Voigt notation (6 components) or flattened (9)
        elif len(v) not in [6, 9]:
            msg = "Stress/Virial must be 3x3 matrix or 6-component vector."
            raise ValueError(msg)

        return v


class TrainingBatch(BaseModel):
    """
    Schema for a batch of training data exported to Pacemaker.
    """

    atoms_list: list["Atoms"]  # Forward reference to ASE Atoms
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("atoms_list")
    @classmethod
    def validate_atoms_list(cls, v: list[Any]) -> list[Any]:
        if not v:
            return v
        # Basic check
        if not hasattr(v[0], "get_positions"):
            msg = "Items in atoms_list do not appear to be ASE Atoms objects."
            raise ValueError(msg)
        return v
