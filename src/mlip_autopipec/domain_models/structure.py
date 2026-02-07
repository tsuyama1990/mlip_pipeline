from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class Structure(BaseModel):
    """
    A simplified wrapper for atomic data.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    positions: np.ndarray
    cell: np.ndarray
    species: list[str]
    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("positions", mode="before")
    @classmethod
    def validate_positions(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
             # Let pydantic raise error or we raise
             msg = f"positions must be a numpy array or list, got {type(v)}"
             raise TypeError(msg)

        if v.ndim != 2 or v.shape[1] != 3:
            msg = "Positions must be an (N, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("cell", mode="before")
    @classmethod
    def validate_cell(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
             msg = f"cell must be a numpy array or list, got {type(v)}"
             raise TypeError(msg)

        if v.shape != (3, 3):
            msg = "Cell must be a (3, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("forces", mode="before")
    @classmethod
    def validate_forces(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return v
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
             msg = f"forces must be a numpy array or list, got {type(v)}"
             raise TypeError(msg)

        if v.ndim != 2 or v.shape[1] != 3:
            msg = "Forces must be an (N, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("stress", mode="before")
    @classmethod
    def validate_stress(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return v
        if isinstance(v, list):
            v = np.array(v)
        if not isinstance(v, np.ndarray):
             msg = f"stress must be a numpy array or list, got {type(v)}"
             raise TypeError(msg)

        if v.shape != (3, 3):
            msg = "Stress must be a (3, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Validate that properties are JSON-safe or numpy arrays.
        """
        allowed_types = (int, float, str, bool, list, tuple, np.ndarray, np.generic)
        for key, value in v.items():
            if not isinstance(key, str):
                msg = f"Property key must be a string, got {type(key)}"
                raise TypeError(msg)
            if value is not None and not isinstance(value, allowed_types):
                # Check for dict (nested)? For now forbid.
                msg = f"Property '{key}' has invalid type {type(value)}. Allowed types: {allowed_types}"
                raise TypeError(msg)
        return v

    @field_serializer("positions", "cell")
    def serialize_array(self, v: np.ndarray, _info: Any) -> list[Any]:
        return cast(list[Any], v.tolist())

    @field_serializer("forces", "stress")
    def serialize_optional_array(self, v: np.ndarray | None, _info: Any) -> list[Any] | None:
        if v is None:
            return None
        return cast(list[Any], v.tolist())

    @field_serializer("properties")
    def serialize_properties(self, v: dict[str, Any], _info: Any) -> dict[str, Any]:
        new_v = {}
        for k, val in v.items():
            if isinstance(val, (np.ndarray, np.generic)):
                new_v[k] = val.tolist()
            else:
                new_v[k] = val
        return new_v

    def __repr__(self) -> str:
        """
        Concise representation of the Structure.
        """
        unique_species, counts = np.unique(self.species, return_counts=True)
        # np.unique guarantees same length for values and counts
        formula = "".join(f"{s}{c if c > 1 else ''}" for s, c in zip(unique_species, counts, strict=True))
        return f"Structure(formula='{formula}', atoms={len(self.species)}, energy={self.energy})"

    def apply_periodic_embedding(self, center: np.ndarray, radius: float, buffer: float) -> "Structure":
        """
        Creates a supercell containing only atoms within `radius + buffer` of `center`,
        ensuring minimal image convention.
        """
        if radius <= 0:
            msg = "radius must be positive"
            raise ValueError(msg)
        if buffer < 0:
            msg = "buffer must be non-negative"
            raise ValueError(msg)

        center = np.array(center)
        if center.shape != (3,):
            msg = "center must be a (3,) array"
            raise ValueError(msg)

        # Check if cell is singular before inversion
        det = np.linalg.det(self.cell)
        if np.isclose(det, 0.0):
            msg = "Cell is singular (determinant is zero)"
            raise ValueError(msg)

        inv_cell = np.linalg.inv(self.cell)

        # Validation: Check if center is within [0, 1) fractional coordinates
        # This is strictly required by audit feedback, although MIC handles wrapping.
        center_frac = center @ inv_cell
        if np.any(center_frac < -1e-6) or np.any(center_frac >= 1.0 + 1e-6):
             # We allow slight numerical noise but warn or error on large deviation.
             # Audit says "validation", so we raise.
             # Wait, usually we wrap centers. But the requirement says "Validate".
             msg = "Center must be within the unit cell (fractional coordinates [0, 1))"
             raise ValueError(msg)

        # Calculate distance with MIC
        diff = self.positions - center

        # Transform to fractional coordinates: r = x*a1 + y*a2 + z*a3 -> r @ inv_cell
        diff_frac = diff @ inv_cell

        # Apply MIC: wrap to [-0.5, 0.5]
        diff_frac -= np.round(diff_frac)

        # Transform back to Cartesian
        diff_cart = diff_frac @ self.cell

        dists = np.linalg.norm(diff_cart, axis=1)
        mask = dists < (radius + buffer)

        new_positions = diff_cart[mask]
        new_species = [s for i, s in enumerate(self.species) if mask[i]]

        # Create a new cubic cell large enough to hold the cluster + vacuum
        # Using 5.0 Angstrom vacuum is a safe default for cluster in box
        L = 2 * (radius + buffer) + 5.0
        new_cell = np.eye(3) * L

        # Center the cluster in the new box
        # new_positions are relative to 'center'.
        # We put 'center' at L/2, L/2, L/2
        new_positions += L / 2.0

        new_forces = None
        if self.forces is not None:
            new_forces = self.forces[mask]

        return Structure(
            positions=new_positions,
            cell=new_cell,
            species=new_species,
            forces=new_forces
        )
