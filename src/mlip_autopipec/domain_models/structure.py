from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("positions")
    @classmethod
    def validate_positions(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = "Positions must be an (N, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("cell")
    @classmethod
    def validate_cell(cls, v: np.ndarray) -> np.ndarray:
        if v.shape != (3, 3):
            msg = "Cell must be a (3, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("forces")
    @classmethod
    def validate_forces(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return v
        if v.ndim != 2 or v.shape[1] != 3:
            msg = "Forces must be an (N, 3) array"
            raise ValueError(msg)
        return v

    @field_validator("stress")
    @classmethod
    def validate_stress(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return v
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

    def __repr__(self) -> str:
        """
        Concise representation of the Structure.
        """
        unique_species, counts = np.unique(self.species, return_counts=True)
        # np.unique guarantees same length for values and counts
        formula = "".join(f"{s}{c if c > 1 else ''}" for s, c in zip(unique_species, counts, strict=True))
        return f"Structure(formula='{formula}', atoms={len(self.species)}, energy={self.energy})"

# Removed Dataset model as it encourages loading all structures into memory.
# Training should accept Iterable[Structure] directly.
