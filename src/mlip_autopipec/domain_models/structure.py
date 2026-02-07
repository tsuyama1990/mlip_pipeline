from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class Structure(BaseModel):
    """
    Represents an atomic configuration (domain model).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    positions: np.ndarray
    atomic_numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("positions", mode="before")
    @classmethod
    def validate_positions(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            return np.array(v, dtype=float)
        if not isinstance(v, np.ndarray):
            msg = "positions must be a numpy array or list"
            raise TypeError(msg)
        if v.ndim != 2 or v.shape[1] != 3:
            msg = "positions must be Nx3 array"
            raise ValueError(msg)
        return v

    @field_validator("atomic_numbers", mode="before")
    @classmethod
    def validate_atomic_numbers(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            return np.array(v, dtype=int)
        if not isinstance(v, np.ndarray):
            msg = "atomic_numbers must be a numpy array or list"
            raise TypeError(msg)
        if v.ndim != 1:
            msg = "atomic_numbers must be 1D array"
            raise ValueError(msg)
        return v

    @field_validator("cell", mode="before")
    @classmethod
    def validate_cell(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            return np.array(v, dtype=float)
        if not isinstance(v, np.ndarray):
            msg = "cell must be a numpy array or list"
            raise TypeError(msg)
        if v.shape != (3, 3):
            msg = "cell must be 3x3 array"
            raise ValueError(msg)
        return v

    @field_validator("pbc", mode="before")
    @classmethod
    def validate_pbc(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            return np.array(v, dtype=bool)
        if not isinstance(v, np.ndarray):
            msg = "pbc must be a numpy array or list"
            raise TypeError(msg)
        if v.shape != (3,):
            msg = "pbc must be array of 3 booleans"
            raise ValueError(msg)
        return v

    @field_validator("forces", mode="before")
    @classmethod
    def validate_forces(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v, dtype=float)
        if not isinstance(v, np.ndarray):
            msg = "forces must be a numpy array or list"
            raise TypeError(msg)
        if v.ndim != 2 or v.shape[1] != 3:
            msg = "forces must be Nx3 array"
            raise ValueError(msg)
        return v

    @field_validator("stress", mode="before")
    @classmethod
    def validate_stress(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v, dtype=float)
        if not isinstance(v, np.ndarray):
            msg = "stress must be a numpy array or list"
            raise TypeError(msg)
        # Stress can be 3x3 or Voigt notation (6,)
        return v

    @field_serializer("positions", "atomic_numbers", "cell", "pbc", "forces", "stress")
    def serialize_numpy(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
