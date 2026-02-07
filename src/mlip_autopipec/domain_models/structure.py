from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class Structure(BaseModel):
    """
    Domain model representing an atomic structure.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    positions: npt.NDArray[np.float64]
    atomic_numbers: npt.NDArray[np.int_]
    cell: npt.NDArray[np.float64]
    pbc: npt.NDArray[np.bool_]
    energy: float | None = None
    forces: npt.NDArray[np.float64] | None = None
    stress: npt.NDArray[np.float64] | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    # Validators for shapes
    @field_validator("positions", mode="before")
    @classmethod
    def validate_positions(cls, v: Any) -> npt.NDArray[np.float64]:
        arr = np.array(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            msg = "Positions must be an (N, 3) array"
            raise ValueError(msg)
        return arr

    @field_validator("atomic_numbers", mode="before")
    @classmethod
    def validate_atomic_numbers(cls, v: Any) -> npt.NDArray[np.int_]:
        arr = np.array(v, dtype=int)
        if arr.ndim != 1:
            msg = "Atomic numbers must be a 1D array"
            raise ValueError(msg)
        return arr

    @field_validator("cell", mode="before")
    @classmethod
    def validate_cell(cls, v: Any) -> npt.NDArray[np.float64]:
        arr = np.array(v, dtype=np.float64)
        if arr.shape != (3, 3):
            msg = "Cell must be a (3, 3) array"
            raise ValueError(msg)
        return arr

    @field_validator("pbc", mode="before")
    @classmethod
    def validate_pbc(cls, v: Any) -> npt.NDArray[np.bool_]:
        arr = np.array(v, dtype=bool)
        if arr.ndim != 1 or arr.shape[0] != 3:
            msg = "PBC must be a 3-element boolean array"
            raise ValueError(msg)
        return arr

    @field_validator("forces", mode="before")
    @classmethod
    def validate_forces(cls, v: Any) -> npt.NDArray[np.float64] | None:
        if v is None:
            return None
        arr = np.array(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            msg = "Forces must be an (N, 3) array"
            raise ValueError(msg)
        return arr

    @field_validator("stress", mode="before")
    @classmethod
    def validate_stress(cls, v: Any) -> npt.NDArray[np.float64] | None:
        if v is None:
            return None
        # Stress can be (3, 3) or (6,) (Voigt) or (9,).
        # Usually ASE uses (6,) or (3, 3). Let's accept (3, 3) or flat array of 6 or 9.
        # But commonly full tensor is 3x3.
        # Let's enforce 3x3 or (6,) for flexibility if needed, but SPEC said float array.
        # Let's assume (3, 3) for now or just check it's an array.
        # If it's Virial stress, it's 3x3.
        return np.array(v, dtype=np.float64)

    @model_validator(mode="after")
    def validate_consistency(self) -> "Structure":
        n_atoms = self.positions.shape[0]
        if self.atomic_numbers.shape[0] != n_atoms:
            msg = f"Length mismatch: positions ({n_atoms}) vs atomic_numbers ({self.atomic_numbers.shape[0]})"
            raise ValueError(msg)
        if self.forces is not None and self.forces.shape[0] != n_atoms:
            msg = f"Length mismatch: positions ({n_atoms}) vs forces ({self.forces.shape[0]})"
            raise ValueError(msg)
        return self

    @field_serializer("positions", "atomic_numbers", "cell", "pbc", "forces", "stress")
    def serialize_numpy(self, v: Any, _info: Any) -> list[Any] | None:
        if v is None:
            return None
        return v.tolist()  # type: ignore[no-any-return]
