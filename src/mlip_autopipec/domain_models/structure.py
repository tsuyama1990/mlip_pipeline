from typing import Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class Structure(BaseModel):
    """
    Domain model representing an atomic structure.
    Wraps an ASE Atoms object and includes MLIP-specific fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    atoms: Atoms
    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forces", "stress", mode="before")
    @classmethod
    def convert_to_array(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v)
        return v  # type: ignore[no-any-return]

    @field_serializer("forces", "stress")
    def serialize_array(self, v: np.ndarray | None, _info: Any) -> list[Any] | None:
        if v is None:
            return None
        return v.tolist()  # type: ignore[no-any-return]

    @field_serializer("atoms")
    def serialize_atoms(self, atoms: Atoms, _info: Any) -> dict[str, Any]:
        return {
            "symbols": str(atoms.symbols),
            "positions": atoms.get_positions().tolist(),  # type: ignore[no-untyped-call]
            "cell": atoms.get_cell().array.tolist(),  # type: ignore[no-untyped-call]
            "pbc": atoms.get_pbc().tolist(),  # type: ignore[no-untyped-call]
        }

    @field_validator("atoms", mode="before")
    @classmethod
    def validate_atoms(cls, v: Any) -> Atoms:
        if isinstance(v, Atoms):
            return v
        if isinstance(v, dict):
            return Atoms(
                symbols=v["symbols"], positions=v["positions"], cell=v["cell"], pbc=v["pbc"]
            )
        msg = "Invalid input for atoms"
        raise ValueError(msg)
