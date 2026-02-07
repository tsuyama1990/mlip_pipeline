from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Structure(BaseModel):
    """
    Represents an atomic configuration.
    """
    model_config = ConfigDict(extra="forbid")

    atomic_numbers: list[int]
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]
    energy: float | None = None
    forces: list[list[float]] | None = None
    stress: list[list[float]] | None = None
    properties: dict[str, Any] | None = None

    @model_validator(mode='after')
    def check_shapes(self) -> 'Structure':
        n_atoms = len(self.atomic_numbers)
        if len(self.positions) != n_atoms:
            msg = f"positions length {len(self.positions)} does not match atomic_numbers length {n_atoms}"
            raise ValueError(msg)

        if self.forces is not None and len(self.forces) != n_atoms:
            msg = f"forces length {len(self.forces)} does not match atomic_numbers length {n_atoms}"
            raise ValueError(msg)

        # Check cell shape (3x3)
        if len(self.cell) != 3 or any(len(v) != 3 for v in self.cell):
             msg = "cell must be 3x3"
             raise ValueError(msg)

        # Check pbc shape (3)
        if len(self.pbc) != 3:
            msg = "pbc must be length 3"
            raise ValueError(msg)

        return self
