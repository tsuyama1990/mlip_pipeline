from typing import Any

import ase
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Structure(BaseModel):
    """
    Fundamental object representing a collection of atoms.
    Wraps ASE Atoms with strict Pydantic validation.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("positions")
    @classmethod
    def validate_positions_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Positions must have shape (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("cell")
    @classmethod
    def validate_cell_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.shape != (3, 3):
            msg = f"Cell must be a 3x3 matrix, got {v.shape}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> "Structure":
        if len(self.symbols) != len(self.positions):
            msg = (
                f"Number of symbols ({len(self.symbols)}) does not match "
                f"number of positions ({len(self.positions)})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_ase(cls, atoms: ase.Atoms) -> "Structure":
        """Factory method to create a Structure from an ASE Atoms object."""
        return cls(
            symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            positions=atoms.get_positions(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell().array,  # type: ignore[no-untyped-call]
            pbc=tuple(atoms.get_pbc()),  # type: ignore[no-untyped-call]
            properties=atoms.info.copy(),
        )

    def to_ase(self) -> ase.Atoms:
        """Convert to ASE Atoms object."""
        return ase.Atoms(
            symbols=self.symbols,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            info=self.properties.copy()
        )

    def get_chemical_formula(self) -> str:
        """Get the chemical formula string."""
        return str(self.to_ase().get_chemical_formula())  # type: ignore[no-untyped-call]
