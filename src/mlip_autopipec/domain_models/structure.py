import contextlib
from typing import Annotated, Any

import numpy as np
from ase import Atoms
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    field_validator,
    model_validator,
)


def numpy_to_list(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def to_numpy(v: Any) -> Any:
    if isinstance(v, list):
        return np.array(v)
    return v


# Custom type for Numpy arrays with JSON serialization support
NumpyArray = Annotated[
    np.ndarray, PlainSerializer(numpy_to_list, return_type=list), BeforeValidator(to_numpy)
]


class Structure(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: NumpyArray
    atomic_numbers: NumpyArray
    cell: NumpyArray
    pbc: NumpyArray
    forces: NumpyArray | None = None
    energy: float | None = None
    stress: NumpyArray | None = None
    properties: dict[str, Any] | None = None

    @field_validator("positions", mode="after")
    @classmethod
    def validate_positions_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Positions must be (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("atomic_numbers", mode="after")
    @classmethod
    def validate_atomic_numbers_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            msg = f"Atomic numbers must be (N,), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("cell", mode="after")
    @classmethod
    def validate_cell_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.shape != (3, 3):
            msg = f"Cell must be (3, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("pbc", mode="after")
    @classmethod
    def validate_pbc_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.shape != (3,):
            msg = f"PBC must be (3,), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("forces", mode="after")
    @classmethod
    def validate_forces_shape(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None and (v.ndim != 2 or v.shape[1] != 3):
            msg = f"Forces must be (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("stress", mode="after")
    @classmethod
    def validate_stress_shape(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None and v.shape not in ((3, 3), (6,)):
            msg = f"Stress must be (3, 3) or (6,), got {v.shape}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "Structure":
        n_atoms = len(self.positions)
        if len(self.atomic_numbers) != n_atoms:
            msg = f"Mismatch: positions={n_atoms}, atomic_numbers={len(self.atomic_numbers)}"
            raise ValueError(msg)
        if self.forces is not None and len(self.forces) != n_atoms:
            msg = f"Mismatch: positions={n_atoms}, forces={len(self.forces)}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        # Extract energy/forces/stress if available in calc or info/arrays
        energy = None
        forces = None
        stress = None

        # Try calculator first
        if atoms.calc:
            with contextlib.suppress(Exception):
                energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            with contextlib.suppress(Exception):
                forces = atoms.get_forces()  # type: ignore[no-untyped-call]
            with contextlib.suppress(Exception):
                stress = atoms.get_stress()  # type: ignore[no-untyped-call]

        # Fallback to arrays/info if not in calc (e.g. read from file)
        if energy is None:
            energy = atoms.info.get("energy")
        if forces is None:
            forces = atoms.arrays.get("forces")
        if stress is None:
            stress = atoms.info.get("stress")

        return cls(
            positions=atoms.get_positions(),  # type: ignore[no-untyped-call]
            atomic_numbers=atoms.get_atomic_numbers(),  # type: ignore[no-untyped-call]
            cell=np.array(atoms.get_cell()),  # type: ignore[no-untyped-call]
            pbc=atoms.get_pbc(),  # type: ignore[no-untyped-call]
            forces=forces,
            energy=energy,
            stress=stress,
        )

    def to_ase(self) -> Atoms:
        atoms = Atoms(
            numbers=self.atomic_numbers, positions=self.positions, cell=self.cell, pbc=self.pbc
        )
        if self.energy is not None:
            atoms.info["energy"] = self.energy
        if self.forces is not None:
            atoms.arrays["forces"] = self.forces
        if self.stress is not None:
            atoms.info["stress"] = self.stress
        return atoms
