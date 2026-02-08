import contextlib
from typing import Annotated, Any

import numpy as np
from ase import Atoms
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
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
    """
    Domain entity representing an atomic structure.

    Attributes `forces`, `energy`, and `stress` are Optional because structures
    are initially generated without labels. Use `validate_labeled()` to enforce
    their presence for labeled datasets.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: NumpyArray
    atomic_numbers: NumpyArray
    cell: NumpyArray
    pbc: NumpyArray
    forces: NumpyArray | None = None
    energy: float | None = None
    stress: NumpyArray | None = None
    tags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("positions", mode="after")
    @classmethod
    def validate_positions_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Positions must be (N, 3), got {v.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(v)):
            msg = "Positions contain non-finite values"
            raise ValueError(msg)
        return v

    @field_validator("atomic_numbers", mode="after")
    @classmethod
    def validate_atomic_numbers(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            msg = f"Atomic numbers must be (N,), got {v.shape}"
            raise ValueError(msg)
        if np.any((v < 1) | (v > 118)):
            msg = "Atomic numbers must be between 1 and 118"
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
    def validate_forces(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None:
            if v.ndim != 2 or v.shape[1] != 3:
                msg = f"Forces must be (N, 3), got {v.shape}"
                raise ValueError(msg)
            if not np.all(np.isfinite(v)):
                msg = "Forces contain non-finite values"
                raise ValueError(msg)
            # Soft check for magnitude (e.g. warn or fail if > 1000 eV/A)
            # For strict data integrity, we fail on absurd values
            if np.any(np.abs(v) > 1000.0):
                msg = "Forces magnitude exceeds reasonable limit (1000 eV/A)"
                raise ValueError(msg)
        return v

    @field_validator("energy", mode="after")
    @classmethod
    def validate_energy(cls, v: float | None) -> float | None:
        if v is not None:
            if not np.isfinite(v):
                msg = "Energy must be finite"
                raise ValueError(msg)
            # Soft check for magnitude per structure
            if abs(v) > 1e6:  # Arbitrary large limit for total energy
                msg = "Energy magnitude exceeds reasonable limit (1e6 eV)"
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

    def validate_labeled(self) -> None:
        """Ensure structure has labels (energy, forces, stress)."""
        if self.energy is None:
            msg = "Structure missing energy label"
            raise ValueError(msg)
        if self.forces is None:
            msg = "Structure missing forces label"
            raise ValueError(msg)
        if self.stress is None:
            msg = "Structure missing stress label"
            raise ValueError(msg)

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

        # Copy info to tags, ensuring we handle non-serializable objects if necessary
        # For now, we assume atoms.info contains serializable data or data we can ignore errors on later?
        # But Structure.tags is dict[str, Any].
        tags = atoms.info.copy()

        # Validation: check atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()  # type: ignore[no-untyped-call]
        if np.any((atomic_numbers < 1) | (atomic_numbers > 118)):
            msg = "Atomic numbers must be between 1 and 118"
            raise ValueError(msg)

        return cls(
            positions=atoms.get_positions(),  # type: ignore[no-untyped-call]
            atomic_numbers=atomic_numbers,
            cell=np.array(atoms.get_cell()),  # type: ignore[no-untyped-call]
            pbc=atoms.get_pbc(),  # type: ignore[no-untyped-call]
            forces=forces,
            energy=energy,
            stress=stress,
            tags=tags,
        )

    def to_ase(self) -> Atoms:
        atoms = Atoms(
            numbers=self.atomic_numbers, positions=self.positions, cell=self.cell, pbc=self.pbc
        )
        if self.tags:
            atoms.info.update(self.tags)

        if self.energy is not None:
            atoms.info["energy"] = self.energy
        if self.forces is not None:
            atoms.arrays["forces"] = self.forces
        if self.stress is not None:
            atoms.info["stress"] = self.stress
        return atoms
