import copy
import logging
from typing import Annotated, Any, cast

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from mlip_autopipec.constants import (
    MAX_ATOMIC_NUMBER,
    MAX_ENERGY_MAGNITUDE,
    MAX_FORCE_MAGNITUDE,
)

logger = logging.getLogger(__name__)


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


def voigt_6_to_full_3x3(stress_voigt: np.ndarray) -> np.ndarray:
    """
    Convert Voigt notation (6,) to full tensor (3, 3).
    ASE Voigt order: xx, yy, zz, yz, xz, xy
    """
    if stress_voigt.shape != (6,):
        msg = f"Expected Voigt stress shape (6,), got {stress_voigt.shape}"
        raise ValueError(msg)

    xx, yy, zz, yz, xz, xy = stress_voigt
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


class Structure(BaseModel):
    """
    Domain entity representing an atomic structure.

    Attributes `forces`, `energy`, and `stress` are Optional because structures
    are initially generated without labels. Use `validate_labeled()` to enforce
    their presence for labeled datasets.

    Stress is always stored as a (3, 3) tensor.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    positions: NumpyArray
    atomic_numbers: NumpyArray
    cell: NumpyArray
    pbc: NumpyArray
    forces: NumpyArray | None = None
    energy: float | None = None
    stress: NumpyArray | None = None
    uncertainty: float | None = None
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
        if np.any((v < 1) | (v > MAX_ATOMIC_NUMBER)):
            msg = f"Atomic numbers must be between 1 and {MAX_ATOMIC_NUMBER}"
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
            if np.any(np.abs(v) > MAX_FORCE_MAGNITUDE):
                msg = f"Forces magnitude exceeds reasonable limit ({MAX_FORCE_MAGNITUDE} eV/A)"
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
            if abs(v) > MAX_ENERGY_MAGNITUDE:  # Arbitrary large limit for total energy
                msg = f"Energy magnitude exceeds reasonable limit ({MAX_ENERGY_MAGNITUDE} eV)"
                raise ValueError(msg)
        return v

    @field_validator("stress", mode="before")
    @classmethod
    def validate_stress_shape_and_convert(cls, v: Any) -> Any:
        if v is not None:
            v_arr = to_numpy(v)
            if v_arr.shape == (6,):
                return voigt_6_to_full_3x3(v_arr)
            if v_arr.shape != (3, 3):
                msg = f"Stress must be (3, 3) or (6,), got {v_arr.shape}"
                raise ValueError(msg)
            return v_arr
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
        """
        Ensure structure has labels (energy, forces, stress) and they are valid.
        """
        if self.energy is None:
            msg = "Structure missing energy label"
            raise ValueError(msg)

        if self.forces is None:
            msg = "Structure missing forces label"
            raise ValueError(msg)

        if self.stress is None:
            msg = "Structure missing stress label"
            raise ValueError(msg)

    def copy(self) -> "Structure":
        """
        Create a deep copy of the Structure.
        """
        return Structure(
            positions=self.positions.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
            forces=self.forces.copy() if self.forces is not None else None,
            energy=self.energy,
            stress=self.stress.copy() if self.stress is not None else None,
            uncertainty=self.uncertainty,
            tags=copy.deepcopy(self.tags),
        )

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        """
        Create a Structure from an ASE Atoms object.
        Performs strict validation of inputs BEFORE creating the Pydantic model
        to prevent any invalid state.
        """
        # Validate critical array lengths match before processing
        n_atoms = len(atoms)
        try:
            positions = atoms.get_positions()
        except Exception as e:
            msg = f"Failed to get positions: {e}"
            raise ValueError(msg) from e

        if len(positions) != n_atoms:
            msg = f"Mismatch: ASE atoms length {n_atoms} != positions length {len(positions)}"
            raise ValueError(msg)

        atomic_numbers = cls._extract_atomic_numbers(atoms)
        # Re-validate positions strictly
        positions = cls._extract_positions(atoms, len(atomic_numbers))
        cell, pbc = cls._extract_cell_pbc(atoms)

        # Extract labels (optional)
        energy, forces, stress = cls._extract_labels(atoms)
        uncertainty = atoms.info.get("uncertainty")
        tags = atoms.info.copy()

        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            forces=forces,
            energy=energy,
            stress=stress,
            uncertainty=uncertainty,
            tags=tags,
        )

    @staticmethod
    def _extract_atomic_numbers(atoms: Atoms) -> np.ndarray:
        try:
            atomic_numbers = atoms.get_atomic_numbers()
        except Exception as e:
            msg = f"Failed to get atomic numbers from ASE atoms: {e}"
            raise ValueError(msg) from e

        if not isinstance(atomic_numbers, np.ndarray):
             atomic_numbers = np.array(atomic_numbers)

        if np.any((atomic_numbers < 1) | (atomic_numbers > MAX_ATOMIC_NUMBER)):
            msg = f"Atomic numbers must be between 1 and {MAX_ATOMIC_NUMBER}"
            raise ValueError(msg)
        # Using cast to silence 'no-any-return' mypy error when returning ndarray from Any context
        return cast(np.ndarray, atomic_numbers)

    @staticmethod
    def _extract_positions(atoms: Atoms, n_atoms: int) -> np.ndarray:
        try:
            positions = atoms.get_positions()
        except Exception as e:
            msg = f"Failed to get positions from ASE atoms: {e}"
            raise ValueError(msg) from e

        if positions.ndim != 2 or positions.shape[1] != 3:
             msg = f"Positions must be (N, 3), got {positions.shape}"
             raise ValueError(msg)

        if len(positions) != n_atoms:
             msg = f"Mismatch: positions={len(positions)}, atomic_numbers={n_atoms}"
             raise ValueError(msg)
        return cast(np.ndarray, positions)

    @staticmethod
    def _extract_cell_pbc(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        try:
             cell = np.array(atoms.get_cell())
             pbc = atoms.get_pbc()
        except Exception as e:
             msg = f"Failed to get cell/pbc from ASE atoms: {e}"
             raise ValueError(msg) from e
        return cell, pbc

    @staticmethod
    def _extract_labels(atoms: Atoms) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        energy = None
        forces = None
        stress = None

        if atoms.calc:
            # Explicit error handling
            try:
                energy = atoms.get_potential_energy()
            except Exception as e:
                logger.warning(f"Could not retrieve potential energy from ASE atoms: {e}")

            try:
                forces = atoms.get_forces()
            except Exception as e:
                logger.warning(f"Could not retrieve forces from ASE atoms: {e}")

            try:
                stress = atoms.get_stress(voigt=False)
            except Exception:
                try:
                    stress = atoms.get_stress()
                except Exception as e:
                    logger.warning(f"Could not retrieve stress from ASE atoms: {e}")

        # Fallback to arrays/info
        if energy is None:
            energy = atoms.info.get("energy")
        if forces is None:
            forces = atoms.arrays.get("forces")
        if stress is None:
            stress = atoms.info.get("stress")

        return energy, forces, stress

    def to_ase(self) -> Atoms:
        """
        Convert to ASE Atoms object with SinglePointCalculator for labels.
        """
        atoms = Atoms(
            numbers=self.atomic_numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
        )
        if self.tags:
            atoms.info.update(self.tags)

        if self.uncertainty is not None:
            atoms.info["uncertainty"] = self.uncertainty

        # Attach labels via SinglePointCalculator if present
        if self.energy is not None or self.forces is not None or self.stress is not None:
            calc = SinglePointCalculator(
                atoms,
                energy=self.energy,
                forces=self.forces,
                stress=self.stress,
            )
            atoms.calc = calc

        return atoms
