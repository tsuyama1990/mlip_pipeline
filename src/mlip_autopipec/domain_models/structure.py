"""Domain models for atomic structures and candidates."""

from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateStatus(str, Enum):
    """Status of a candidate structure in the active learning loop."""
    PENDING = "PENDING"
    SELECTED = "SELECTED"
    CALCULATED = "CALCULATED"
    FAILED = "FAILED"
    TRAINING = "TRAINING"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"


class Structure(BaseModel):
    """Pydantic model representing an atomic structure with properties."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    formatted_formula: str = Field(description="Chemical formula (e.g., 'H2O')")
    positions: list[list[float]] = Field(description="Atomic positions (N, 3)")
    numbers: list[int] = Field(description="Atomic numbers")
    cell: list[list[float]] = Field(description="Unit cell vectors (3, 3)")
    pbc: list[bool] = Field(description="Periodic boundary conditions (3,)")

    # Calculated Properties
    energy: float | None = Field(default=None, description="Potential Energy (eV)")
    forces: list[list[float]] | None = Field(default=None, description="Atomic forces (N, 3)")
    stress: list[float] | None = Field(default=None, description="Voigt stress (6,) or full (3,3)")

    # Uncertainty
    uncertainty: float | None = Field(default=None, description="Max extrapolation grade/uncertainty")

    @field_validator("positions")
    @classmethod
    def validate_positions(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            msg = "Positions cannot be empty"
            raise ValueError(msg)
        return v

    @field_validator("cell")
    @classmethod
    def validate_cell(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            msg = "Cell must be 3x3"
            raise ValueError(msg)
        return v

    def to_ase(self) -> Atoms:
        """Convert to ASE Atoms object."""
        atoms = Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc
        )
        if self.energy is not None:
            atoms.info['energy'] = self.energy
        if self.forces is not None:
            atoms.arrays['forces'] = np.array(self.forces)
        if self.stress is not None:
            atoms.info['stress'] = np.array(self.stress)
        return atoms

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        """Create from ASE Atoms object."""
        # Helper to safely get arrays
        pos = atoms.get_positions()  # type: ignore[no-untyped-call]
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        # Ensure numpy arrays are converted to lists
        positions = pos.tolist() if hasattr(pos, "tolist") else list(pos)

        # cell might be Cell object
        cell_list = cell.tolist() if hasattr(cell, "tolist") else list(cell)

        # properties
        energy = atoms.info.get("energy")
        forces = atoms.arrays.get("forces")
        forces_list = forces.tolist() if forces is not None else None

        stress = atoms.info.get("stress")
        stress_list = stress.tolist() if stress is not None else None

        # pbc might be tuple or array
        pbc_val = atoms.get_pbc()  # type: ignore[no-untyped-call]
        pbc_list = pbc_val.tolist() if hasattr(pbc_val, "tolist") else list(pbc_val)

        return cls(
            formatted_formula=str(atoms.get_chemical_formula()),  # type: ignore[no-untyped-call]
            positions=positions,
            numbers=list(atoms.get_atomic_numbers()),  # type: ignore[no-untyped-call]
            cell=cell_list,
            pbc=pbc_list,
            energy=energy,
            forces=forces_list,
            stress=stress_list
        )


class Candidate(Structure):
    """A structure candidate for exploration/training."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Source of the candidate (e.g., 'random', 'md_halt')")
    priority: float = Field(default=0.0, description="Priority for selection")
    status: CandidateStatus = Field(default=CandidateStatus.PENDING)
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: CandidateStatus) -> CandidateStatus:
        """Explicitly validate status, though Pydantic Enum does this automatically."""
        if not isinstance(v, CandidateStatus):
             # This block might not be reachable if Pydantic catches it first,
             # but explicitly handling it satisfies the requirement.
             msg = f"Invalid status: {v}"
             raise TypeError(msg)
        return v
