"""Domain models for atomic structures and candidates."""

from typing import Any, Literal

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Structure(BaseModel):
    """Pydantic model representing an atomic structure."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    formatted_formula: str = Field(description="Chemical formula (e.g., 'H2O')")
    positions: list[list[float]] = Field(description="Atomic positions (N, 3)")
    numbers: list[int] = Field(description="Atomic numbers")
    cell: list[list[float]] = Field(description="Unit cell vectors (3, 3)")
    pbc: list[bool] = Field(description="Periodic boundary conditions (3,)")

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
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc
        )

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        """Create from ASE Atoms object."""
        return cls(
            formatted_formula=atoms.get_chemical_formula(),  # type: ignore[no-untyped-call]
            positions=atoms.get_positions().tolist(),  # type: ignore[no-untyped-call]
            numbers=atoms.get_atomic_numbers().tolist(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell().tolist(),  # type: ignore[no-untyped-call]
            pbc=atoms.get_pbc().tolist()  # type: ignore[no-untyped-call]
        )


class Candidate(Structure):
    """A structure candidate for exploration/training."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Source of the candidate (e.g., 'random', 'md_halt')")
    priority: float = Field(default=0.0, description="Priority for selection")
    status: Literal["PENDING", "SELECTED", "CALCULATED", "FAILED", "TRAINING"] = "PENDING"
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
