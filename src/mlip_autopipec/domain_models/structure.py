from collections.abc import Iterator
from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Structure(BaseModel):
    """
    Represents an atomic structure with associated properties and metadata.
    """

    atoms: Atoms = Field(..., description="ASE Atoms object representing the structure")
    provenance: str = Field(..., description="Source of the structure (e.g., 'random', 'md_halt')")
    uncertainty_score: float | None = Field(
        default=None, description="Uncertainty metric from the model"
    )
    label_status: str = Field(
        default="unlabeled", description="Status of ground truth availability (unlabeled, labeled, failed)"
    )
    energy: float | None = Field(default=None, description="Total energy (eV)")
    forces: list[list[float]] | None = Field(default=None, description="Forces on atoms (eV/Angstrom)")
    stress: list[float] | None = Field(default=None, description="Stress tensor (Voigt notation, eV/Angstrom^3 or kBar)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("atoms", mode="before")
    @classmethod
    def validate_atoms(cls, v: Any) -> Atoms:
        if not isinstance(v, Atoms):
            msg = f"Must be an ase.Atoms object, got {type(v)}"
            raise TypeError(msg)
        return v

    @property
    def ase_atoms(self) -> Atoms:
        """Returns the atoms object cast to ase.Atoms for type safety."""
        return self.atoms

    def to_ase(self) -> Atoms:
        """
        Returns a copy of the internal ase.Atoms object with updated info.

        This method creates a deep copy of the underlying ASE Atoms object
        and populates its `.info` dictionary with metadata from the Structure
        instance (provenance, label_status, uncertainty_score).
        This ensures the returned Atoms object carries all relevant context.
        """
        # Pydantic guarantees self.atoms is an Atoms object
        atoms: Atoms = self.atoms.copy()  # type: ignore[no-untyped-call]

        info_update: dict[str, Any] = {
            "provenance": self.provenance,
            "label_status": self.label_status,
        }

        if self.uncertainty_score is not None:
             info_update["uncertainty_score"] = self.uncertainty_score

        atoms.info.update(info_update)
        return atoms


class Trajectory(BaseModel):
    """
    A sequence of Structures representing a simulation trajectory.
    """
    structures: list[Structure] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __iter__(self) -> Iterator[Structure]: # type: ignore[override]
        return iter(self.structures)

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, item: int) -> Structure:
        return self.structures[item]


class HaltInfo(BaseModel):
    """
    Information about a simulation halt event due to uncertainty or error.
    """
    step: int = Field(..., ge=0, description="MD/kMC step where halt occurred")
    max_gamma: float = Field(..., ge=0, description="Max extrapolation grade detected")
    structure: Structure = Field(..., description="The structure snapshot at halt")
    reason: str = Field(default="uncertainty_threshold", description="Reason for halt")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
