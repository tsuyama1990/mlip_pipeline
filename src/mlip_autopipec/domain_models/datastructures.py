from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import TaskType


def validate_path_safety(v: Path | None) -> Path | None:
    """
    Validates that a path is safe (no traversal).

    This function checks if the path contains '..' components in its parts,
    which would indicate a directory traversal attempt.
    Absolute paths are allowed (as users may specify absolute paths for inputs),
    but relative paths must not attempt to go up the directory tree.
    """
    if v is None:
        return None

    # Check for explicit traversal components
    if ".." in v.parts:
        msg = f"Path traversal detected in {v}"
        raise ValueError(msg)

    return v


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

    @field_validator("atoms")
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
        """Returns a copy of the internal ase.Atoms object with updated info."""
        # Validate that atoms is definitely an Atoms object before copying
        if not isinstance(self.ase_atoms, Atoms):
            msg = f"Internal atoms object is not of type ase.Atoms, got {type(self.ase_atoms)}"
            raise TypeError(msg)

        # Use copy() to prevent side effects on the persistent data model
        # Cast return type of copy() to Atoms since ase doesn't type hint it well
        atoms: Atoms = self.ase_atoms.copy()  # type: ignore[no-untyped-call]

        # Validate critical fields before updating atoms.info
        if not isinstance(self.provenance, str):
            msg = f"Provenance must be a string, got {type(self.provenance)}"
            raise TypeError(msg)

        if not isinstance(self.label_status, str):
            msg = f"Label status must be a string, got {type(self.label_status)}"
            raise TypeError(msg)

        # Basic type checks for optional fields if present are handled by Pydantic on model init
        # but we are updating atoms.info here.

        info_update: dict[str, Any] = {
            "provenance": self.provenance,
            "label_status": self.label_status,
        }

        if self.uncertainty_score is not None:
             info_update["uncertainty_score"] = self.uncertainty_score

        atoms.info.update(info_update)
        return atoms


class Dataset(BaseModel):
    """
    A collection of Structures representing a training dataset.
    """
    structures: list[Structure] = Field(default_factory=list)
    description: str = Field(default="", description="Description of the dataset")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __len__(self) -> int:
        return len(self.structures)

    def add(self, structure: Structure) -> None:
        self.structures.append(structure)


class Potential(BaseModel):
    """
    Represents a trained MLIP potential file.
    """
    path: Path = Field(..., description="Path to the potential file (.yace, .pot, etc.)")
    format: str = Field(..., description="Format of the potential (e.g., 'yace')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters")

    model_config = ConfigDict(extra="forbid")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        validate_path_safety(v)
        return v


class WorkflowState(BaseModel):
    """
    Captures the current state of the Orchestrator workflow for persistence.
    """
    current_cycle: int = Field(default=0, ge=0)
    current_step: TaskType = Field(default=TaskType.EXPLORATION)
    active_potential_path: Path | None = None
    dataset_path: Path | None = None
    iteration: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("active_potential_path", "dataset_path")
    @classmethod
    def validate_paths(cls, v: Path | None) -> Path | None:
        return validate_path_safety(v)


class ValidationResult(BaseModel):
    """
    Result of a validation run.
    """
    passed: bool = Field(..., description="Whether the validation passed")
    metrics: dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    report_path: Path | None = Field(None, description="Path to detailed report")

    model_config = ConfigDict(extra="forbid")

    @field_validator("report_path")
    @classmethod
    def validate_path(cls, v: Path | None) -> Path | None:
        return validate_path_safety(v)


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
    max_gamma: float = Field(..., ge=0.0, description="Max extrapolation grade detected")
    structure: Structure = Field(..., description="The structure snapshot at halt")
    reason: str = Field(default="uncertainty_threshold", description="Reason for halt")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
