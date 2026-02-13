from pathlib import Path
from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import TaskType


class Structure(BaseModel):
    # Use object as type hint instead of Any, though Pydantic treats them similarly with arbitrary_types_allowed
    # But Any is discouraged.
    atoms: Atoms
    provenance: str = Field(..., description="Source of the structure (e.g., 'random', 'md_halt')")
    uncertainty_score: float | None = Field(
        default=None, description="Uncertainty metric from the model"
    )
    label_status: str = Field(
        default="unlabeled", description="Status of ground truth availability"
    )
    energy: float | None = Field(default=None, description="Total energy per atom")
    forces: list[list[float]] | None = Field(default=None, description="Forces on atoms")
    stress: list[float] | None = Field(default=None, description="Stress tensor")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("atoms")
    @classmethod
    def validate_atoms(cls, v: Any) -> Any:
        if not isinstance(v, Atoms):
            msg = "Must be an ase.Atoms object"
            raise TypeError(msg)
        return v

    @property
    def ase_atoms(self) -> Atoms:
        """Returns the atoms object cast to ase.Atoms for type safety."""
        return self.atoms

    def to_ase(self) -> Atoms:
        """Returns the internal ase.Atoms object with updated info."""
        # atoms is verified to be Atoms in validate_atoms
        atoms = self.ase_atoms
        atoms.info.update(
            {
                "provenance": self.provenance,
                "uncertainty_score": self.uncertainty_score,
                "label_status": self.label_status,
            }
        )
        return atoms


class Dataset(BaseModel):
    structures: list[Structure] = Field(default_factory=list)
    description: str = Field(default="", description="Description of the dataset")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __len__(self) -> int:
        return len(self.structures)

    def add(self, structure: Structure) -> None:
        self.structures.append(structure)


class Potential(BaseModel):
    path: Path = Field(..., description="Path to the potential file (.yace, .pot, etc.)")
    format: str = Field(..., description="Format of the potential (e.g., 'yace')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters")

    model_config = ConfigDict(extra="forbid")


class WorkflowState(BaseModel):
    current_cycle: int = Field(default=0, ge=0)
    current_step: TaskType = Field(default=TaskType.EXPLORATION)
    active_potential_path: Path | None = None
    dataset_path: Path | None = None
    iteration: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class ValidationResult(BaseModel):
    passed: bool = Field(..., description="Whether the validation passed")
    metrics: dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    report_path: Path | None = Field(None, description="Path to detailed report")

    model_config = ConfigDict(extra="forbid")


class Trajectory(BaseModel):
    structures: list[Structure] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class HaltInfo(BaseModel):
    step: int = Field(..., ge=0, description="MD/kMC step where halt occurred")
    max_gamma: float = Field(..., ge=0.0, description="Max extrapolation grade detected")
    structure: Structure = Field(..., description="The structure snapshot at halt")
    reason: str = Field(default="uncertainty_threshold", description="Reason for halt")

    model_config = ConfigDict(extra="forbid")
