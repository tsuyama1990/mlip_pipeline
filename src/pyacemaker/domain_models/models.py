"""Domain models for PYACEMAKER."""

import math
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class StructureStatus(str, Enum):
    """Status of a structure in the active learning cycle."""

    NEW = "NEW"
    CALCULATED = "CALCULATED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class MaterialDNA(BaseModel):
    """Material DNA features (composition, symmetry, etc.)."""

    model_config = ConfigDict(extra="forbid")

    composition: dict[str, float] = Field(
        default_factory=dict, description="Elemental composition (e.g., {'Fe': 0.8, 'C': 0.2})"
    )
    average_valence_electrons: float | None = Field(
        default=None, description="Average number of valence electrons"
    )
    crystal_system: str | None = Field(default=None, description="Crystal system (e.g., cubic)")
    space_group: str | None = Field(default=None, description="Space group symbol (e.g., Fm-3m)")


class PredictedProperties(BaseModel):
    """Predicted properties from universal potentials or simple models."""

    model_config = ConfigDict(extra="forbid")

    band_gap: float | None = Field(default=None, description="Predicted band gap (eV)")
    melting_point: float | None = Field(default=None, description="Predicted melting point (K)")
    bulk_modulus: float | None = Field(default=None, description="Predicted bulk modulus (GPa)")


class UncertaintyState(BaseModel):
    """Uncertainty metrics (extrapolation grade)."""

    model_config = ConfigDict(extra="forbid")

    gamma_mean: float | None = Field(default=None, description="Mean extrapolation grade")
    gamma_variance: float | None = Field(
        default=None, description="Variance of extrapolation grade"
    )
    gamma_max: float | None = Field(default=None, description="Maximum extrapolation grade")


class StructureMetadata(BaseModel):
    """Metadata for a structure."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the structure")

    # Core features
    material_dna: MaterialDNA | None = Field(default=None, description="Material DNA features")
    predicted_properties: PredictedProperties | None = Field(
        default=None, description="Predicted physical properties"
    )
    uncertainty_state: UncertaintyState | None = Field(
        default=None, description="Uncertainty metrics"
    )

    # Calculation results (Oracle/DFT output)
    energy: float | None = Field(default=None, description="Total potential energy (eV)")
    forces: list[list[float]] | None = Field(
        default=None, description="Atomic forces (eV/A) as Nx3 list"
    )
    stress: list[float] | None = Field(
        default=None, description="Stress tensor (Voigt notation, eV/A^3) as 6-element list"
    )

    # Legacy/Flexible storage (e.g., for ASE Atoms object which is not Pydantic-serializable)
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extracted features or raw objects (e.g. atoms)",
    )

    tags: list[str] = Field(
        default_factory=list, description="Tags (e.g., 'initial', 'high_uncertainty')"
    )
    status: StructureStatus = Field(
        default=StructureStatus.NEW, description="Processing status of the structure"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=utc_now, description="Last update timestamp")

    @field_validator("energy")
    @classmethod
    def validate_energy(cls, v: float | None) -> float | None:
        """Validate energy is finite and within physical bounds."""
        if v is not None:
            if not math.isfinite(v):
                msg = "Energy must be a finite number"
                raise ValueError(msg)
            # Rough bound check (e.g. per atom energy shouldn't be insanely low/high)
            # Assuming total energy, this is harder, but let's prevent abs(E) > 1e6 eV which implies singularity
            if abs(v) > 1e6:
                msg = f"Energy value {v} is physically implausible (> 1e6 eV)"
                raise ValueError(msg)
        return v

    @field_validator("forces", "stress")
    @classmethod
    def validate_tensor_values(
        cls, v: list[list[float]] | list[float] | None
    ) -> list[list[float]] | list[float] | None:
        """Validate tensor values are finite and within physical bounds."""
        if v is None:
            return v

        # Flatten logic to handle both nested lists (forces) and flat lists (stress)
        flattened: list[float] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)

        if not all(math.isfinite(x) for x in flattened):
            msg = "Forces and stress must contain finite numbers"
            raise ValueError(msg)

        # Physical plausibility check (e.g. Force < 1000 eV/A implies core overlap)
        if any(abs(x) > 1000.0 for x in flattened):
            msg = "Forces/Stress values contain physically implausible magnitudes (> 1000)"
            raise ValueError(msg)

        return v

    @model_validator(mode="after")
    def validate_calculated_fields(self) -> "StructureMetadata":
        """Ensure energy and forces are present if status is CALCULATED."""
        if self.status == StructureStatus.CALCULATED:
            if self.energy is None:
                msg = "Energy must be present when status is CALCULATED"
                raise ValueError(msg)
            if self.forces is None:
                msg = "Forces must be present when status is CALCULATED"
                raise ValueError(msg)
        return self


class PotentialType(str, Enum):
    """Type of interatomic potential."""

    PACE = "PACE"
    M3GNET = "M3GNET"
    LJ = "LJ"
    EAM = "EAM"


class Potential(BaseModel):
    """Representation of an interatomic potential."""

    model_config = ConfigDict(extra="forbid")

    path: Path = Field(..., description="Path to the potential file")
    type: PotentialType = Field(..., description="Type of the potential")
    version: str = Field(..., description="Version identifier")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used to generate this potential"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Validation metrics (RMSE, etc.)"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")

    @field_validator("path")
    @classmethod
    def validate_path_format(cls, v: Path) -> Path:
        """Validate path format."""
        if str(v).strip() == "":
            msg = "Path cannot be empty"
            raise ValueError(msg)
        return v


class TaskType(str, Enum):
    """Type of computational task."""

    DFT = "DFT"
    TRAINING = "TRAINING"
    MD = "MD"
    KMC = "KMC"


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Task(BaseModel):
    """Representation of a computational task."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the task")
    type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    result: dict[str, Any] = Field(
        default_factory=dict, description="Result data (metrics, artifacts)"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
    started_at: datetime | None = Field(default=None, description="Start timestamp")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Task":
        """Validate temporal consistency."""
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            msg = "Completion time cannot be before start time"
            raise ValueError(msg)
        return self


class CycleStatus(str, Enum):
    """Status of the active learning cycle."""

    EXPLORATION = "EXPLORATION"
    LABELING = "LABELING"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    CONVERGED = "CONVERGED"
    FAILED = "FAILED"


class ActiveSet(BaseModel):
    """Collection of structures selected for active learning."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the active set")
    structure_ids: list[UUID] = Field(..., description="List of structure IDs in this set")
    selection_criteria: str = Field(
        ..., description="Description of selection criteria (e.g., 'max_uncertainty')"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")
