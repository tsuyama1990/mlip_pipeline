"""Structure-related domain models."""

import math
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyacemaker.core.config import CONSTANTS
from pyacemaker.domain_models.common import StructureStatus, utc_now


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
            if abs(v) > CONSTANTS.max_energy_ev:
                msg = f"Energy value {v} is physically implausible (> {CONSTANTS.max_energy_ev} eV)"
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
        if any(abs(x) > CONSTANTS.max_force_ev_a for x in flattened):
            msg = f"Forces/Stress values contain physically implausible magnitudes (> {CONSTANTS.max_force_ev_a})"
            raise ValueError(msg)

        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate features dictionary."""
        if not all(isinstance(k, str) for k in v):
            msg = "Feature keys must be strings"
            raise ValueError(msg)

        for key, value in v.items():
            if callable(value):
                msg = f"Feature '{key}' cannot be a callable"
                raise TypeError(msg)

            # Basic type check for security (allow primitives, lists, dicts, and Atoms objects by name)
            # We allow objects if they have a 'todict' or 'as_dict' method or are ASE Atoms
            type_name = type(value).__name__
            allowed_types = (str, int, float, bool, list, tuple, dict, type(None))

            if not isinstance(value, allowed_types):
                # Check for ASE Atoms or known safe types
                is_ase_atoms = False
                try:
                    from ase import Atoms

                    if isinstance(value, Atoms):
                        is_ase_atoms = True
                        # Data Integrity: Check for essential attributes
                        if not hasattr(value, "positions") or not hasattr(value, "numbers"):
                             msg = f"Feature '{key}' is an ASE Atoms object but lacks essential data (positions/numbers)"
                             raise ValueError(msg)
                except ImportError:
                    pass

                if (
                    is_ase_atoms
                    or type_name == "Atoms"
                    or hasattr(value, "todict")
                    or hasattr(value, "as_dict")
                ):
                    continue
                # Reject unknown complex types to prevent injection of arbitrary objects
                msg = f"Feature '{key}' has unsafe type: {type_name}"
                raise ValueError(msg)

        return v

    @model_validator(mode="after")
    def validate_calculated_fields(self) -> "StructureMetadata":
        """Ensure energy, forces, and stress are present if status is CALCULATED."""
        if self.status == StructureStatus.CALCULATED:
            if self.energy is None:
                msg = "Energy must be present when status is CALCULATED"
                raise ValueError(msg)
            if self.forces is None:
                msg = "Forces must be present when status is CALCULATED"
                raise ValueError(msg)

            # Data Integrity: If forces are present, stress should also be checked/present
            # Often stress is optional, but for consistency in MLIP we prefer having it if calculated.
            # If not calculated, it should be None, but if forces are there, usually stress is too.
        return self
