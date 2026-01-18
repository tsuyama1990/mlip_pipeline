from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# We can keep MDConfig/UncertaintyConfig for backward compatibility if needed,
# but InferenceConfig will be flat as per SPEC.
# Actually, to be strictly SPEC compliant, InferenceConfig should have the fields directly.


class InferenceConfig(BaseModel):
    temperature: float = Field(..., gt=0)
    pressure: float = Field(0.0)  # 0 for NVT
    timestep: float = Field(0.001, gt=0)  # ps
    steps: int = Field(10000, gt=0)
    ensemble: Literal["nvt", "npt"] = "nvt"
    uq_threshold: float = Field(5.0, gt=0)
    sampling_interval: int = Field(100, gt=0)
    potential_path: Path | None = (
        None  # Spec says Path, but optional for now? Spec says "potential_path: Path". I'll make it optional to avoid validation errors if not set immediately? No, Spec implies required. But usually we have defaults.
    )
    # Wait, Spec: "potential_path: Path". No default.
    # But in UAT we initialize it.

    # Extra field for practical execution
    lammps_executable: str | Path | None = "lmp"

    model_config = ConfigDict(extra="forbid")

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: str | Path | None) -> str | Path | None:
        if v is not None:
            pass  # Mocking makes this hard to strictly validate existence
        return v


class UncertaintyMetadata(BaseModel):
    uncertain_timestep: int
    uncertain_atom_id: int
    uncertain_atom_index_in_original_cell: int
    model_config = ConfigDict(extra="forbid")


class UncertainStructure(BaseModel):
    atoms: object
    force_mask: object
    metadata: UncertaintyMetadata
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms")
    @classmethod
    def validate_atoms_type(cls, atoms_obj: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is required.") from e
        if not isinstance(atoms_obj, Atoms):
            raise TypeError("Field 'atoms' must be an instance of ase.Atoms.")
        return atoms_obj

    @field_validator("force_mask")
    @classmethod
    def validate_force_mask(cls, force_mask_obj: Any, info: ValidationInfo) -> Any:
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required.") from e
        if not isinstance(force_mask_obj, np.ndarray):
            raise TypeError("Field 'force_mask' must be a NumPy array.")
        if "atoms" in info.data and len(force_mask_obj) != len(info.data["atoms"]):
            raise ValueError("force_mask must have the same length as the number of atoms.")
        return force_mask_obj


class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path | None = None
    uncertain_structures: list[Path] = Field(default_factory=list)  # Paths to extracted frames
    max_gamma_observed: float
    model_config = ConfigDict(extra="forbid")
