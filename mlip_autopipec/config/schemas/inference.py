import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, ValidationInfo, field_validator


class MDConfig(BaseModel):
    ensemble: Literal["nvt", "npt"] = "nvt"
    temperature: float = Field(300.0, gt=0)
    timestep: float = Field(1.0, gt=0)
    run_duration: int = Field(1000, gt=0)
    model_config = ConfigDict(extra="forbid")

class UncertaintyConfig(BaseModel):
    threshold: float = Field(5.0, gt=0)
    embedding_cutoff: float = Field(8.0, gt=0)
    masking_cutoff: float = Field(5.0, gt=0)
    model_config = ConfigDict(extra="forbid")

    @field_validator("masking_cutoff")
    @classmethod
    def masking_must_be_less_than_embedding(cls, masking_cutoff: float, info: ValidationInfo) -> float:
        if "embedding_cutoff" in info.data and masking_cutoff >= info.data["embedding_cutoff"]:
            raise ValueError("masking_cutoff must be smaller than embedding_cutoff.")
        return masking_cutoff

class InferenceConfig(BaseModel):
    lammps_executable: FilePath | None = None
    potential_path: FilePath | None = None
    md_params: MDConfig = Field(default_factory=MDConfig)
    uncertainty_params: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    model_config = ConfigDict(extra="forbid")

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: FilePath | None) -> FilePath | None:
        if v is not None:
            if not os.access(v, os.X_OK):
                raise ValueError(f"File at {v} is not executable.")
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
