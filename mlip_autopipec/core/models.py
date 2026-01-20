"""
Core data models for MLIP-AutoPipe.
"""
from typing import Any, Self

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DFTResult(BaseModel):
    """
    Standardized result from a DFT calculation.
    """
    atoms: Any = Field(..., description="The final atomic structure (ase.Atoms).")
    energy: float = Field(..., description="The total energy in eV.")
    forces: Any = Field(..., description="The Hellmann-Feynman forces (Nx3 array).")
    stress: Any = Field(..., description="The virial stress tensor.")

    model_config = ConfigDict(extra="forbid") # arbitrary_types_allowed=False by default

    @field_validator("atoms")
    @classmethod
    def validate_atoms_type(cls, v: Any) -> Any:
        """Ensures atoms is an ase.Atoms object."""
        if not isinstance(v, Atoms):
            msg = f"Expected ase.Atoms object, got {type(v)}"
            raise TypeError(msg)
        return v

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: Any) -> Any:
        """Ensures forces array matches atoms length."""
        if not isinstance(v, (np.ndarray, list)):
             msg = "Forces must be a numpy array or list."
             raise TypeError(msg)
        return np.array(v)

    @field_validator("stress")
    @classmethod
    def validate_stress(cls, v: Any) -> Any:
        """Ensures stress is a numpy array."""
        if not isinstance(v, (np.ndarray, list)):
            msg = "Stress must be a numpy array or list."
            raise TypeError(msg)
        return np.array(v)

    @model_validator(mode='after')
    def validate_consistency(self) -> Self:
        """Post-initialization validation."""
        # Check forces shape against atoms length
        if len(self.forces) != len(self.atoms):
            msg = f"Forces shape {self.forces.shape} does not match atoms length {len(self.atoms)}"
            raise ValueError(msg)
        if self.forces.shape[1] != 3:
             msg = f"Forces must have 3 components, got {self.forces.shape[1]}"
             raise ValueError(msg)
        return self
