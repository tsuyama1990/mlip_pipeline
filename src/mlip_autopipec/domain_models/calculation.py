from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTConfig(BaseModel):
    """Configuration for DFT calculations."""

    model_config = ConfigDict(extra="forbid")

    command: str
    pseudopotentials: dict[str, Path]
    ecutwfc: float
    kspacing: float
    # Optional parameters for self-healing and detailed control
    mixing_beta: float = 0.7
    smearing: str = "mv"
    degauss: float = 0.02
    diagonalization: str = "david"


class DFTResult(BaseModel):
    """Result of a DFT calculation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    energy: float = Field(..., description="Total energy in eV")
    forces: NDArray[np.float64] = Field(..., description="Forces in eV/A")
    stress: Optional[NDArray[np.float64]] = Field(
        None, description="Stress tensor in eV/A^3"
    )
    magmoms: Optional[NDArray[np.float64]] = Field(None, description="Magnetic moments")

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Forces must have shape (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("stress")
    @classmethod
    def validate_stress_shape(
        cls, v: Optional[NDArray[np.float64]]
    ) -> Optional[NDArray[np.float64]]:
        if v is not None:
            if v.shape != (3, 3) and v.shape != (6,):
                # Voigt notation (6,) is sometimes used, but SPEC says (3,3).
                # Let's enforce (3,3) as per SPEC: "NDArray[(3, 3), float]"
                # But ASE often returns (6,) for stress.
                # I will allow (3,3) for now as per SPEC.
                if v.shape == (6,):
                    # Option: Auto-convert or fail?
                    # Failing is safer for strict Schema compliance.
                    pass
                if v.shape != (3, 3):
                    msg = f"Stress must be a 3x3 matrix, got {v.shape}"
                    raise ValueError(msg)
        return v


class DFTError(Exception):
    """Base exception for DFT failures."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}


class SCFError(DFTError):
    """Raised when SCF cycle fails to converge."""

    pass


class MemoryError(DFTError):
    """Raised when calculation runs out of memory."""

    pass


class WalltimeError(DFTError):
    """Raised when calculation hits walltime limit."""

    pass
