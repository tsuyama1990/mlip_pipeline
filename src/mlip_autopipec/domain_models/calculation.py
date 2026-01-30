from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from mlip_autopipec.domain_models.job import JobResult


class DFTConfig(BaseModel):
    """
    Configuration for a Density Functional Theory (DFT) calculation.
    Targeting Quantum Espresso.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = "pw.x"
    mpi_command: str = "mpirun -np 1"
    pseudopotentials: Dict[str, Path]
    ecutwfc: float
    kspacing: float

    # SCF / Electronic parameters
    smearing: str = "gaussian"
    degauss: float = 0.02
    mixing_beta: float = 0.7

    timeout: int = 3600  # seconds

    @field_validator("ecutwfc", "kspacing", "degauss", "mixing_beta")
    @classmethod
    def validate_positive(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("mixing_beta")
    @classmethod
    def validate_beta(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError("mixing_beta must be <= 1.0")
        return v


class DFTResult(JobResult):
    """
    Result of a DFT calculation.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    energy: float  # eV
    forces: np.ndarray  # (N, 3) eV/A
    stress: Optional[np.ndarray] = None  # (3, 3) or (6,) eV/A^3
    magmoms: Optional[np.ndarray] = None  # (N,)

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"Forces must have shape (N, 3), got {v.shape}")
        return v


class DFTError(Exception):
    """Base class for DFT calculation errors."""

    pass


class SCFError(DFTError):
    """SCF convergence failure."""

    pass


class MemoryError(DFTError):
    """Insufficient memory."""

    pass


class WalltimeError(DFTError):
    """Timeout exceeded."""

    pass
