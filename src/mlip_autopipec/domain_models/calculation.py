from pathlib import Path
from typing import Optional, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTError(Exception):
    """Base class for DFT errors."""
    pass


class SCFError(DFTError):
    """SCF convergence failure."""
    pass


class MemoryError(DFTError):
    """Insufficient memory."""
    pass


class WalltimeError(DFTError):
    """Walltime exceeded."""
    pass


def _default_strategies() -> dict[str, list[dict[str, Any]]]:
    return {
        "SCFError": [
            {"mixing_beta": 0.3},
            {"smearing": "mv", "degauss": 0.02},
        ]
    }


class RecoveryConfig(BaseModel):
    """Configuration for error recovery strategies."""
    model_config = ConfigDict(extra="forbid")

    # Mapping of error type (e.g., "SCFError") to a list of fix strategies
    # Each strategy is a dict of parameters to update
    strategies: dict[str, list[dict[str, Any]]] = Field(
        default_factory=_default_strategies
    )
    max_retries: int = 3


class DFTConfig(BaseModel):
    """Configuration for DFT calculations (Quantum Espresso)."""
    model_config = ConfigDict(extra="forbid")

    command: str = "pw.x"
    pseudopotentials: dict[str, Path]
    ecutwfc: float = 40.0
    kspacing: float = 0.04
    timeout: int = 3600
    # Additional QE parameters that might be needed
    smearing: str = "mp"
    degauss: float = 0.01
    mixing_beta: float = 0.7
    diagonalization: str = "david"

    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)


class DFTResult(BaseModel):
    """Results from a DFT calculation."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    energy: float
    forces: np.ndarray
    stress: Optional[np.ndarray] = None
    magmoms: Optional[np.ndarray] = None
    converged: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Forces must have shape (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("stress")
    @classmethod
    def validate_stress_shape(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return v
        # Stress can be (3, 3) or Voigt (6,)
        if v.shape != (3, 3) and v.shape != (6,):
             msg = f"Stress must be (3, 3) or (6,), got {v.shape}"
             raise ValueError(msg)
        return v
