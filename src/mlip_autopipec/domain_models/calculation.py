from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.job import JobResult


def _default_scf_strategies() -> list[dict[str, Any]]:
    return [
        {"mixing_beta": 0.3},
        {"mixing_beta": 0.1},
        {"smearing": "mv", "degauss": 0.02},
    ]


def _default_memory_strategies() -> list[dict[str, Any]]:
    return [
        {"diagonalization": "cg"},
        {"mixing_ndim": 4},
    ]


def _default_walltime_strategies() -> list[dict[str, Any]]:
    return [
        {"diagonalization": "cg"},
        {"conv_thr": 1e-5},
    ]


class RecoveryConfig(BaseModel):
    """Configuration for DFT recovery strategies."""

    model_config = ConfigDict(extra="forbid")

    scf_strategies: list[dict[str, Any]] = Field(
        default_factory=_default_scf_strategies
    )
    memory_strategies: list[dict[str, Any]] = Field(
        default_factory=_default_memory_strategies
    )
    walltime_strategies: list[dict[str, Any]] = Field(
        default_factory=_default_walltime_strategies
    )


class DFTConfig(BaseModel):
    """
    Configuration for Density Functional Theory (DFT) calculations.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = "mpirun -np 16 pw.x"
    pseudopotentials: dict[str, Path]
    ecutwfc: float
    kspacing: float = 0.04
    timeout: int = 3600  # Default timeout in seconds
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)


class DFTResult(JobResult):
    """
    Result of a DFT calculation.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    energy: float
    forces: np.ndarray
    stress: Optional[np.ndarray] = None
    magmoms: Optional[np.ndarray] = None


class DFTError(Exception):
    """Base class for DFT calculation errors."""

    pass


class SCFError(DFTError):
    """Raised when SCF cycle fails to converge."""

    pass


class MemoryError(DFTError):
    """Raised when the calculation runs out of memory."""

    pass


class WalltimeError(DFTError):
    """Raised when the calculation exceeds the time limit."""

    pass
