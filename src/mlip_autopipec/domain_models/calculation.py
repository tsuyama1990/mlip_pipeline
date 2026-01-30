from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.job import JobResult


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
