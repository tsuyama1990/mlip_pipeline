from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.job import JobResult


class DFTConfig(BaseModel):
    """
    Configuration for DFT calculations (Quantum Espresso).
    """
    model_config = ConfigDict(extra="forbid")

    command: str = "pw.x"
    mpi_command: str = "mpirun -np 16"
    use_mpi: bool = True

    # Physics parameters
    pseudopotentials: dict[str, Path]
    ecutwfc: float
    kspacing: float = 0.04  # Inverse k-point density

    # Convergence / System parameters
    smearing: str = "mv"
    degauss: float = 0.02
    mixing_beta: float = 0.7
    diagonalization: str = "david"

    # Flags
    tprnfor: bool = True
    tstress: bool = True

    # Runtime
    timeout: int = 3600


class DFTResult(JobResult):
    """
    Result of a DFT calculation.
    """
    energy: float  # eV
    forces: np.ndarray  # eV/A, shape (N, 3)
    stress: Optional[np.ndarray] = None  # eV/A^3, shape (3, 3)
    magmoms: Optional[np.ndarray] = None  # Magnetic moments

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DFTError(Exception):
    """Base class for DFT calculation errors."""
    pass


class SCFError(DFTError):
    """SCF convergence failed."""
    pass


class MemoryError(DFTError):
    """Job ran out of memory."""
    pass


class WalltimeError(DFTError):
    """Job timed out."""
    pass
