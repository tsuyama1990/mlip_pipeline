from typing import Literal, Optional
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, field_validator

from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.structure import Structure


class LammpsConfig(BaseModel):
    """Configuration for the LAMMPS executable and runtime environment."""

    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """
        Validate LAMMPS command to prevent shell injection or execution of arbitrary binaries.
        Strict Allowlist approach:
        - Must match a safe pattern (alphanumeric, underscores, hyphens, periods, forward slashes).
        - Explicitly reject '..' to prevent path traversal.
        """
        # Reject path traversal
        if ".." in v:
             raise ValueError(f"LAMMPS command '{v}' contains illegal path traversal '..'")

        # Strict regex enforcement
        # This regex allows alphanumeric, /, _, -, .
        # It ensures the command does not contain shell metacharacters like ; | & etc.
        if not re.match(r"^[\w/\.\-]+$", v):
            raise ValueError(f"LAMMPS command '{v}' contains unsafe characters.")

        return v


class MDConfig(BaseModel):
    """Configuration for MD simulation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    pressure: Optional[float] = None
    n_steps: int
    timestep: float = 0.001
    ensemble: Literal["NVT", "NPT"]

    # Uncertainty Quantification (UQ)
    uncertainty_threshold: Optional[float] = None

    @field_validator("n_steps")
    @classmethod
    def validate_n_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_steps must be positive")
        return v


# Alias for backward compatibility
MDParams = MDConfig


class LammpsResult(JobResult):
    """
    Result of a LAMMPS MD simulation.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    final_structure: Structure
    trajectory_path: Path
    max_gamma: Optional[float] = None
