from typing import Literal, Optional
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, field_validator

from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.structure import Structure


class LammpsConfig(BaseModel):
    """Configuration for the LAMMPS executable and runtime environment."""

    model_config = ConfigDict(extra="forbid")

    command: str = "lmp"
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"

    # File Naming Conventions (LAMMPS specific)
    dump_file: str = "dump.lammpstrj"
    log_file: str = "log.lammps"
    input_file: str = "in.lammps"
    data_file: str = "data.lammps"
    stdout_file: str = "stdout.log"
    stderr_file: str = "stderr.log"

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """
        Validate LAMMPS command to prevent shell injection or execution of arbitrary binaries.
        Allowlist approach:
        - Must start with lmp, lammps, or be a specific safe path.
        - OR match a safe pattern (alphanumeric, underscores, hyphens).
        """
        # Allow standard lammps executable names
        allowed_prefixes = ["lmp", "lammps", "mpirun", "srun"]
        if any(v.startswith(prefix) for prefix in allowed_prefixes):
            return v

        # Allow absolute paths if they look safe (no shell metas)
        # This regex allows alphanumeric, /, _, -, .
        if re.match(r"^[\w/\.\-]+$", v):
            return v

        raise ValueError(f"LAMMPS command '{v}' is not in allowlist or contains unsafe characters.")


class MDConfig(BaseModel):
    """Configuration for MD simulation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 300.0
    pressure: Optional[float] = None
    n_steps: int = 1000
    timestep: float = 0.001
    ensemble: Literal["NVT", "NPT"] = "NVT"

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
