from typing import Literal, Optional
from pathlib import Path
import re
import shutil

from pydantic import BaseModel, ConfigDict, field_validator

from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec import defaults


class LammpsConfig(BaseModel):
    """Configuration for the LAMMPS executable and runtime environment."""

    model_config = ConfigDict(extra="forbid")

    command: str = defaults.DEFAULT_LAMMPS_COMMAND
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"

    # File Naming Conventions (LAMMPS specific)
    dump_file: str = defaults.DEFAULT_TRAJ_FILE_LAMMPS
    log_file: str = defaults.DEFAULT_LOG_FILE_LAMMPS
    input_file: str = defaults.DEFAULT_INPUT_FILE_LAMMPS
    data_file: str = defaults.DEFAULT_DATA_FILE_LAMMPS
    stdout_file: str = defaults.DEFAULT_STDOUT_FILE
    stderr_file: str = defaults.DEFAULT_STDERR_FILE

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """
        Validate LAMMPS command to prevent shell injection or execution of arbitrary binaries.
        Allowlist approach:
        - Must start with lmp, lammps, or be a specific safe path.
        - OR match a safe pattern (alphanumeric, underscores, hyphens, forward slashes).
        - Explicitly disallow '..' to prevent directory traversal.
        """
        # Explicit check for directory traversal
        if ".." in v:
            raise ValueError(f"LAMMPS command '{v}' contains forbidden directory traversal '..'")

        # Allow standard lammps executable names
        if any(v.startswith(prefix) for prefix in defaults.DEFAULT_ALLOWED_COMMAND_PREFIXES):
            return v

        # Allow absolute paths if they look safe (no shell metas)
        # This regex allows alphanumeric, /, _, -, .
        # Stricter: disallow '..' usage even if path valid characters (already checked above but reinforcing)
        if re.match(r"^[\w/\.\-]+$", v) and ".." not in v:
            # If path is absolute, check if executable exists and is executable
            path = Path(v)
            if path.is_absolute():
                # We use shutil.which for robust executability check
                if not shutil.which(v):
                     raise ValueError(f"Executable not found or not executable at absolute path: {v}")

                # Double check with os.access for explicit R_OK | X_OK if possible,
                # but shutil.which is standard.
                import os
                if not (os.path.isfile(v) and os.access(v, os.X_OK)):
                    raise ValueError(f"Path {v} points to a non-executable file.")
            return v

        raise ValueError(f"LAMMPS command '{v}' is not in allowlist or contains unsafe characters.")


class MDConfig(BaseModel):
    """Configuration for MD simulation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = defaults.DEFAULT_MD_TEMP
    pressure: Optional[float] = None
    n_steps: int = defaults.DEFAULT_MD_STEPS
    timestep: float = defaults.DEFAULT_MD_TIMESTEP
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


class EonConfig(BaseModel):
    """Configuration for EON (Adaptive KMC)."""

    model_config = ConfigDict(extra="forbid")

    command: str = "eon"
    timeout: int = 86400  # 24 hours (KMC is slow)
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"

    # File Naming Conventions
    config_file: str = "config.ini"
    stdout_file: str = defaults.DEFAULT_STDOUT_FILE
    stderr_file: str = defaults.DEFAULT_STDERR_FILE

    # Paths
    driver_path: Optional[Path] = None

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Reuse LAMMPS command validation logic for EON."""
        return LammpsConfig.validate_command(v)


class EonResult(JobResult):
    """
    Result of an EON aKMC simulation.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    final_structure: Optional[Structure] = None # Saddle point or high-gamma state
    max_gamma: Optional[float] = None
