import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTConfig(BaseModel):
    """
    Configuration for Density Functional Theory calculations.
    """

    # Required fields defined without default values
    pseudopotential_dir: Path = Field(description="Directory containing .UPF files")
    ecutwfc: float = Field(gt=0, description="Wavefunction cutoff energy (Ry)")
    kspacing: float = Field(gt=0, description="Inverse K-point density (1/A)")

    # Optional fields with defaults
    command: str = Field(
        "pw.x", description="Command to run Quantum Espresso (e.g. 'mpirun -np 4 pw.x')"
    )
    nspin: int = Field(1, description="Spin polarization (1=off, 2=on)")
    mixing_beta: float = Field(0.7, ge=0.0, le=1.0, description="SCF Mixing parameter (0.0-1.0)")
    diagonalization: Literal["david", "cg"] = Field("david", description="Solver ('david', 'cg')")
    smearing: Literal["mv", "mp", "fd"] = Field(
        "mv", description="Smearing type ('mv', 'mp', 'fd')"
    )
    degauss: float = Field(0.02, gt=0, description="Smearing width (Ry)")
    recoverable: bool = Field(True, description="Enable auto-recovery")
    max_retries: int = Field(5, ge=0, description="Maximum number of retries")
    timeout: float = Field(3600.0, gt=0, description="Timeout in seconds")

    model_config = ConfigDict(extra="forbid")

    @field_validator("pseudopotential_dir")
    @classmethod
    def validate_pseudo_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Pseudopotential directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Pseudopotential path is not a directory: {v}")

        # Check for UPF files
        upf_files = list(v.glob("*.UPF")) + list(v.glob("*.upf"))
        if not upf_files:
            raise ValueError(f"No .UPF or .upf files found in pseudopotential directory: {v}")

        return v

    @field_validator("nspin")
    @classmethod
    def validate_nspin(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("nspin must be 1 or 2.")
        return v

    @field_validator("command")
    @classmethod
    def validate_command_security(cls, v: str) -> str:
        """
        Security check to prevent shell injection.
        Disallows unsafe characters and validates splitting with shlex.
        """
        # Strict check for shell control operators
        if re.search(r"[;&|`$()]", v):
            raise ValueError(
                "Command contains unsafe shell characters: ; & | ` $ ( ). Execution is blocked."
            )

        try:
            parts = shlex.split(v)
            if not parts:
                raise ValueError("Command is empty.")
        except ValueError as e:
            raise ValueError(f"Command string is invalid according to shlex: {e}") from e

        return v
