import re
import shlex
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTConfig(BaseModel):
    """
    Configuration for Density Functional Theory calculations.
    """
    # Required fields
    pseudopotential_dir: Path = Field(description="Directory containing .UPF files")

    # Optional/Default fields
    ecutwfc: float = Field(default=60.0, gt=0, description="Wavefunction cutoff energy (Ry)")
    kspacing: float = Field(default=0.05, gt=0, description="Inverse K-point density (1/A)")
    command: str = Field(
        default="pw.x", description="Command to run Quantum Espresso (e.g. 'mpirun -np 4 pw.x')"
    )
    nspin: int = Field(default=1, description="Spin polarization (1=off, 2=on)")
    mixing_beta: float = Field(default=0.7, ge=0.0, le=1.0, description="SCF Mixing parameter")
    diagonalization: Literal["david", "cg"] = Field(default="david", description="Solver")
    smearing: Literal["mv", "mp", "fd"] = Field(
        default="mv", description="Smearing type ('mv', 'mp', 'fd')"
    )
    degauss: float = Field(default=0.02, gt=0, description="Smearing width (Ry)")
    recoverable: bool = Field(default=True, description="Enable auto-recovery")
    max_retries: int = Field(default=5, ge=0, description="Maximum number of retries")
    timeout: float = Field(default=3600.0, gt=0, description="Timeout in seconds")
    retry_delay_min: float = Field(default=4.0, gt=0, description="Minimum retry delay")
    retry_delay_max: float = Field(default=10.0, gt=0, description="Maximum retry delay")

    # Cycle 01 compatibility
    pseudopotentials: dict[str, str] = Field(default_factory=dict, description="Mapping of element symbol to filename")
    scf_params: dict[str, Any] = Field(default_factory=dict, description="Override parameters for QE input")

    model_config = ConfigDict(extra="forbid")

    @field_validator("pseudopotential_dir")
    @classmethod
    def validate_pseudo_dir(cls, v: Path) -> Path:
        if not v.exists():
            msg = f"Pseudopotential directory does not exist: {v}"
            raise ValueError(msg)
        if not v.is_dir():
            msg = f"Pseudopotential path is not a directory: {v}"
            raise ValueError(msg)

        # Check for UPF files (case-insensitive)
        upf_files = list(v.glob("*.[uU][pP][fF]"))
        if not upf_files:
            msg = f"No .UPF files found in pseudopotential directory: {v}"
            raise ValueError(msg)

        return v

    @field_validator("nspin")
    @classmethod
    def validate_nspin(cls, v: int) -> int:
        if v not in (1, 2):
            msg = "nspin must be 1 or 2."
            raise ValueError(msg)
        return v

    @field_validator("command")
    @classmethod
    def validate_command_security(cls, v: str) -> str:
        if re.search(r"[;&|`$()]", v):
            msg = "Command contains unsafe shell characters"
            raise ValueError(msg)
        try:
            parts = shlex.split(v)
            if not parts:
                msg = "Command is empty."
                raise ValueError(msg)
        except ValueError as e:
            msg = f"Command string is invalid: {e}"
            raise ValueError(msg) from e
        return v
