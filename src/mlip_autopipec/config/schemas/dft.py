import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DFTConfig(BaseModel):
    """
    Configuration for Density Functional Theory calculations.
    """

    # Required fields defined without default values
    pseudopotential_dir: Path = Field(description="Directory containing .UPF files")

    # Complex fields (with defaults or optional)
    ecutwfc: float = Field(60.0, gt=0, description="Wavefunction cutoff energy (Ry)")
    kspacing: float = Field(0.05, gt=0, description="Inverse K-point density (1/A)")

    # Optional fields with defaults
    command: str = Field(
        "pw.x", description="Command to run Quantum Espresso (e.g. 'mpirun -np 4 pw.x')"
    )
    nspin: int | None = Field(None, description="Spin polarization (1=non-spin, 2=spin-polarized)")
    diagonalization: Literal["david", "cg"] = Field("david", description="Solver")
    smearing: Literal["mv", "mp", "fd"] = Field(
        "mv", description="Smearing type ('mv', 'mp', 'fd')"
    )
    degauss: float = Field(0.02, gt=0, description="Smearing width (Ry)")
    recoverable: bool = Field(True, description="Enable auto-recovery")
    max_retries: int = Field(5, ge=0, description="Maximum number of retries")
    timeout: int = Field(3600, gt=0, description="Timeout in seconds for DFT calculation")
    pseudopotentials: dict[str, str] | None = Field(
        None, description="Map of element to filename"
    )

    @field_validator("pseudopotential_dir")
    @classmethod
    def validate_pseudo_dir(cls, v: Path) -> Path:
        if not v.exists():
            msg = f"Pseudopotential directory {v} does not exist"
            raise ValueError(msg)
        # Check for at least one UPF file
        if not list(v.glob("*.UPF")) and not list(v.glob("*.upf")):
            msg = f"No .UPF files found in {v}"
            raise ValueError(msg)
        return v

    @field_validator("command")
    @classmethod
    def validate_command_security(cls, v: str) -> str:
        # Security check for unsafe characters
        if re.search(r"[;&|`$()]", v):
            msg = "Command contains unsafe shell characters: ; & | ` $ ( ). Execution is blocked."
            raise ValueError(msg)

        try:
            parts = shlex.split(v)
            if not parts:
                msg = "Command cannot be empty"
                raise ValueError(msg)
        except ValueError as e:
            msg = f"Command string is invalid: {e}"
            raise ValueError(msg) from e
        return v
