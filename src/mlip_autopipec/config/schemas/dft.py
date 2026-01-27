import re
import shlex
from pathlib import Path
from typing import Literal, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from mlip_autopipec.config.schemas.common import Element

class DFTConfig(BaseModel):
    """
    Configuration for Density Functional Theory calculations.
    """
    # Required fields (Simple + Complex overlap)
    pseudopotential_dir: Path = Field(description="Directory containing .UPF files")

    # Complex fields (with defaults or optional)
    ecutwfc: float = Field(60.0, gt=0, description="Wavefunction cutoff energy (Ry)")
    kspacing: float = Field(0.05, gt=0, description="Inverse K-point density (1/A)")
    command: str = Field("pw.x", description="Command to run Quantum Espresso")
    nspin: int = Field(1, description="Spin polarization (1=off, 2=on)")
    mixing_beta: float = Field(0.7, ge=0.0, le=1.0, description="SCF Mixing parameter")
    diagonalization: Literal["david", "cg"] = Field("david", description="Solver")
    smearing: Literal["mv", "mp", "fd"] = Field("mv", description="Smearing type")
    degauss: float = Field(0.02, gt=0, description="Smearing width (Ry)")
    recoverable: bool = Field(True, description="Enable auto-recovery")
    max_retries: int = Field(5, ge=0, description="Maximum number of retries")
    timeout: float = Field(3600.0, gt=0, description="Timeout in seconds")
    retry_delay_min: float = Field(4.0, gt=0, description="Minimum retry delay")
    retry_delay_max: float = Field(10.0, gt=0, description="Maximum retry delay")

    # Simple fields (Added for Cycle 01 compatibility)
    pseudopotentials: Dict[str, str] = Field(default_factory=dict, description="Mapping of element symbol to filename")
    scf_params: Dict[str, Any] = Field(default_factory=dict, description="Override parameters for QE input")

    model_config = ConfigDict(extra="forbid")

    @field_validator("pseudopotential_dir")
    @classmethod
    def validate_pseudo_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Pseudopotential directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Pseudopotential path is not a directory: {v}")
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
        if re.search(r"[;&|`$()]", v):
            raise ValueError("Command contains unsafe shell characters")
        try:
            parts = shlex.split(v)
            if not parts:
                raise ValueError("Command is empty.")
        except ValueError as e:
            raise ValueError(f"Command string is invalid: {e}") from e
        return v
