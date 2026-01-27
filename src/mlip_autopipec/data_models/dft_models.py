from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTErrorType(str, Enum):
    CONVERGENCE_FAIL = "CONVERGENCE_FAIL"
    DIAGONALIZATION_ERROR = "DIAGONALIZATION_ERROR"
    MAX_CPU_TIME = "MAX_CPU_TIME"
    OOM_KILL = "OOM_KILL"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class DFTInputParams(BaseModel):
    """
    Structured parameters for DFT input generation.
    """
    kspacing: float | None = None
    k_density: float | None = None  # Legacy/Alias support
    ecutwfc: float = 60.0
    ecutrho: float | None = None
    mixing_beta: float = 0.7
    electron_maxstep: int = 100
    diagonalization: Literal["david", "cg"] = "david"
    smearing: Literal["mv", "mp", "fd"] = "mv"
    degauss: float = 0.02
    input_data: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class DFTResult(BaseModel):
    uid: str
    energy: float
    forces: list[list[float]] = Field(..., description="Nx3 array")
    stress: list[list[float]] | None = Field(None, description="3x3 array")
    succeeded: bool
    converged: bool = Field(default=False) # Backwards compat
    error_message: str | None = None
    wall_time: float
    # Metadata for provenance
    parameters: dict[str, Any]
    final_mixing_beta: float | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        if not forces:
             return forces
        if not all(len(row) == 3 for row in forces):
            msg = "Forces must have a shape of (N_atoms, 3)."
            raise ValueError(msg)
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[list[float]] | None) -> list[list[float]] | None:
        if not stress:
            return stress
        if len(stress) == 3 and not all(len(row) == 3 for row in stress):
            msg = "Stress tensor must be 3x3."
            raise ValueError(msg)
        return stress
