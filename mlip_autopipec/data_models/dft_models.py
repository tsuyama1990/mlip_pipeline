from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTErrorType(str, Enum):
    CONVERGENCE_FAIL = "CONVERGENCE_FAIL"
    DIAGONALIZATION_ERROR = "DIAGONALIZATION_ERROR"
    MAX_CPU_TIME = "MAX_CPU_TIME"
    OOM_KILL = "OOM_KILL"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class DFTResult(BaseModel):
    uid: str
    energy: float
    forces: list[list[float]] = Field(..., description="Nx3 array")
    stress: list[list[float]] = Field(..., description="3x3 array")
    succeeded: bool
    error_message: str | None = None
    wall_time: float
    # Metadata for provenance
    parameters: dict[str, Any]
    final_mixing_beta: float

    model_config = ConfigDict(extra="forbid")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        if not all(len(row) == 3 for row in forces):
            msg = "Forces must have a shape of (N_atoms, 3)."
            raise ValueError(msg)
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[list[float]]) -> list[list[float]]:
        # SPEC says 3x3 array for DFTResult in Data Models section.
        if len(stress) != 3:
            msg = "Stress tensor must be 3x3."
            raise ValueError(msg)
        if not all(len(row) == 3 for row in stress):
            msg = "Stress tensor must be 3x3."
            raise ValueError(msg)
        return stress
