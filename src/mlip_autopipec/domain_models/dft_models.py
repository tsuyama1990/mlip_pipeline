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
    energy: float = Field(..., description="Potential energy in eV")
    forces: list[list[float]] = Field(..., description="Forces on atoms in eV/A (Nx3)")
    stress: list[list[float]] | None = Field(None, description="Stress tensor (3x3) in eV/A^3")
    succeeded: bool
    converged: bool = Field(default=False, description="Whether the calculation converged")
    error_message: str | None = Field(None, description="Error message if failed")
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
        for i, row in enumerate(forces):
            if len(row) != 3:
                msg = f"Forces row {i} must have 3 components."
                raise ValueError(msg)
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[list[float]] | None) -> list[list[float]] | None:
        if stress is None:
            return stress

        # Allow empty list as None
        if not stress:
            return None

        # Check for 3x3
        if len(stress) == 3:
            for row in stress:
                if len(row) != 3:
                    msg = "Stress tensor must be 3x3."
                    raise ValueError(msg)
        elif len(stress) == 6:
             pass # Voigt
        elif len(stress) == 9:
             pass # Flattened 3x3
        else:
            msg = "Stress must be 3x3 (matrix), 6 (Voigt), or 9 (flattened)."
            raise ValueError(msg)

        return stress
