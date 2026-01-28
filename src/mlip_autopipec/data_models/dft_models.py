from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DFTErrorType(str, Enum):
    CONVERGENCE_FAIL = "CONVERGENCE_FAIL"
    DIAGONALIZATION_ERROR = "DIAGONALIZATION_ERROR"
    MAX_CPU_TIME = "MAX_CPU_TIME"
    OOM_KILL = "OOM_KILL"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class DFTInputParams(BaseModel):
    """Parameters for generating DFT input files."""

    model_config = ConfigDict(extra="forbid")

    ecutwfc: float = Field(..., gt=0, description="Wavefunction cutoff (Ry).")
    ecutrho: float | None = Field(default=None, description="Charge density cutoff (Ry).")
    kspacing: float = Field(..., gt=0, description="K-point spacing (1/A).")
    k_density: float | None = Field(default=None, description="Legacy parameter for K-point density (1/A).")
    smearing: str = Field(default="mp", description="Smearing method.")
    degauss: float = Field(default=0.02, ge=0, description="Smearing width (Ry).")
    mixing_beta: float = Field(default=0.7, gt=0, le=1.0, description="Mixing beta.")
    diagonalization: str = Field(default="david", description="Diagonalization method.")
    electron_maxstep: int = Field(default=100, gt=0, description="Max SCF steps.")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Override parameters for input sections.")


class DFTResult(BaseModel):
    """Result of a DFT calculation."""

    model_config = ConfigDict(extra="forbid")

    uid: str = Field(..., description="Unique identifier for the calculation.")
    energy: float = Field(..., description="Total energy (eV).")
    forces: list[list[float]] = Field(..., description="Forces on atoms (eV/A).")
    stress: list[list[float]] = Field(..., description="Stress tensor (eV/A^3).")
    succeeded: bool = Field(..., description="Whether the calculation finished successfully.")
    converged: bool = Field(..., description="Whether the SCF cycle converged.")
    error_message: str | None = Field(default=None, description="Error message if failed.")
    wall_time: float = Field(..., ge=0, description="Execution time in seconds.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Input parameters used.")
    final_mixing_beta: float | None = Field(default=None, description="Final mixing beta used.")

    @model_validator(mode="after")
    def check_shapes(self) -> "DFTResult":
        # Validate Forces Shape (N x 3)
        if self.forces:
            for f in self.forces:
                if len(f) != 3:
                    msg = "Each force vector must have 3 components."
                    raise ValueError(msg)

        # Validate Stress Shape (3 x 3)
        if self.stress:
            if len(self.stress) != 3:
                msg = "Stress tensor must be 3x3."
                raise ValueError(msg)
            for row in self.stress:
                if len(row) != 3:
                    msg = "Stress tensor rows must have 3 components."
                    raise ValueError(msg)
        return self
