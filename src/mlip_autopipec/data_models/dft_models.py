
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    diagonalization: str = "david"
    smearing: str = "mv"
    degauss: float = 0.02
    input_data: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

class DFTResult(BaseModel):
    """Result of a DFT calculation."""
    uid: str = Field(..., description="Unique Identifier for the calculation")
    succeeded: bool = Field(..., description="Whether the calculation finished successfully")
    converged: bool = Field(..., description="Whether the calculation converged")
    error_message: str | None = Field(None, description="Error message if failed")

    # Physics results
    energy: float = Field(..., description="Potential energy in eV")
    forces: list[list[float]] = Field(..., description="Forces on atoms in eV/A (Nx3)")
    stress: list[list[float]] | None = Field(None, description="Stress tensor (3x3) in eV/A^3")

    # Metadata
    wall_time: float = Field(0.0, description="Execution time in seconds")
    parameters: dict = Field(default_factory=dict, description="Parameters used")
    final_mixing_beta: float | None = Field(None, description="Final mixing beta used (if recovered)")

    model_config = ConfigDict(extra="forbid")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        if not forces:
            return forces
        for i, f in enumerate(forces):
            if len(f) != 3:
                msg = f"Force vector at index {i} must be length 3."
                raise ValueError(msg)
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[list[float]] | None) -> list[list[float]] | None:
        if stress is not None and len(stress) == 3 and isinstance(stress[0], list):
             for i, row in enumerate(stress):
                if len(row) != 3:
                    msg = f"Stress tensor row {i} must be length 3."
                    raise ValueError(msg)
        return stress
