from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    energy: float = Field(..., description="Potential energy in eV")
    forces: list[list[float]] = Field(..., description="Forces on atoms in eV/A (Nx3)")
    stress: list[list[float]] | None = Field(None, description="Stress tensor (3x3) in eV/A^3")
    converged: bool = Field(..., description="Whether the calculation converged")
    error_message: str | None = Field(None, description="Error message if failed")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        # Basic check that inner lists have length 3
        for i, f in enumerate(forces):
            if len(f) != 3:
                raise ValueError(f"Force vector at index {i} must be length 3, got {len(f)}")
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[list[float]] | None) -> list[list[float]] | None:
        if stress is None:
            return None

        # Handle flattened list if it comes that way, but here we expect list of lists
        # If it's a 3x3 matrix
        if len(stress) == 3:
            for row in stress:
                if len(row) != 3:
                    raise ValueError("Stress tensor rows must be length 3.")
        # If it's Voigt (6,) - but Type hint says List[List], so we expect matrix.
        # If user passes empty list (which happened in tests), handle it?
        if len(stress) == 0:
            return None

        if len(stress) != 3:
            # Could be Voigt notation passed as list of lists? Unlikely.
            # Strict 3x3 enforcement
            raise ValueError("Stress tensor must be 3x3.")
        return stress
