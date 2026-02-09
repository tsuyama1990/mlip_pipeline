from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationMetrics(BaseModel):
    """
    Standardized metrics returned by a Validator component.
    """

    model_config = ConfigDict(extra="forbid")

    energy_rmse: float | None = None
    force_rmse: float | None = None
    stress_rmse: float | None = None

    # Phonon Validation
    phonon_stable: bool | None = None

    # Elastic Validation
    elastic_stable: bool | None = None
    bulk_modulus: float | None = Field(default=None, description="Bulk Modulus in GPa")
    shear_modulus: float | None = Field(default=None, description="Shear Modulus in GPa")

    # EOS Validation
    eos_rmse: float | None = Field(default=None, description="RMSE of EOS fit")

    # Validation status
    passed: bool = Field(default=False, description="Whether the validation criteria were met.")

    # Detailed breakdown (optional)
    details: dict[str, Any] = Field(default_factory=dict)

    # Failed structures for loop
    # Using Any for Structure to avoid circular imports if needed, or string representation
    # But ideally it holds Structure objects.
    # We will use Any or specialized Pydantic model if we can import Structure.
    # To keep it simple and avoid circular dependency hell in strict mode:
    failed_structures: list[Any] = Field(default_factory=list)

    def __repr__(self) -> str:
        return f"<ValidationMetrics(passed={self.passed}, energy_rmse={self.energy_rmse})>"

    def __str__(self) -> str:
        return f"ValidationMetrics(passed={self.passed})"
