from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


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
    failed_structures: list[Structure] = Field(default_factory=list)

    def __repr__(self) -> str:
        return f"<ValidationMetrics(passed={self.passed}, energy_rmse={self.energy_rmse})>"

    def __str__(self) -> str:
        return f"ValidationMetrics(passed={self.passed})"
