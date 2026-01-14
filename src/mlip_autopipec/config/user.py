"""User-facing configuration schemas for MLIP-AutoPipe."""

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


class SimulationGoal(str, Enum):
    """Enumeration of supported simulation goals."""

    MELT_QUENCH = "melt_quench"
    ELASTIC = "elastic"
    VACANCY_DIFFUSION = "vacancy_diffusion"


class TargetSystem(BaseModel):
    """Defines the chemical system to be simulated."""

    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(
        ...,
        min_length=1,
        description="A list of chemical symbols of the constituent elements.",
    )
    composition: dict[str, float] = Field(
        ...,
        description=(
            "A dictionary mapping element symbols to their atomic fractions. "
            "The sum of fractions must be 1.0."
        ),
    )

    @field_validator("composition")
    @classmethod
    def validate_composition_fractions(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that the composition fractions sum to 1.0."""
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0")
        return v

    @field_validator("composition")
    @classmethod
    def validate_composition_elements(
        cls, v: dict[str, float], info: ValidationInfo
    ) -> dict[str, float]:
        """Validate that the elements in composition match the elements list."""
        if "elements" in info.data and set(v.keys()) != set(info.data["elements"]):
            raise ValueError("Elements in composition must match the elements list")
        return v


class Resources(BaseModel):
    """Defines the computational resources for the simulation."""

    model_config = ConfigDict(extra="forbid")

    dft_cores: int = Field(
        1, ge=1, description="Number of CPU cores for each DFT calculation."
    )
    md_cores: int = Field(
        1, ge=1, description="Number of CPU cores for each MD simulation."
    )
    max_dft_calculations: int = Field(
        100,
        ge=1,
        description="Maximum number of DFT calculations to perform in a run.",
    )


class UserConfig(BaseModel):
    """User-facing configuration for an MLIP-AutoPipe workflow.

    This schema defines the minimal, high-level input required from a user
    to launch a full automated MLIP generation pipeline. It focuses on the
    scientific intent rather than the low-level simulation parameters.
    """

    model_config = ConfigDict(extra="forbid")

    target_system: TargetSystem = Field(
        ..., description="The chemical system to be simulated."
    )
    simulation_goal: SimulationGoal = Field(
        ..., description="The primary scientific goal of the simulation campaign."
    )
    resources: Resources = Field(
        default_factory=Resources,
        description="Computational resources for the workflow.",
    )
