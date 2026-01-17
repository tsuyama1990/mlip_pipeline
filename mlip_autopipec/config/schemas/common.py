"""
Common configuration schemas shared across the application.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator

from .resources import Resources

# Common / Shared Models

class SimulationGoal(BaseModel):
    """
    Defines the high-level scientific goal of the simulation.
    """
    type: Literal["melt_quench", "elastic", "diffusion"]
    temperature_range: tuple[float, float] | None = None
    model_config = ConfigDict(extra="forbid")

class Composition(RootModel[dict[str, float]]):
    """
    Represents the chemical composition of the system.
    Keys must be valid chemical symbols, and values must sum to 1.0.
    """

    @field_validator("root")
    def validate_composition(cls, v: dict[str, float]) -> dict[str, float]:
        """
        Validates that composition fractions sum to 1.0 and keys are valid symbols.
        """
        from ase.data import chemical_symbols

        # Check sum
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0.")

        # Check keys are valid symbols
        for symbol in v:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")

        return v

class TargetSystem(BaseModel):
    """
    Defines the material system to be studied.
    """
    elements: list[str]
    composition: Composition
    crystal_structure: str
    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    def validate_elements(cls, elements: list[str]) -> list[str]:
        """
        Validates that all elements are valid chemical symbols.
        """
        from ase.data import chemical_symbols
        for symbol in elements:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return elements

    @model_validator(mode="after")
    def check_composition_keys(self):
        """
        Ensures that the keys in the composition map match the elements list.
        """
        if set(self.elements) != set(self.composition.root.keys()):
            raise ValueError("Composition keys must match the elements list.")
        return self

class MinimalConfig(BaseModel):
    """
    The user-facing configuration model.
    Represents the content of the input.yaml file provided by the user.
    """
    project_name: str
    target_system: TargetSystem
    resources: Resources
    simulation_goal: SimulationGoal | None = None
    model_config = ConfigDict(extra="forbid")
