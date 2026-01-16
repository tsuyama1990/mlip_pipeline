from pydantic import BaseModel, ConfigDict, field_validator

# Common / Shared Models

class SimulationGoal(BaseModel):
    # Moved from models.py
    # But wait, SimulationGoal was using Enum in one version and Literal in another.
    # config/models.py used Literal.
    from typing import Literal
    type: Literal["melt_quench", "elastic", "diffusion"]
    temperature_range: tuple[float, float] | None = None
    model_config = ConfigDict(extra="forbid")

class TargetSystem(BaseModel):
    elements: list[str]
    composition: dict[str, float]
    crystal_structure: str
    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    def validate_elements(cls, elements: list[str]) -> list[str]:
        from ase.data import chemical_symbols
        for symbol in elements:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return elements

    # Validator for composition sum... implemented in UserInputConfig or here?
    # It was in TargetSystem in models.py.
    from pydantic import model_validator
    @model_validator(mode="after")
    def check_composition_keys_and_sum(self):
        if set(self.elements) != set(self.composition.keys()):
            raise ValueError("Composition keys must match the elements list.")
        if not abs(sum(self.composition.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0.")
        return self

class UserInputConfig(BaseModel):
    project_name: str
    target_system: TargetSystem
    simulation_goal: SimulationGoal
    model_config = ConfigDict(extra="forbid")
