from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator

# Common / Shared Models

class SimulationGoal(BaseModel):
    type: Literal["melt_quench", "elastic", "diffusion"]
    temperature_range: tuple[float, float] | None = None
    model_config = ConfigDict(extra="forbid")

class Composition(RootModel[dict[str, float]]):
    # RootModel does not support extra="forbid" because it's a wrapper around a type,
    # and dict[str, float] naturally allows any keys that are strings.
    # Validation of keys is done in TargetSystem.

    @field_validator("root")
    def validate_fractions(cls, v: dict[str, float]) -> dict[str, float]:
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0.")
        return v

class TargetSystem(BaseModel):
    elements: list[str]
    composition: Composition
    crystal_structure: str
    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    def validate_elements(cls, elements: list[str]) -> list[str]:
        from ase.data import chemical_symbols
        for symbol in elements:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return elements

    @model_validator(mode="after")
    def check_composition_keys(self):
        if set(self.elements) != set(self.composition.root.keys()):
            raise ValueError("Composition keys must match the elements list.")
        return self

class UserInputConfig(BaseModel):
    project_name: str
    target_system: TargetSystem
    simulation_goal: SimulationGoal
    model_config = ConfigDict(extra="forbid")
