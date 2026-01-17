from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

# Common / Shared Models

class SimulationGoal(BaseModel):
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
    elements: list[str]
    composition: Composition
    # crystal_structure is not in SPEC minimal config but useful. Making it optional or keeping as is?
    # UAT example input does NOT have crystal_structure inside target_system either!
    # UAT input: elements, composition.
    # So I should make crystal_structure optional or remove it from strict validation if I want to pass UAT with that exact input.
    # I will make it optional.
    crystal_structure: str | None = None

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

class Resources(BaseModel):
    dft_code: Literal["quantum_espresso", "vasp"]
    parallel_cores: int = Field(gt=0)
    gpu_enabled: bool = False
    model_config = ConfigDict(extra="forbid")

class MinimalConfig(BaseModel):
    project_name: str
    target_system: TargetSystem
    resources: Resources
    # Simulation goal is not in Cycle 1 UAT
    model_config = ConfigDict(extra="forbid")

# Alias for backward compatibility if needed, but we should prefer MinimalConfig
UserInputConfig = MinimalConfig
