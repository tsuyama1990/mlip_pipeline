from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

# Common / Shared Models

class Composition(RootModel[dict[str, float]]):
    """
    Represents the chemical composition of the system.
    Keys must be valid chemical symbols, and values must sum to 1.0.
    """

    @field_validator("root")
    def validate_composition(cls, v: dict[str, float]) -> dict[str, float]:  # noqa: N805
        from ase.data import chemical_symbols

        # Check sum
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            msg = "Composition fractions must sum to 1.0."
            raise ValueError(msg)

        # Check keys are valid symbols
        for symbol in v:
            if symbol not in chemical_symbols:
                msg = f"'{symbol}' is not a valid chemical symbol."
                raise ValueError(msg)

        return v

class TargetSystem(BaseModel):
    elements: list[str]
    composition: Composition
    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    def validate_elements(cls, elements: list[str]) -> list[str]:  # noqa: N805
        from ase.data import chemical_symbols
        for symbol in elements:
            if symbol not in chemical_symbols:
                msg = f"'{symbol}' is not a valid chemical symbol."
                raise ValueError(msg)
        return elements

    @model_validator(mode="after")
    def check_composition_keys(self):
        if set(self.elements) != set(self.composition.root.keys()):
            msg = "Composition keys must match the elements list."
            raise ValueError(msg)
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
    model_config = ConfigDict(extra="forbid")
