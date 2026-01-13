"""Pydantic models for the user-facing configuration (input.yaml)."""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TargetSystem(BaseModel):
    """Defines the material system to be simulated."""

    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    composition: dict[str, float]
    crystal_structure: str


class GenerationConfig(BaseModel):
    """Defines the type of structure generation to perform."""

    model_config = ConfigDict(extra="forbid")

    generation_type: str


class UserConfig(BaseModel):
    """Top-level Pydantic model for the user_config.yaml file."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    target_system: TargetSystem
    generation_config: GenerationConfig

    @model_validator(mode="after")
    def check_elements_match_composition(self) -> "UserConfig":
        """Validator to ensure elements and composition keys match."""
        elements = sorted(self.target_system.elements)
        composition_keys = sorted(self.target_system.composition.keys())
        if elements != composition_keys:
            error_msg = "`elements` and `composition` keys do not match."
            raise ValueError(error_msg)
        return self

    @field_validator("target_system")
    @classmethod
    def check_composition_sum(cls, v: TargetSystem) -> TargetSystem:
        """Validator to ensure composition fractions sum to 1.0."""
        total_fraction = sum(v.composition.values())
        if not (1.0 - 1e-6 < total_fraction < 1.0 + 1e-6):
            error_msg = "Composition fractions must sum to 1.0"
            raise ValueError(error_msg)
        return v
