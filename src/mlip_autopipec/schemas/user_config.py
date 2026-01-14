from pydantic import BaseModel, ConfigDict, field_validator


class TargetSystem(BaseModel):
    """Configuration for the target material system."""

    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    composition: dict[str, float]
    crystal_structure: str


class GenerationConfig(BaseModel):
    """Configuration for the structure generation process."""

    model_config = ConfigDict(extra="forbid")

    generation_type: str


class UserConfig(BaseModel):
    """Top-level user configuration model."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    target_system: TargetSystem
    generation_config: GenerationConfig

    @field_validator("target_system")
    def validate_target_system(cls, v: TargetSystem) -> TargetSystem:  # noqa: N805
        """Validate the consistency of the target system configuration."""
        elements_composition_mismatch = "elements and composition keys must match"
        if set(v.elements) != set(v.composition.keys()):
            raise ValueError(elements_composition_mismatch)

        composition_sum_invalid = "composition values must sum to 1.0"
        if not abs(sum(v.composition.values()) - 1.0) < 1e-6:
            raise ValueError(composition_sum_invalid)
        return v
