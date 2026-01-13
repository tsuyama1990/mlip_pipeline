from pydantic import BaseModel, ConfigDict, field_validator


class TargetSystem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    composition: dict[str, float]
    crystal_structure: str


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation_type: str


class UserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str
    target_system: TargetSystem
    generation_config: GenerationConfig

    @field_validator("target_system")
    def validate_target_system(cls, v: TargetSystem) -> TargetSystem:  # noqa: N805
        if not v.elements:
            msg = "Elements list cannot be empty."
            raise ValueError(msg)
        if not v.composition:
            msg = "Composition dictionary cannot be empty."
            raise ValueError(msg)
        if set(v.elements) != set(v.composition.keys()):
            msg = "Elements and composition keys must match."
            raise ValueError(msg)
        if not abs(sum(v.composition.values()) - 1.0) < 1e-6:
            msg = "Composition fractions must sum to 1.0."
            raise ValueError(msg)
        return v
