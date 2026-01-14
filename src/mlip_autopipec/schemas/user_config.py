from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class SurrogateConfig(BaseModel):
    """Configuration for the surrogate model explorer."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    num_to_select_fps: int = Field(..., ge=0)
    descriptor_type: str


class TrainerConfig(BaseModel):
    """Configuration for the MLIP trainer."""

    model_config = ConfigDict(extra="forbid")

    radial_basis: str
    max_body_order: int
    loss_weights: dict[str, float]

    @field_validator("loss_weights")
    def validate_loss_weights(cls, v: dict[str, float]) -> dict[str, float]:  # noqa: N805
        """Validate the keys in loss_weights."""
        allowed_keys = {"energy", "forces", "stress"}
        if not set(v.keys()).issubset(allowed_keys):
            error_msg = f"Loss weights keys must be in {allowed_keys}"
            raise ValueError(error_msg)
        return v


class UserConfig(BaseModel):
    """Top-level user configuration model."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    target_system: TargetSystem
    generation_config: GenerationConfig
    surrogate_config: SurrogateConfig
    trainer_config: TrainerConfig

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
