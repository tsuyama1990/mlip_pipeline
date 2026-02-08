from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentConfig(BaseModel):
    """
    Configuration for a pipeline component.
    Must have a 'type' field.
    Allows extra fields for component-specific settings.
    """

    type: str
    model_config = ConfigDict(extra="allow")


class GlobalConfig(BaseModel):
    """
    Root configuration for the active learning pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(ge=1)

    # Components
    generator: ComponentConfig
    oracle: ComponentConfig
    trainer: ComponentConfig
    dynamics: ComponentConfig
    validator: ComponentConfig | None = None

    # Defaults
    dataset_filename: str = "dataset.jsonl"
    potential_extension: str = ".yace"

    @model_validator(mode="after")
    def validate_components(self) -> "GlobalConfig":
        """
        Validates that all components have a type field.
        (Already enforced by ComponentConfig, but good for explicit check if we changed it).
        """
        return self
