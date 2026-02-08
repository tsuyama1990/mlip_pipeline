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

    workdir: Path = Field(default=Path("runs/default"))
    max_cycles: int = Field(default=5, ge=1)
    logging_level: str = "INFO"
    logging_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

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
        """
        return self

    @model_validator(mode="after")
    def validate_paths(self) -> "GlobalConfig":
        """
        Ensures dataset_filename is relative and safe.
        """
        p = Path(self.dataset_filename)
        if p.is_absolute():
            msg = f"dataset_filename must be relative, got {self.dataset_filename}"
            raise ValueError(msg)
        if ".." in self.dataset_filename:
             msg = f"dataset_filename cannot contain parent directory traversal: {self.dataset_filename}"
             raise ValueError(msg)
        return self
