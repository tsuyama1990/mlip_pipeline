from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentConfig(BaseModel):
    """
    Configuration for a pipeline component.
    Must have a 'type' field.
    Component-specific parameters should be in the 'params' dictionary.
    """

    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


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
    validator: Optional[ComponentConfig] = None

    # Defaults
    dataset_filename: str = "dataset.jsonl"
    potential_extension: str = ".yace"

    # Constants/Defaults that might be used by components
    default_trainer_output: str = "potential"

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
