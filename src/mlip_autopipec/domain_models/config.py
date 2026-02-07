from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GlobalConfig(BaseModel):
    """
    The root configuration object.
    """
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(ge=1)

    generator: dict[str, Any]
    oracle: dict[str, Any]
    trainer: dict[str, Any]
    dynamics: dict[str, Any]

    @model_validator(mode='after')
    def validate_components(self) -> 'GlobalConfig':
        """Ensure all component configs have a 'type' field."""
        for name in ['generator', 'oracle', 'trainer', 'dynamics']:
            cfg = getattr(self, name)
            if 'type' not in cfg:
                raise ValueError(f"{name} config must specify 'type'")
        return self
