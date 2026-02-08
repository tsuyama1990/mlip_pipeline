from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(gt=0)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Components configuration
    # expected keys: generator, oracle, trainer, dynamics, validator
    components: dict[str, Any]

    @field_validator("components")
    @classmethod
    def validate_components(cls, v: dict[str, Any]) -> dict[str, Any]:
        required: set[str] = {"generator", "oracle", "trainer", "dynamics", "validator"}
        missing = required - v.keys()
        if missing:
            msg = f"Missing component configurations: {missing}"
            raise ValueError(msg)
        return v
