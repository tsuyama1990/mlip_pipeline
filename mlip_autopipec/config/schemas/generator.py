from typing import Literal

from pydantic import BaseModel, ConfigDict


class GeneratorConfig(BaseModel):
    """Configuration for structure generation."""
    generator_type: Literal["sqs", "random", "defect"] = "sqs"
    supercell_size: tuple[int, int, int] = (2, 2, 2)
    model_config = ConfigDict(extra="forbid")
