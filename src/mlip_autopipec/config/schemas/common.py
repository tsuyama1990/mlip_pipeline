from typing import Annotated

from ase.data import chemical_symbols
from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def validate_element(v: str) -> str:
    if v not in chemical_symbols:
        msg = f"Invalid chemical symbol: {v}"
        raise ValueError(msg)
    return v

Element = Annotated[str, AfterValidator(validate_element)]

class CommonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class EmbeddingConfig(BaseModel):
    """Configuration for cluster embedding."""
    core_radius: float = Field(default=4.0, gt=0.0)
    buffer_width: float = Field(default=2.0, gt=0.0)
    model_config = ConfigDict(extra="forbid")

    @property
    def box_size(self) -> float:
        return 2 * (self.core_radius + self.buffer_width) + 2.0
