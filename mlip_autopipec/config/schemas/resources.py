from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Resources(BaseModel):
    dft_code: Literal["quantum_espresso", "vasp"]
    parallel_cores: int = Field(gt=0)
    gpu_enabled: bool = False
    model_config = ConfigDict(extra="forbid")
