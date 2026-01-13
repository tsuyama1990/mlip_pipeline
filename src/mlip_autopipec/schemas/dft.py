
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class DFTInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    atoms: Atoms
    dft_params: "system_config.DFTParams" = Field(..., description="DFT parameters")

class DFTOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_energy: float
    forces: list[list[float]]
    stress: list[list[float]]

# Forward reference resolution
from . import system_config

DFTInput.model_rebuild()
