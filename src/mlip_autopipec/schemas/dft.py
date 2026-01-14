from ase import Atoms
from pydantic import BaseModel, ConfigDict

from .system_config import DFTParams


class DFTInput(BaseModel):
    """Input for a single DFT calculation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    atoms: Atoms
    dft_params: DFTParams


class DFTOutput(BaseModel):
    """Output of a successful DFT calculation."""

    model_config = ConfigDict(extra="forbid")

    total_energy: float
    forces: list[list[float]]
    stress: list[list[float]]
