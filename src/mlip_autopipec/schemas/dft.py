"""Pydantic models for DFT calculation inputs and outputs."""

from ase import Atoms
from pydantic import BaseModel, ConfigDict

from .system_config import DFTParams


class DFTInput(BaseModel):
    """Represents a single, ready-to-run DFT calculation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    atoms: Atoms
    params: DFTParams


class DFTOutput(BaseModel):
    """Represents the results of a successful DFT calculation."""

    model_config = ConfigDict(extra="forbid")

    total_energy: float
    forces: list[list[float]]
    stress: list[list[float]]
