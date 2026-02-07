
import numpy as np
from pydantic import BaseModel, ConfigDict


class Structure(BaseModel):
    """
    Represents an atomic structure with positions, cell, and optional properties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: list[list[float]] | np.ndarray
    cell: list[list[float]] | np.ndarray
    species: list[str]
    energy: float | None = None
    forces: list[list[float]] | np.ndarray | None = None
    stress: list[list[float]] | np.ndarray | None = None
