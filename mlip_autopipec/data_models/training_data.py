import numpy as np
from ase import Atoms
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class TrainingBatch(BaseModel):
    """
    A collection of structures ready for training.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: list[Atoms] = Field(..., description="List of atomic structures")
    force_masks: list[NDArray[np.float64]] = Field(
        ..., description="Weights for each atom (0 or 1)"
    )
