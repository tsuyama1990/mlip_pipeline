from typing import List
from pydantic import BaseModel, ConfigDict, Field
from ase import Atoms
import numpy as np
from numpy.typing import NDArray

class TrainingBatch(BaseModel):
    """
    A collection of structures ready for training.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: List[Atoms] = Field(..., description="List of atomic structures")
    force_masks: List[NDArray[np.float64]] = Field(..., description="Weights for each atom (0 or 1)")
