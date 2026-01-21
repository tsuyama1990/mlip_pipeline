from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from numpy.typing import NDArray

class DFTResult(BaseModel):
    """
    Data model for the result of a DFT calculation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    energy: float = Field(..., description="Total potential energy in eV")
    forces: NDArray[np.float64] = Field(..., description="Atomic forces in eV/A")
    stress: Optional[NDArray[np.float64]] = Field(None, description="Stress tensor in eV/A^3")
