from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

class DFTResult(BaseModel):
    """
    Result of a DFT calculation.
    """
    model_config = ConfigDict(extra="forbid")

    energy: float = Field(..., description="Potential energy in eV")
    forces: List[List[float]] = Field(..., description="Forces on atoms in eV/A (Nx3)")
    stress: Optional[List[List[float]]] = Field(None, description="Stress tensor (3x3) in eV/A^3")
    converged: bool = Field(..., description="Whether the calculation converged")
    error_message: Optional[str] = Field(None, description="Error message if failed")
