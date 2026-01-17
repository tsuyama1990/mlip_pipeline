"""
Configuration schemas for computational resources.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Resources(BaseModel):
    """
    Defines the computational resources available for the workflow.
    """
    dft_code: Literal["quantum_espresso", "vasp"]
    parallel_cores: int = Field(gt=0, description="Number of cores to use for MPI tasks.")
    gpu_enabled: bool = False
    model_config = ConfigDict(extra="forbid")
