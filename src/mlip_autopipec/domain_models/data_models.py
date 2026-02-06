from pydantic import BaseModel, ConfigDict, Field
from ase import Atoms
from typing import List, Optional, Dict
from pathlib import Path

class StructureMetadata(BaseModel):
    """
    Wraps an ASE Atoms object with additional metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structure: Atoms
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    virial: Optional[List[List[float]]] = None
    iteration: int = 0

class Dataset(BaseModel):
    """
    A collection of StructureMetadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: List[StructureMetadata] = Field(default_factory=list)

class ValidationResult(BaseModel):
    """
    Results from a validation run.
    """
    model_config = ConfigDict(extra="forbid")

    metrics: Dict[str, float]
    is_stable: bool
