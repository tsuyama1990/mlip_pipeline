from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_autopipec.domain_models.enums import TaskStatus


class Structure(BaseModel):
    """
    Domain entity wrapping an atomic structure with metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    uid: UUID = Field(default_factory=uuid4)
    atoms: Atoms
    provenance: str = Field(..., min_length=1)
    tags: dict[str, Any] = Field(default_factory=dict)
    energy: float | None = None
    forces: np.ndarray | None = None

    @model_validator(mode="after")
    def validate_forces_shape(self) -> "Structure":
        if self.forces is not None:
            if len(self.forces) != len(self.atoms):
                msg = f"Forces array length ({len(self.forces)}) must match number of atoms ({len(self.atoms)})"
                raise ValueError(msg)
            if self.forces.ndim != 2 or self.forces.shape[1] != 3:
                msg = f"Forces array must have shape (N, 3), got {self.forces.shape}"
                raise ValueError(msg)
        return self


class WorkflowState(BaseModel):
    """
    Tracks the global state of the active learning loop.
    """

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=0)
    current_potential_path: Path | None = None
    dataset_path: Path | None = None
    status: TaskStatus = Field(default=TaskStatus.PENDING)
