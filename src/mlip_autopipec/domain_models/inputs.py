from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import TaskStatus


def sanitize_value(v: Any) -> Any:
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(v, dict):
        return {k: sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize_value(val) for val in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


class Structure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positions: list[list[float]] = Field(..., description="Nx3 array of atomic positions")
    numbers: list[int] = Field(..., description="List of atomic numbers (Z)")
    cell: list[list[float]] = Field(..., description="3x3 unit cell matrix")
    pbc: list[bool] = Field(..., description="Periodic boundary conditions [x, y, z]")
    tags: dict[str, Any] = Field(default_factory=dict, description="Metadata tags")

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return sanitize_value(v)
        return v

    @classmethod
    def from_ase(cls, atoms: Atoms) -> Structure:
        """Convert ASE Atoms object to Structure model."""
        # Sanitize info dictionary
        tags = sanitize_value(atoms.info.copy())

        return cls(
            positions=atoms.get_positions().tolist(),  # type: ignore[no-untyped-call]
            numbers=atoms.get_atomic_numbers().tolist(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell().tolist(),  # type: ignore[no-untyped-call]
            pbc=atoms.get_pbc().tolist(),  # type: ignore[no-untyped-call]
            tags=tags,
        )

    def to_ase(self) -> Atoms:
        """Convert Structure model to ASE Atoms object."""
        atoms = Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
        )
        atoms.info.update(self.tags)

        # ASE expects numpy arrays for certain physical quantities in info
        # especially for IO operations (e.g. write_xyz checking stress shape)
        for key in ["stress", "forces", "dipole", "virial"]:
            if key in atoms.info and isinstance(atoms.info[key], list):
                atoms.info[key] = np.array(atoms.info[key])

        return atoms


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Human-readable job name")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    command: str | None = Field(None, description="Command executed")
    work_dir: Path = Field(..., description="Working directory for the job")
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_iteration: int = Field(default=0, ge=0)
    current_potential_path: Path | None = None
    current_dataset_path: Path | None = None
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    history: list[dict[str, Any]] = Field(default_factory=list, description="History of iterations")
