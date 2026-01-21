"""
This module contains the schemas for the Inference Engine configuration.
"""

import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator


class InferenceConfig(BaseModel):
    """
    Configuration for Inference Engine.

    Attributes:
        temperature (float): Simulation temperature in Kelvin.
        pressure (float): Simulation pressure in Bar (for NPT). 0 for NVT.
        timestep (float): Time step in picoseconds.
        steps (int): Number of MD steps to run.
        ensemble (str): Thermodynamic ensemble ("nvt" or "npt").
        uq_threshold (float): Threshold for extrapolation grade (gamma) to trigger active learning dump.
        sampling_interval (int): Interval (in steps) for thermo output and dumping.
        potential_path (FilePath): Path to the MLIP potential file (.yace).
        lammps_executable (FilePath | None): Path to the LAMMPS executable (optional).
        elements (list[str]): List of chemical elements in the system, order-matched to potential.
    """

    temperature: float = Field(..., gt=0, description="Temperature in Kelvin")
    pressure: float = Field(0.0, ge=0, description="Pressure in Bar (0 for NVT)")
    timestep: float = Field(0.001, gt=0, description="Time step in ps")
    steps: int = Field(10000, gt=0, description="Number of steps")
    ensemble: Literal["nvt", "npt"] = Field("nvt", description="Ensemble type")
    uq_threshold: float = Field(5.0, gt=0, description="Uncertainty threshold (gamma)")
    sampling_interval: int = Field(100, gt=0, description="Sampling interval")
    potential_path: FilePath = Field(..., description="Path to potential file")
    lammps_executable: FilePath | None = Field(None, description="Path to LAMMPS executable")
    elements: list[str] = Field(default_factory=lambda: ["Al"], description="List of elements")

    model_config = ConfigDict(extra="forbid")

    @field_validator("pressure")
    @classmethod
    def validate_pressure_for_npt(cls, v: float, info) -> float:
        if info.data.get("ensemble") == "npt" and v <= 0 and info.data.get("pressure", 0) == 0:
            pass
        return v

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: FilePath | None) -> FilePath | None:
        if v is not None:
            # Check executable permission using stat to avoid os.access (TOCTOU)
            # We check if User Execute bit is set.
            st_mode = v.stat().st_mode
            if not (st_mode & stat.S_IXUSR):
                raise ValueError(f"File at {v} is not executable.")
        return v

    @field_validator("potential_path")
    @classmethod
    def validate_potential_exists(cls, v: FilePath) -> FilePath:
        if not v.exists():
            raise ValueError(f"Potential file {v} does not exist.")
        return v


class InferenceResult(BaseModel):
    """
    Result object for Inference run.

    Attributes:
        succeeded (bool): Whether the simulation completed successfully.
        final_structure (Path | None): Path to the final structure file (if success).
        uncertain_structures (list[Path]): List of paths to files containing uncertain structures.
        max_gamma_observed (float): The maximum extrapolation grade observed during the run.
    """

    succeeded: bool
    final_structure: Path | None = None
    uncertain_structures: list[Path] = Field(default_factory=list)
    max_gamma_observed: float = Field(0.0, ge=0.0, description="Max gamma observed")

    model_config = ConfigDict(extra="forbid")


class EmbeddingConfig(BaseModel):
    """
    Configuration for Local Environment Extraction (Embedding).

    Attributes:
        core_radius (float): Radius of the trusted core region (Angstroms).
        buffer_width (float): Width of the buffer region (Angstroms).
    """

    core_radius: float = Field(
        4.0, gt=0, description="Radius of the trusted core region (Angstroms)"
    )
    buffer_width: float = Field(2.0, gt=0, description="Width of the buffer region (Angstroms)")

    model_config = ConfigDict(extra="forbid")

    @property
    def box_size(self) -> float:
        """Derived box size ~ 2 * (core + buffer)"""
        return 2.0 * (self.core_radius + self.buffer_width)
