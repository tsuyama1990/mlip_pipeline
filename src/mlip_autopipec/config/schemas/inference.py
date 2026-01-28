from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Default Constants
DEFAULT_TEMP = 300.0
DEFAULT_PRESSURE = 1.0
DEFAULT_TIMESTEP = 1.0
DEFAULT_STEPS = 1000
DEFAULT_UNCERTAINTY = 5.0
DEFAULT_INTERVAL = 10
DEFAULT_ZBL_INNER = 1.0
DEFAULT_ZBL_OUTER = 2.0
DEFAULT_RESTART_INTERVAL = 1000


class BaselinePotential(str, Enum):
    """Enumeration of supported baseline potentials."""
    ZBL = "ZBL"
    LJ = "LJ"
    NONE = "None"


class EONConfig(BaseModel):
    """Configuration for EON (kMC) inference."""

    eon_executable: Path | None = Field(None, description="Path to EON executable")
    job: Literal["process_search", "saddle_search", "minimization"] = Field(
        "process_search", description="EON Job Type"
    )
    temperature: float = Field(300.0, ge=0.0, description="Temperature (K)")
    pot_name: str = Field("pace_driver", description="Potential name (corresponds to script name)")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional EON parameters"
    )

    model_config = ConfigDict(extra="forbid")


class InferenceConfig(BaseModel):
    """Configuration for Inference / Dynamics (LAMMPS or EON)."""

    lammps_executable: Path | None = Field(None, description="Path to LAMMPS executable")
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, description="MD temperature in Kelvin (K)")
    pressure: float = Field(DEFAULT_PRESSURE, ge=0.0, description="MD pressure in Bar")
    timestep: float = Field(DEFAULT_TIMESTEP, gt=0.0, description="Timestep in fs")
    steps: int = Field(DEFAULT_STEPS, gt=0, description="Number of MD steps")
    uncertainty_threshold: float = Field(
        DEFAULT_UNCERTAINTY, gt=0.0, description="Max extrapolation grade before stop"
    )

    # New fields
    ensemble: Literal["nvt", "npt"] = Field("nvt", description="MD Ensemble (nvt or npt)")
    sampling_interval: int = Field(
        DEFAULT_INTERVAL, gt=0, description="Interval for thermo output and dumping"
    )

    baseline_potential: BaselinePotential = Field(
        BaselinePotential.ZBL, description="Baseline potential to use (hybrid/overlay)"
    )

    zbl_inner_cutoff: float = Field(
        DEFAULT_ZBL_INNER, gt=0.0, description="Inner cutoff for ZBL switching"
    )
    zbl_outer_cutoff: float = Field(
        DEFAULT_ZBL_OUTER, gt=0.0, description="Outer cutoff for ZBL switching"
    )
    restart_interval: int = Field(
        DEFAULT_RESTART_INTERVAL, gt=0, description="Interval for writing restart files"
    )

    # EON Integration
    eon: EONConfig | None = Field(None, description="EON Configuration")
    active_engine: Literal["lammps", "eon"] = Field("lammps", description="Active Dynamics Engine")

    model_config = ConfigDict(extra="forbid")

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: Path | None) -> Path | None:
        """
        Validates that the executable path string is not empty if provided.
        Actual existence check is runtime dependent (e.g. remote nodes),
        but we can check if it's a valid path structure.
        """
        if v is not None and str(v).strip() == "":
            msg = "LAMMPS executable path cannot be empty."
            raise ValueError(msg)
        return v
