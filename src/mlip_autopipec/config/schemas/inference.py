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

class EONConfig(BaseModel):
    """Configuration for EON (kMC) inference."""
    eon_executable: Path | None = Field(None, description="Path to EON executable")
    job: Literal["process_search", "saddle_search", "minimization"] = Field("process_search", description="EON Job Type")
    temperature: float = Field(300.0, ge=0.0, description="Temperature (K)")
    pot_name: str = Field("pace_driver", description="Potential name (corresponds to script name)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional EON parameters")

    model_config = ConfigDict(extra="forbid")

class InferenceRuntimeConfig(BaseModel):
    """Configuration for MD runtime parameters."""
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, description="MD temperature in Kelvin (K)")
    pressure: float = Field(DEFAULT_PRESSURE, ge=0.0, description="MD pressure in Bar")
    timestep: float = Field(DEFAULT_TIMESTEP, gt=0.0, description="Timestep in fs")
    steps: int = Field(DEFAULT_STEPS, gt=0, description="Number of MD steps")
    ensemble: Literal["nvt", "npt"] = Field("nvt", description="MD Ensemble (nvt or npt)")

    model_config = ConfigDict(extra="forbid")

class InferenceAlgorithmConfig(BaseModel):
    """Configuration for MD/Inference algorithms."""
    uncertainty_threshold: float = Field(DEFAULT_UNCERTAINTY, gt=0.0, description="Max extrapolation grade before stop")
    sampling_interval: int = Field(DEFAULT_INTERVAL, gt=0, description="Interval for thermo output and dumping")
    use_zbl_baseline: bool = Field(True, description="Whether to use ZBL baseline (hybrid/overlay)")
    zbl_inner_cutoff: float = Field(DEFAULT_ZBL_INNER, gt=0.0, description="Inner cutoff for ZBL switching")
    zbl_outer_cutoff: float = Field(DEFAULT_ZBL_OUTER, gt=0.0, description="Outer cutoff for ZBL switching")
    restart_interval: int = Field(DEFAULT_RESTART_INTERVAL, gt=0, description="Interval for writing restart files")

    model_config = ConfigDict(extra="forbid")

class InferenceConfig(BaseModel):
    """
    Configuration for Inference / Dynamics (LAMMPS or EON).

    Composite config including runtime and algorithm settings.
    Fields are flattened for backward compatibility where possible, but better accessed via sub-models.
    Actually, to maintain current API without breakage, we can use fields that alias to internal sub-models
    or just keep the flat structure on the surface and map internally.
    However, the audit request was "Break down...".
    So we expose them.
    """
    lammps_executable: Path | None = Field(None, description="Path to LAMMPS executable")

    # We mix in the fields to keep 'config.yaml' structure valid for users?
    # Or do we change the structure? The audit implies internal maintainability.
    # Changing the structure breaks existing users (UAT).
    # We will keep the fields but group them logically if we can, or just composition.
    # To satisfy "Break down", I will use inheritance or composition.
    # Composition breaks 'config.inference_config.temperature'.
    # Inheritance is better here for API compatibility.

    # Actually, let's use composition but with top-level fields delegating? No that's boilerplate.
    # I will simply inline the mixins.

    # Wait, the instruction is "Consider breaking down...".
    # I will create sub-configs but `InferenceConfig` will inherit from them OR include them.
    # Inheritance is cleanest for maintaining the "flat" YAML structure defined in SPEC.

    # Re-defining using mixins:

    # Runtime Mixin
    temperature: float = Field(DEFAULT_TEMP, ge=0.0, description="MD temperature in Kelvin (K)")
    pressure: float = Field(DEFAULT_PRESSURE, ge=0.0, description="MD pressure in Bar")
    timestep: float = Field(DEFAULT_TIMESTEP, gt=0.0, description="Timestep in fs")
    steps: int = Field(DEFAULT_STEPS, gt=0, description="Number of MD steps")
    ensemble: Literal["nvt", "npt"] = Field("nvt", description="MD Ensemble (nvt or npt)")

    # Algorithm Mixin
    uncertainty_threshold: float = Field(DEFAULT_UNCERTAINTY, gt=0.0, description="Max extrapolation grade before stop")
    sampling_interval: int = Field(DEFAULT_INTERVAL, gt=0, description="Interval for thermo output and dumping")
    use_zbl_baseline: bool = Field(True, description="Whether to use ZBL baseline (hybrid/overlay)")
    zbl_inner_cutoff: float = Field(DEFAULT_ZBL_INNER, gt=0.0, description="Inner cutoff for ZBL switching")
    zbl_outer_cutoff: float = Field(DEFAULT_ZBL_OUTER, gt=0.0, description="Outer cutoff for ZBL switching")
    restart_interval: int = Field(DEFAULT_RESTART_INTERVAL, gt=0, description="Interval for writing restart files")

    # EON Integration
    eon: EONConfig | None = Field(None, description="EON Configuration")
    active_engine: Literal["lammps", "eon"] = Field("lammps", description="Active Dynamics Engine")

    model_config = ConfigDict(extra="forbid")

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: Path | None) -> Path | None:
        if v is not None:
            if str(v).strip() == "":
                raise ValueError("LAMMPS executable path cannot be empty.")
        return v

    # Helper properties to expose sub-configs if logic needs them grouped
    @property
    def runtime(self) -> InferenceRuntimeConfig:
        return InferenceRuntimeConfig(
            temperature=self.temperature,
            pressure=self.pressure,
            timestep=self.timestep,
            steps=self.steps,
            ensemble=self.ensemble
        )

    @property
    def algorithm(self) -> InferenceAlgorithmConfig:
        return InferenceAlgorithmConfig(
            uncertainty_threshold=self.uncertainty_threshold,
            sampling_interval=self.sampling_interval,
            use_zbl_baseline=self.use_zbl_baseline,
            zbl_inner_cutoff=self.zbl_inner_cutoff,
            zbl_outer_cutoff=self.zbl_outer_cutoff,
            restart_interval=self.restart_interval
        )
