from pathlib import Path

from pydantic import BaseModel, ConfigDict


class InferenceConfig(BaseModel):
    """Configuration for LAMMPS inference."""
    lammps_executable: Path | None = None
    temperature: float = 300.0
    steps: int = 1000
    model_config = ConfigDict(extra="forbid")

class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path | None
    uncertain_structures: list[Path]
    max_gamma_observed: float
    model_config = ConfigDict(extra="forbid")

class EmbeddingConfig(BaseModel):
    """Configuration for cluster embedding."""
    core_radius: float = 4.0
    buffer_width: float = 2.0
    model_config = ConfigDict(extra="forbid")

    @property
    def box_size(self) -> float:
        return 2 * (self.core_radius + self.buffer_width) + 2.0
