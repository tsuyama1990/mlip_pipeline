from typing import Literal, List
from pydantic import BaseModel, ConfigDict, Field

class SurrogateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(default="medium", description="MACE model size or path")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device to run inference on")
    fps_n_samples: int = Field(default=100, ge=1, description="Number of samples to select via FPS")
    force_threshold: float = Field(default=50.0, gt=0.0, description="Force threshold in eV/A for filtering")
    descriptor_type: Literal["soap", "ace", "mace_latent"] = Field(default="soap", description="Type of descriptor to use")

class SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_indices: List[int] = Field(description="Indices of selected structures")
    scores: List[float] = Field(description="Distance scores for selected structures")
