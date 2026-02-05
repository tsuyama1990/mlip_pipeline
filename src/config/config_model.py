from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GlobalConfig(BaseModel):
    """
    Global configuration for the MLIP pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(..., description="Working directory for the pipeline")
    max_cycles: int = Field(..., gt=0, description="Maximum number of active learning cycles")
    random_seed: int = Field(42, description="Random seed for reproducibility")
