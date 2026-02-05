from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GlobalConfig(BaseModel):
    """
    Global configuration for the MLIP pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    work_dir: Path
    max_cycles: int = Field(..., gt=0, description="Number of active learning cycles to run.")
    random_seed: int = Field(42, description="Random seed for reproducibility.")
