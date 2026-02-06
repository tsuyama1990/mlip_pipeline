from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GlobalConfig(BaseModel):
    """
    Global configuration for the pipeline.

    Attributes:
        work_dir: Directory for artifacts.
        max_cycles: Number of active learning cycles to run.
        random_seed: Seed for reproducibility.
    """

    work_dir: Path
    max_cycles: int = Field(gt=0)
    random_seed: int = Field(default=42)

    model_config = ConfigDict(extra="forbid")

    @field_validator("work_dir")
    @classmethod
    def validate_work_dir(cls, v: Path) -> Path:
        """Validate that the working directory is safe."""
        # Check for suspicious path components
        if ".." in str(v):
            msg = "Work directory path cannot contain '..'"
            raise ValueError(msg)

        # We don't necessarily want to resolve it here if we want to allow relative paths
        # that are resolved at runtime, but for security, absolute paths are better.
        # Let's just ensure it's not empty or current directory.
        if str(v).strip() == "" or str(v) == ".":
            msg = "Work directory cannot be empty or current directory"
            raise ValueError(msg)

        return v
