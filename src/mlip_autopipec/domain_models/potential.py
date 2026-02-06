from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Domain model representing a trained MLIP potential.
    """
    path: Path = Field(description="Path to the potential file (e.g., .yace)")
    type: str = Field(default="yace", description="Type of the potential (e.g., yace, onnx)")
    name: str = Field(default="potential", description="Name of the potential")

    model_config = ConfigDict(extra="forbid")

class ExplorationResult(BaseModel):
    """
    Domain model representing the result of an exploration (simulation) run.
    """
    halted: bool = Field(description="Whether the simulation halted due to high uncertainty")
    dump_file: Path = Field(description="Path to the trajectory/dump file")
    high_gamma_frames: list[int] = Field(default_factory=list, description="List of frame indices with high extrapolation grade")

    model_config = ConfigDict(extra="forbid")
