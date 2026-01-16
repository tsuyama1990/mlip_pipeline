"""Pydantic models for configuring the Pacemaker training process."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath


class LossWeights(BaseModel):
    """A nested model to define the relative weights in the loss function."""
    model_config = ConfigDict(extra="forbid")
    energy: float = Field(1.0, gt=0)
    forces: float = Field(100.0, gt=0)
    stress: float = Field(10.0, gt=0)


class TrainingConfig(BaseModel):
    """The main configuration object for the PacemakerTrainer."""
    model_config = ConfigDict(extra="forbid")
    pacemaker_executable: FilePath
    data_source_db: Path
    template_file: FilePath
    delta_learning: bool = True
    loss_weights: LossWeights = Field(default_factory=LossWeights)


class TrainingData(BaseModel):
    """A Pydantic model to validate the data read from the ASE database."""
    energy: float
    forces: list[list[float]]
