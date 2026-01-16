"""Pydantic models for configuring the Pacemaker training process."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath


class LossWeights(BaseModel):
    """A nested model to define the relative weights in the loss function.

    Attributes:
        energy: Weight for the energy term.
        forces: Weight for the force components.
        stress: Weight for the stress tensor components.
    """

    model_config = ConfigDict(extra="forbid")
    energy: float = Field(1.0, gt=0)
    forces: float = Field(100.0, gt=0)
    stress: float = Field(10.0, gt=0)


class TrainingConfig(BaseModel):
    """The main configuration object for the PacemakerTrainer.

    This model defines the data contract for initiating a training run, ensuring
    that all necessary paths and hyperparameters are validated before the
    process begins.

    Attributes:
        pacemaker_executable: The path to the Pacemaker training binary.
        data_source_db: The path to the ASE database with training data.
        template_file: The path to the Jinja2 template for `pacemaker.in`.
        delta_learning: A flag to enable or disable learning relative to a baseline.
        loss_weights: The nested model for loss function weights.
    """



    model_config = ConfigDict(extra="forbid")
    pacemaker_executable: FilePath
    data_source_db: Path
    template_file: FilePath
    delta_learning: bool = True
    loss_weights: LossWeights = Field(default_factory=LossWeights)
