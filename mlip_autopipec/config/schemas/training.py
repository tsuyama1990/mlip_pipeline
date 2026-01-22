from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class TrainingConfig(BaseModel):
    """
    Configuration for Pacemaker training.
    """
    # File Paths
    training_data_path: str = Field("data/train.xyz", description="Path to training data file")
    test_data_path: str = Field("data/test.xyz", description="Path to validation data file")

    # Physics Constraints
    cutoff: PositiveFloat = Field(..., description="Radial cutoff for the potential (Angstrom)")
    b_basis_size: PositiveInt = Field(..., description="Number of basis functions (complexity)")

    # Loss Function Weights
    kappa: PositiveFloat = Field(..., description="Weight for Energy loss")
    kappa_f: PositiveFloat = Field(..., description="Weight for Force loss")

    # Optimization
    max_iter: PositiveInt = Field(1000, description="Maximum training epochs")

    model_config = ConfigDict(extra="forbid")


class TrainConfig(TrainingConfig):
    """Alias for backward compatibility if needed."""


class TrainingMetrics(BaseModel):
    """
    Metrics extracted from training logs.
    """
    epoch: int = Field(..., ge=0, description="Current epoch")
    rmse_energy: float = Field(..., ge=0.0, description="Energy error (meV/atom)")
    rmse_force: float = Field(..., ge=0.0, description="Force error (eV/A)")

    model_config = ConfigDict(extra="forbid")


class TrainingResult(BaseModel):
    """Result of training."""
    success: bool
    potential_path: str | None = None
    metrics: TrainingMetrics | None = None

    model_config = ConfigDict(extra="forbid")
