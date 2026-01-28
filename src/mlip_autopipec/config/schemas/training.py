from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator


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
    max_num_epochs: PositiveInt = Field(
        1000, alias="max_iter", description="Maximum training epochs"
    )
    batch_size: PositiveInt = Field(..., description="Batch size for training")
    ladder_step: list[int] = Field(
        default_factory=list, description="Ladder steps for hierarchical training"
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        """
        Validates that the cutoff is a positive value and within a reasonable physical range.
        Standard interatomic potentials rarely exceed 20 Angstroms and are at least 1 Angstrom.
        """
        if v < 1.0:
            msg = "Cutoff is too small (< 1.0 A)."
            raise ValueError(msg)
        if v > 20.0:
            msg = "Cutoff is unusually large (> 20.0 A). Please verify."
            raise ValueError(msg)
        return v

    @field_validator("b_basis_size")
    @classmethod
    def validate_b_basis_size(cls, v: int) -> int:
        """
        Validates that the basis size is a positive integer.
        """
        if v <= 0:
            msg = "Basis size must be positive."
            raise ValueError(msg)
        return v


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
