"""Metrics domain models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiversityMetrics(BaseModel):
    """Metrics for diversity of a structure set."""

    model_config = ConfigDict(extra="forbid")

    mean_pairwise_distance: float = Field(..., description="Mean pairwise distance in descriptor space")
    min_pairwise_distance: float = Field(..., description="Minimum pairwise distance in descriptor space")
    descriptor_coverage: float | None = Field(default=None, description="Coverage of descriptor space")

    # Store raw values if needed for debugging
    raw_distances: list[float] | None = Field(default=None, description="Raw pairwise distances")


class UncertaintyMetrics(BaseModel):
    """Metrics for uncertainty of a structure set."""

    model_config = ConfigDict(extra="forbid")

    mean_uncertainty: float = Field(..., description="Mean uncertainty across the set")
    max_uncertainty: float = Field(..., description="Maximum uncertainty in the set")
    uncertainty_distribution: dict[str, float] | None = Field(
        default=None, description="Distribution stats (e.g. percentiles)"
    )


class ActiveLearningMetrics(BaseModel):
    """Metrics for an active learning cycle."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(..., description="Cycle number")
    n_candidates: int = Field(..., description="Number of candidates evaluated")
    n_selected: int = Field(..., description="Number of structures selected")
    selection_threshold: float | None = Field(default=None, description="Uncertainty threshold used")
    diversity_score: float | None = Field(default=None, description="Diversity score of selected batch")
