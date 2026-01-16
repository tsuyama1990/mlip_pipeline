"""Pydantic schemas for the Pacemaker configuration file."""

from pydantic import BaseModel, ConfigDict, Field


class PacemakerLossWeights(BaseModel):
    """Schema for the loss_weights section of the Pacemaker config."""

    model_config = ConfigDict(extra="forbid")
    energy: float = Field(..., gt=0)
    forces: float = Field(..., gt=0)
    stress: float = Field(..., gt=0)


class PacemakerACEParams(BaseModel):
    """Schema for the ace section of the Pacemaker config."""

    model_config = ConfigDict(extra="forbid")
    radial_basis: str
    correlation_order: int = Field(..., ge=2)
    element_dependent_cutoffs: bool


class PacemakerFitParams(BaseModel):
    """Schema for the fit_params section of the Pacemaker config."""

    model_config = ConfigDict(extra="forbid")
    dataset_filename: str
    loss_weights: PacemakerLossWeights
    ace: PacemakerACEParams


class PacemakerConfig(BaseModel):
    """Root schema for the Pacemaker configuration file."""

    model_config = ConfigDict(extra="forbid")
    fit_params: PacemakerFitParams
