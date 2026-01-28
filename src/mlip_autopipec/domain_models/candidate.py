from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.atoms import ASEAtoms


class CandidateConfig(BaseModel):
    """
    Configuration for candidate generation and processing.
    """

    perturbation_radius: float = Field(
        default=0.1, description="Radius for random perturbations in Angstrom.", gt=0.0
    )
    num_perturbations: int = Field(
        default=5, description="Number of perturbed candidates to generate per halted structure.", ge=1
    )
    cluster_cutoff: float = Field(
        default=5.0, description="Cutoff radius for cluster embedding in Angstrom.", gt=0.0
    )

    model_config = ConfigDict(extra="forbid")


class CandidateData(BaseModel):
    """
    Schema for candidate structures metadata and payload.
    """

    atoms: ASEAtoms = Field(..., description="ASE Atoms object")
    config_type: str = Field(..., description="Tag indicating origin")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    status: str = Field("pending", pattern="^(pending|training|failed)$")
    generation: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")
