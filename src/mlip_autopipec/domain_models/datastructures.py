from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.enums import WorkflowStatus


class WorkflowState(BaseModel):
    """
    Represents the serializable state of the workflow.
    """

    iteration: int = Field(default=0, ge=0)
    status: WorkflowStatus = Field(default=WorkflowStatus.IDLE)
    current_potential_path: Path | None = None
    current_dataset_path: Path | None = None

    model_config = ConfigDict(extra="forbid")
