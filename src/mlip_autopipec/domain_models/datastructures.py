from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_iteration: int = Field(default=0, description="The current iteration number")
    completed_tasks: list[str] = Field(default_factory=list, description="List of completed task IDs")
