from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    """
    Configuration for the Workflow Manager and orchestration.
    """

    max_generations: int = Field(
        default=5, description="Maximum number of active learning generations.", ge=1
    )
    dask_scheduler_address: str | None = Field(
        default=None, description="Address of existing Dask scheduler. If None, start LocalCluster."
    )
    workers: int = Field(default=4, description="Number of workers for LocalCluster.", ge=1)

    model_config = ConfigDict(extra="forbid")
