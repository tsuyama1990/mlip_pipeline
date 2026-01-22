from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    """Configuration for workflow orchestration."""
    max_generations: int = Field(10, gt=0, description="Maximum number of active learning generations")
    workers: int = Field(4, gt=0, description="Number of Dask workers")
    dask_scheduler_address: str | None = Field(None, description="Address of existing Dask scheduler")

    # Optional legacy alias support if needed, but strictly we use max_generations

    model_config = ConfigDict(extra="forbid")
