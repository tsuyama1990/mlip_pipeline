from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(BaseModel):
    """
    Tracks the current state of the automated workflow.
    """

    current_generation: int = Field(
        0, description="The current active learning generation (0-based)."
    )
    status: Literal["idle", "dft", "training", "inference", "extraction"] = Field(
        "idle", description="Current phase of the workflow."
    )
    pending_tasks: list[str] = Field(
        default_factory=list, description="List of pending task IDs or descriptions."
    )

    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseModel):
    """
    Configuration for the Workflow Manager and Task Queue.
    """

    max_generations: int = Field(
        5, description="Maximum number of active learning generations.", ge=1
    )
    dask_scheduler_address: str | None = Field(
        None, description="Address of existing Dask scheduler. If None, start LocalCluster."
    )
    workers: int = Field(4, description="Number of workers for LocalCluster.", ge=1)

    model_config = ConfigDict(extra="forbid")


class DashboardData(BaseModel):
    """
    Data structure for dashboard reporting.
    """

    generations: list[int] = Field(default_factory=list)
    rmse_values: list[float] = Field(default_factory=list)
    structure_counts: list[int] = Field(default_factory=list)
    status: str = Field("Unknown")

    model_config = ConfigDict(extra="forbid")
