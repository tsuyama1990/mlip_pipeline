from pydantic import BaseModel, ConfigDict


class WorkflowConfig(BaseModel):
    """Configuration for workflow orchestration."""
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    model_config = ConfigDict(extra="forbid")
