from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrchestratorConfig(BaseModel):
    """
    Configuration for the Orchestrator component.
    """

    work_dir: Path
    max_cycles: int = Field(default=5, ge=1)

    model_config = ConfigDict(extra="forbid")


class GlobalConfig(BaseModel):
    """
    The root configuration model.
    """

    orchestrator: OrchestratorConfig
    # Placeholders for future cycles
    dft: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    dynamics: dict[str, Any] | None = None
    validator: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")
