from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplorerConfig(BaseModel):
    """
    Configuration for the exploration module.
    """
    method: Literal["random", "active_learning"] = Field("random", description="Exploration method")

    model_config = ConfigDict(extra="forbid")
