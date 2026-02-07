from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Potential(BaseModel):
    """
    Metadata for a trained potential.
    """
    model_config = ConfigDict(extra="forbid")

    path: Path
    format: str = "yace"
    description: str = ""
