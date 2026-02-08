from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.constants import (
    DEFAULT_PACEMAKER_ACTIVESET_FILENAME,
    DEFAULT_PACEMAKER_DATASET_FILENAME,
    DEFAULT_PACEMAKER_INPUT_FILENAME,
    DEFAULT_PACEMAKER_POTENTIAL_FILENAME,
)


class FileConfig(BaseModel):
    """Configuration for file names and paths."""
    model_config = ConfigDict(extra="forbid")

    input_filename: str = Field(default=DEFAULT_PACEMAKER_INPUT_FILENAME)
    dataset_filename: str = Field(default=DEFAULT_PACEMAKER_DATASET_FILENAME)
    activeset_filename: str = Field(default=DEFAULT_PACEMAKER_ACTIVESET_FILENAME)
    potential_filename: str = Field(default=DEFAULT_PACEMAKER_POTENTIAL_FILENAME)

    def __repr__(self) -> str:
        return "<FileConfig>"

    def __str__(self) -> str:
        return "FileConfig"
