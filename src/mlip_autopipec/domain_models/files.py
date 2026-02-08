
from pydantic import BaseModel, ConfigDict, Field

# --- Defaults ---
DEFAULT_PACEMAKER_INPUT_FILENAME = "input.yaml"
DEFAULT_PACEMAKER_DATASET_FILENAME = "dataset.pckl.gzip"
DEFAULT_PACEMAKER_ACTIVESET_FILENAME = "dataset_activeset.pckl.gzip"
DEFAULT_PACEMAKER_POTENTIAL_FILENAME = "output_potential.yace"


class FileConfig(BaseModel):
    """Configuration for file names and paths."""
    model_config = ConfigDict(extra="forbid")

    input_filename: str = Field(default=DEFAULT_PACEMAKER_INPUT_FILENAME)
    dataset_filename: str = Field(default=DEFAULT_PACEMAKER_DATASET_FILENAME)
    activeset_filename: str = Field(default=DEFAULT_PACEMAKER_ACTIVESET_FILENAME)
    potential_filename: str = Field(default=DEFAULT_PACEMAKER_POTENTIAL_FILENAME)
