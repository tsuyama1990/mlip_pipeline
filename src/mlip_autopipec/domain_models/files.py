from pydantic import BaseModel, ConfigDict, Field


class FileConfig(BaseModel):
    """Configuration for file names and paths."""
    model_config = ConfigDict(extra="forbid")

    input_filename: str = Field(default="input.yaml")
    dataset_filename: str = Field(default="dataset.pckl.gzip")
    activeset_filename: str = Field(default="dataset_activeset.pckl.gzip")
    potential_filename: str = Field(default="output_potential.yace")

    def __repr__(self) -> str:
        return "<FileConfig>"

    def __str__(self) -> str:
        return "FileConfig"
