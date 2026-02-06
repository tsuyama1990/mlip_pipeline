from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Dataset(BaseModel):
    """
    A reference to a file containing structures (e.g., .xyz, .extxyz).
    Strictly file-based to ensure scalability and prevent OOM errors.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: Path
