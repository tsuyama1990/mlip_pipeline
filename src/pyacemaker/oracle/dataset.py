"""Dataset Manager for handling atomic structure datasets."""

import gzip
import pickle
from pathlib import Path

from ase import Atoms


class DatasetManager:
    """Manages reading and writing of datasets (lists of Atoms)."""

    def load(self, path: Path) -> list[Atoms]:
        """Load a dataset from a gzipped pickle file."""
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        with gzip.open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        if not isinstance(data, list):
            msg = "Dataset file must contain a list of structures"
            raise TypeError(msg)

        return data

    def save(self, data: list[Atoms], path: Path) -> None:
        """Save a dataset to a gzipped pickle file."""
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
