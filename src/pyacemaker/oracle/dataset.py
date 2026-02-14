"""Dataset Manager for handling atomic structure datasets."""

import gzip
import pickle
from collections.abc import Iterator
from pathlib import Path

from ase import Atoms
from loguru import logger


class DatasetManager:
    """Manages reading and writing of datasets (lists of Atoms).

    WARNING: This module uses `pickle` for serialization, which is not secure
    against untrusted data. Only load datasets from trusted sources.
    """

    def __init__(self) -> None:
        """Initialize the Dataset Manager."""
        self.logger = logger.bind(name="DatasetManager")

    def load(self, path: Path) -> list[Atoms]:
        """Load a complete dataset from a gzipped pickle file.

        Args:
            path: Path to the .pckl.gzip file.

        Returns:
            List of ase.Atoms objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the file content is not a list.

        """
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        # SECURITY WARNING: pickle is unsafe. Do not load untrusted data.
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        if not isinstance(data, list):
            msg = "Dataset file must contain a list of structures"
            raise TypeError(msg)

        return data

    def load_iter(self, path: Path) -> Iterator[Atoms]:
        """Iterate over a dataset from a gzipped pickle file (if supported).

        Note: Standard pickle dump of a list loads the whole list at once.
        This method supports files where objects were dumped sequentially using
        `dump_iter` (simulated here for compatibility). If the file contains
        a single list object, it will load it all and iterate.

        Args:
            path: Path to the .pckl.gzip file.

        Yields:
            ase.Atoms objects.

        """
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        with gzip.open(path, "rb") as f:
            try:
                # Try to load as a single object first (standard format)
                data = pickle.load(f)  # noqa: S301
                if isinstance(data, list):
                    yield from data
                    return
                # If it's a single atom, yield it and continue
                if isinstance(data, Atoms):
                    yield data
            except EOFError:
                return

            # If we are here, we might have multiple objects dumped sequentially
            while True:
                try:
                    obj = pickle.load(f)  # noqa: S301
                    if isinstance(obj, list):
                        yield from obj
                    elif isinstance(obj, Atoms):
                        yield obj
                except EOFError:
                    break

    def save(self, data: list[Atoms], path: Path) -> None:
        """Save a dataset to a gzipped pickle file (standard format).

        Args:
            data: List of ase.Atoms objects.
            path: Target path.

        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
