"""Dataset Manager for handling atomic structure datasets."""

import gzip
import pickle
from collections.abc import Iterator
from pathlib import Path

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS


class DatasetManager:
    """Manages reading and writing of datasets (lists of Atoms).

    WARNING: This module uses `pickle` for serialization, which is not secure
    against untrusted data. Only load datasets from trusted sources.
    """

    def __init__(self) -> None:
        """Initialize the Dataset Manager."""
        self.logger = logger.bind(name="DatasetManager")

    def load_iter(self, path: Path) -> Iterator[Atoms]:
        """Iterate over a dataset from a gzipped pickle file (Streaming).

        This method reads objects sequentially from the file. It expects the file
        to be created using `save_iter` (sequentially dumped objects).

        If the file was created as a single list dump (legacy format), this method
        will fail gracefully or attempt to read the first object if possible, but
        streaming support is primarily for sequential dumps.

        Args:
            path: Path to the .pckl.gzip file.

        Yields:
            ase.Atoms objects.

        Raises:
            FileNotFoundError: If the file does not exist.

        """
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        self.logger.warning(CONSTANTS.PICKLE_SECURITY_WARNING)

        with gzip.open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)  # noqa: S301
                    if isinstance(obj, list):
                        # If we encounter a list, it means it's a legacy dump or chunk.
                        # We yield from it, but warn about memory usage if it's huge.
                        # Since we already loaded it, the memory hit happened.
                        yield from obj
                    elif isinstance(obj, Atoms):
                        yield obj
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    self.logger.exception(f"Corrupted record found in {path}. Stop reading.")
                    break

    def save(self, data: list[Atoms], path: Path) -> None:
        """Save a dataset to a gzipped pickle file using streaming format.

        This method now delegates to `save_iter` to ensure files are always
        saved in a stream-friendly format.

        Args:
            data: List of ase.Atoms objects.
            path: Target path.

        """
        self.save_iter(data, path)

    def save_iter(self, data: Iterator[Atoms] | list[Atoms], path: Path) -> None:
        """Save a dataset by dumping objects sequentially (Stream-friendly).

        This format allows `load_iter` to read one object at a time without
        loading the whole file into RAM.

        Args:
            data: Iterable of ase.Atoms objects.
            path: Target path.

        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            for atoms in data:
                pickle.dump(atoms, f)
