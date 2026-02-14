"""Dataset Manager for handling atomic structure datasets."""

import gzip
import pickle
import warnings
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
        """Load a complete dataset from a gzipped pickle file (Memory Intensive).

        DEPRECATED: Use `load_iter` for handling large datasets to avoid OOM.
        This method loads the entire dataset into memory.

        Args:
            path: Path to the .pckl.gzip file.

        Returns:
            List of ase.Atoms objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the file content is not a list.

        """
        warnings.warn(
            "DatasetManager.load() is deprecated and not memory-safe for large datasets. "
            "Use DatasetManager.load_iter() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        self.logger.warning(
            f"Loading entire dataset from {path}. "
            "Use load_iter() for large files to avoid Memory Errors."
        )
        self.logger.warning("SECURITY WARNING: pickle is unsafe. Do not load untrusted data.")

        with gzip.open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        if not isinstance(data, list):
            msg = "Dataset file must contain a list of structures"
            raise TypeError(msg)

        return data

    def load_iter(self, path: Path) -> Iterator[Atoms]:
        """Iterate over a dataset from a gzipped pickle file.

        This method supports two formats:
        1. A single pickled list (legacy) - Loads entire list then iterates.
        2. Sequentially pickled objects - Reads one object at a time (Streaming).

        For large datasets, ensure they are saved using `save_iter` (sequentially)
        to enable true streaming.

        Args:
            path: Path to the .pckl.gzip file.

        Yields:
            ase.Atoms objects.

        """
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        self.logger.warning("SECURITY WARNING: pickle is unsafe. Do not load untrusted data.")

        with gzip.open(path, "rb") as f:
            try:
                # Try to load as a single object first
                data = pickle.load(f)  # noqa: S301
                if isinstance(data, list):
                    yield from data
                    return
                # If it's a single atom, yield it and continue to next check
                if isinstance(data, Atoms):
                    yield data
            except (EOFError, pickle.UnpicklingError):
                # If loading fail immediately, it might be empty or corrupted differently,
                # but EOFError usually means end of file or empty.
                # UnpicklingError might happen if it's not a valid pickle stream start.
                # We catch and proceed to loop or return if empty.
                pass

            # Continue reading sequentially
            while True:
                try:
                    obj = pickle.load(f)  # noqa: S301
                    if isinstance(obj, list):
                        yield from obj
                    elif isinstance(obj, Atoms):
                        yield obj
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    # If we encounter corruption in stream
                    self.logger.exception(f"Corrupted record found in {path}. Stop reading.")
                    break

    def save(self, data: list[Atoms], path: Path) -> None:
        """Save a dataset to a gzipped pickle file (standard list format).

        Note: This creates a file that requires loading the full list metadata first.
        Use `save_iter` for large datasets.

        Args:
            data: List of ase.Atoms objects.
            path: Target path.

        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)

    def save_iter(self, data: Iterator[Atoms] | list[Atoms], path: Path) -> None:
        """Save a dataset by dumping objects sequentially (Stream-friendly).

        This format allows `load_iter` to read one object at a time without
        loading the whole file into RAM.

        Args:
            data: Iterable of ase.Atoms objects.
            path: Target path.

        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            for atoms in data:
                pickle.dump(atoms, f)
