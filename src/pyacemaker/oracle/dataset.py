"""Dataset Manager for handling atomic structure datasets."""

import gzip
import pickle
import warnings
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

        CRITICAL SCALABILITY NOTE:
        This method STRICTLY supports sequential streams. If a file contains
        a single pickled list object (legacy format), this method will raise
        a TypeError to prevent implicit loading of massive datasets into memory.

        Args:
            path: Path to the .pckl.gzip file.

        Yields:
            ase.Atoms objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If a list object is encountered (Legacy format rejected).

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
                        msg = (
                            "Encountered a list object in stream. "
                            "Legacy single-list dumps are not supported in load_iter "
                            "to prevent Out-Of-Memory errors. "
                            "Please convert dataset to sequential format using safe tools."
                        )
                        self.logger.error(msg)
                        raise TypeError(msg)
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
        warnings.warn(
            "DatasetManager.save() delegates to save_iter() for scalability. "
            "Prefer using save_iter() directly with iterators.",
            stacklevel=2,
        )
        self.save_iter(data, path)

    def save_iter(self, data: Iterator[Atoms] | list[Atoms], path: Path, mode: str = "wb") -> None:
        """Save a dataset by dumping objects sequentially (Stream-friendly).

        This format allows `load_iter` to read one object at a time without
        loading the whole file into RAM.

        Args:
            data: Iterable of ase.Atoms objects.
            path: Target path.
            mode: File mode ('wb' for write/overwrite, 'ab' for append).

        """
        # Validate mode
        if mode not in ("wb", "ab"):
            msg = f"Invalid mode: {mode}. Must be 'wb' or 'ab'."
            raise ValueError(msg)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, mode) as f:
            for atoms in data:
                pickle.dump(atoms, f)  # type: ignore[arg-type]
