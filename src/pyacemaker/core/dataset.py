"""Dataset utilities for PYACEMAKER."""

import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from pyacemaker.core.utils import atoms_to_metadata, metadata_to_atoms
from pyacemaker.domain_models.models import StructureMetadata

if TYPE_CHECKING:
    from pyacemaker.oracle.dataset import DatasetManager


class DatasetSplitter:
    """Helper to split dataset into training stream and validation file."""

    def __init__(
        self,
        dataset_path: "Path",
        validation_path: "Path",
        dataset_manager: "DatasetManager",
        validation_split: float,
        max_validation_size: int,
        buffer_size: int | None = None,
        start_index: int = 0,
    ) -> None:
        """Initialize the DatasetSplitter."""
        from pyacemaker.core.config import CONSTANTS

        self.dataset_path = dataset_path
        self.validation_path = validation_path
        self.dataset_manager = dataset_manager
        self.validation_split = validation_split
        self.max_validation_size = max_validation_size
        self.buffer_size = (
            buffer_size if buffer_size is not None else CONSTANTS.default_validation_buffer_size
        )
        self._rng = secrets.SystemRandom()
        self._val_count = 0
        self.start_index = start_index
        self.processed_count = 0

    def train_stream(self) -> Iterator[StructureMetadata]:
        """Yield training items and save validation items to file as side effect."""
        if not self.dataset_path.exists():
            return

        val_buffer: list[StructureMetadata] = []

        # Optimization: Use start_index in load_iter to skip deserialization of old items
        try:
            stream = self.dataset_manager.load_iter(self.dataset_path, start_index=self.start_index)

            for atoms in stream:
                self.processed_count += 1
                # Simple random split for new items
                is_full = self._val_count >= self.max_validation_size
                should_validate = (not is_full) and (self._rng.random() < self.validation_split)

                if should_validate:
                    val_buffer.append(atoms_to_metadata(atoms))
                    self._val_count += 1
                    if len(val_buffer) >= self.buffer_size:
                        self._flush_validation(val_buffer)
                        val_buffer = []
                else:
                    yield atoms_to_metadata(atoms)

        finally:
            # Ensure remaining buffer is flushed even on error if safe
            if val_buffer:
                import contextlib

                with contextlib.suppress(Exception):
                    self._flush_validation(val_buffer)

    def _flush_validation(self, items: list[StructureMetadata]) -> None:
        """Flush validation buffer to disk."""
        if not items:
            return

        # Convert to atoms
        atoms_iter = (metadata_to_atoms(s) for s in items)
        # Use append mode
        self.dataset_manager.save_iter(
            atoms_iter,
            self.validation_path,
            mode="ab",
            calculate_checksum=False,  # Checksum for append is expensive/complex, skipping for speed
        )
