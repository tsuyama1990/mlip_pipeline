"""Utility functions for pyacemaker."""

import io
import random
from collections.abc import Iterator
from typing import IO, Any

from ase import Atoms

from pyacemaker.core.exceptions import ConfigurationError
from pyacemaker.domain_models.models import MaterialDNA, StructureMetadata


class LimitedStream:
    """Wrapper around a file object to limit the number of bytes read."""

    def __init__(self, stream: IO[str], limit: int) -> None:
        """Initialize the limited stream."""
        self.stream = stream
        self.limit = limit
        self.total_read = 0

    def read(self, size: int = -1) -> str:
        """Read from the stream, tracking total bytes read."""
        if size < 0:
            # Read all, but in chunks to enforce limit
            chunk_size = 4096
            buffer = io.StringIO()
            while True:
                chunk = self.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
            return buffer.getvalue()

        chunk = self.stream.read(size)
        self.total_read += len(chunk)
        if self.total_read > self.limit:
            msg = f"Configuration file exceeds limit of {self.limit} bytes"
            raise ConfigurationError(msg)
        return chunk

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the underlying stream."""
        return getattr(self.stream, name)


def generate_dummy_structures(
    count: int, tags: list[str] | None = None
) -> Iterator[StructureMetadata]:
    """Generate a sequence of dummy structures for testing/mocking.

    Returns an iterator to avoid loading all structures into memory at once.
    """
    tags = tags or ["dummy"]

    # Create dummy atoms template
    base_atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.87, 2.87, 2.87], pbc=True)

    for i in range(count):
        # Generate diverse MaterialDNA
        # Vary composition slightly to simulate diversity
        comp_fe = 0.9 + (0.1 * random.random())  # noqa: S311
        dna = MaterialDNA(
            composition={"Fe": comp_fe, "C": 1.0 - comp_fe},
            crystal_system="cubic",
            space_group="Im-3m" if i % 2 == 0 else "Fm-3m",
        )

        yield StructureMetadata(
            tags=tags,
            material_dna=dna,
            # Features can be used for extra data, but core properties are explicit now
            # Include 'atoms' feature for Trainer compatibility
            features={"mock_feature": "test", "atoms": base_atoms.copy()},  # type: ignore[no-untyped-call]
        )
