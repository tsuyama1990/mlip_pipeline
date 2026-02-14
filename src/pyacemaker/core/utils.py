"""Utility functions for pyacemaker."""

from collections.abc import Iterator
from typing import IO, Any

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
            chunks = []
            while True:
                chunk = self.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            # Use join for efficient string concatenation
            return "".join(chunks)

        chunk = self.stream.read(size)
        self.total_read += len(chunk)
        if self.total_read > self.limit:
            msg = f"Configuration file exceeds limit of {self.limit} bytes"
            raise ConfigurationError(msg)
        return chunk

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the underlying stream."""
        return getattr(self.stream, name)


def generate_dummy_structures(count: int, tags: list[str] | None = None) -> Iterator[StructureMetadata]:
    """Generate a sequence of dummy structures for testing/mocking.

    Returns an iterator to avoid loading all structures into memory at once.
    """
    tags = tags or ["dummy"]
    # Provide minimal MaterialDNA for testing
    dna = MaterialDNA(composition={"Fe": 1.0}, crystal_system="cubic")
    for _ in range(count):
        yield StructureMetadata(
            tags=tags,
            material_dna=dna,
            # Features can be used for extra data, but core properties are explicit now
            features={"mock_feature": "test"},
        )
