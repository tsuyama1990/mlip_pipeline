"""Utility functions for pyacemaker."""

from typing import IO, Any

from pyacemaker.core.exceptions import ConfigurationError
from pyacemaker.domain_models.models import StructureMetadata


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


def generate_dummy_structures(count: int, tags: list[str] | None = None) -> list[StructureMetadata]:
    """Generate a list of dummy structures for testing/mocking."""
    tags = tags or ["dummy"]
    return [
        StructureMetadata(tags=tags, features={"energy": -100.0, "forces": [[0.0, 0.0, 0.0]]})
        for _ in range(count)
    ]
