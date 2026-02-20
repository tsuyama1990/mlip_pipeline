"""IO Utility functions for PYACEMAKER."""

import io
from typing import TextIO


class LimitedStream(io.StringIO):
    """Stream wrapper that enforces a maximum size limit."""

    def __init__(self, stream: TextIO, limit: int) -> None:
        """Initialize LimitedStream.

        Args:
            stream: The underlying stream object.
            limit: The maximum number of bytes to read.

        Raises:
            ValueError: If the stream content exceeds the limit.

        """
        content = stream.read(limit + 1)
        if len(content) > limit:
            msg = f"Configuration file exceeds limit of {limit} bytes"
            raise ValueError(msg)
        self.total_read = len(content)  # Track bytes read for testing
        super().__init__(content)
