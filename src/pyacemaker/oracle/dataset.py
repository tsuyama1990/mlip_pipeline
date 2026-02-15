"""Dataset Manager for handling atomic structure datasets.

Implements a Framed Pickle format for safe streaming and size validation.
"""

import gzip
import hashlib
import io
import pickle
import struct
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS
from pyacemaker.core.utils import verify_checksum


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler for security."""

    def find_class(self, module: str, name: str) -> object:
        """Whitelist allowed modules for unpickling."""
        # Allow standard builtins and numpy/ase modules
        if module in {"builtins", "copy_reg"}:
            return super().find_class(module, name)

        if module.startswith(("ase", "numpy", "collections")):
            return super().find_class(module, name)

        # Forbid everything else
        msg = f"Global '{module}.{name}' is forbidden during unpickling."
        raise pickle.UnpicklingError(msg)


class LimitedStream(io.BytesIO):
    """A limited stream wrapper to restrict reading to a specific size.

    This avoids loading the entire object into memory before unpickling.
    """

    def __init__(self, stream: IO[bytes], size: int) -> None:
        """Initialize the limited stream."""
        self._stream = stream
        self._remaining = size
        self._total_read = 0

    def read(self, size: int | None = -1) -> bytes:
        """Read bytes from the stream, up to the remaining limit."""
        if self._remaining <= 0:
            return b""

        req_size = -1 if size is None else size

        if req_size < 0 or req_size > self._remaining:
            req_size = self._remaining

        chunk = self._stream.read(req_size)
        read_len = len(chunk)
        self._remaining -= read_len
        self._total_read += read_len
        return chunk

    def readline(self, size: int | None = -1) -> bytes:
        """Read a line from the stream."""
        if self._remaining <= 0:
            return b""

        req_size = -1 if size is None else size

        if req_size < 0 or req_size > self._remaining:
            req_size = self._remaining

        chunk = self._stream.readline(req_size)
        read_len = len(chunk)
        self._remaining -= read_len
        self._total_read += read_len
        return chunk


class DatasetManager:
    """Manages reading and writing of datasets (lists of Atoms).

    Uses a Framed Pickle format:
    [8-byte size][pickled_bytes][8-byte size][pickled_bytes]...
    """

    def __init__(self) -> None:
        """Initialize the Dataset Manager."""
        self.logger = logger.bind(name="DatasetManager")

    def _read_and_process_object_stream(
        self, f: IO[bytes], size: int, path: Path
    ) -> Atoms | None:
        """Deserialize and validate a single object from stream."""
        try:
            limited_stream = LimitedStream(f, size)

            unpickler = RestrictedUnpickler(limited_stream)
            obj = unpickler.load()

            remaining = limited_stream._remaining
            if remaining > 0:
                try:
                    f.seek(remaining, 1)
                except (OSError, AttributeError, io.UnsupportedOperation):
                    f.read(remaining)

            if isinstance(obj, list):
                msg = (
                    "Encountered a list object in stream. "
                    "This likely indicates legacy format usage. "
                    "Please convert dataset to framed format."
                )
                self.logger.error(msg)
                raise TypeError(msg)
            if isinstance(obj, Atoms):
                return obj
        except pickle.UnpicklingError:
            self.logger.exception(f"Corrupted record found in {path}. Stop reading.")
            return None
        return None

    def _read_frame_size(self, f: IO[bytes], path: Path) -> int | None:
        """Read the size of the next frame."""
        size_bytes = f.read(8)
        if not size_bytes:
            return None  # EOF

        try:
            size_val = struct.unpack(">Q", size_bytes)[0]
        except struct.error:
            self.logger.exception(f"Corrupted size header in {path}")
            return None

        size = int(size_val)

        if size > CONSTANTS.max_object_size:
            msg = (
                f"Object size {size} bytes exceeds limit of "
                f"{CONSTANTS.max_object_size} bytes. Potential OOM risk."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return size

    def _skip_frame(self, f: IO[bytes], path: Path) -> bool:
        """Skip the next frame in the stream. Returns True if successful, False on EOF/Error."""
        size = self._read_frame_size(f, path)
        if size is None:
            return False

        try:
            f.seek(size, 1)
        except (OSError, AttributeError, io.UnsupportedOperation):
            # Fallback to read (chunked skip)
            chunk_size = 1024 * 1024
            while size > 0:
                to_read = min(size, chunk_size)
                f.read(to_read)
                size -= to_read
        return True

    def load_iter(
        self,
        path: Path,
        verify: bool = True,
        buffer_size: int = CONSTANTS.default_buffer_size,
        start_index: int = 0,
    ) -> Iterator[Atoms]:
        """Iterate over a dataset from a gzipped framed pickle file (Streaming).

        Args:
            path: Path to the .pckl.gzip file.
            verify: Whether to verify checksum (default: True).
            buffer_size: Read buffer size in bytes.
            start_index: Index to start yielding from (skips deserialization of previous items).

        Yields:
            ase.Atoms objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If checksum verification fails or object size exceeds limit.
            EOFError: If stream ends abruptly (handled internally).

        """
        if not path.exists():
            msg = f"Dataset file not found: {path}"
            raise FileNotFoundError(msg)

        # Verify checksum if present and requested
        if verify:
            checksum_path = path.with_suffix(path.suffix + ".sha256")
            if checksum_path.exists():
                expected = checksum_path.read_text().strip()
                if not verify_checksum(path, expected):
                    msg = f"Checksum verification failed for {path}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.logger.info(f"Checksum verified for {path}")
            else:
                self.logger.warning(f"No checksum file found for {path}")

        self.logger.warning(CONSTANTS.PICKLE_SECURITY_WARNING)

        # Buffered reading optimization
        with (
            gzip.open(path, "rb") as gz_file,
            io.BufferedReader(gz_file, buffer_size=buffer_size) as f,
        ):
            yield from self._process_frames(f, path, start_index)

    def _process_frames(
        self, f: IO[bytes], path: Path, start_index: int
    ) -> Iterator[Atoms]:
        """Process frames from the buffered stream."""
        current_idx = 0
        while True:
            # Optimization: Skip frames if before start_index
            if current_idx < start_index:
                if not self._skip_frame(f, path):
                    break
                current_idx += 1
                continue

            size = self._read_frame_size(f, path)
            if size is None:
                break

            obj = self._read_and_process_object_stream(f, size, path)
            if obj:
                yield obj
            elif obj is None and isinstance(obj, type(None)):
                # If None returned but not because of EOF (which is handled by size check), it might be corruption
                # _read_and_process_object_stream handles corruption by logging and returning None
                break

            current_idx += 1

    def save(self, data: list[Atoms], path: Path) -> None:
        """Save a dataset to a gzipped framed pickle file.

        Delegates to save_iter.

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

    def save_iter(
        self,
        data: Iterator[Atoms] | list[Atoms],
        path: Path,
        mode: str = "wb",
        calculate_checksum: bool = True,
        buffer_size: int = CONSTANTS.default_buffer_size,
    ) -> None:
        """Save a dataset by dumping objects sequentially using Framed Pickle format.

        Args:
            data: Iterable of ase.Atoms objects.
            path: Target path.
            mode: File mode ('wb' for write/overwrite, 'ab' for append).
            calculate_checksum: Whether to calculate and save SHA256 checksum.
            buffer_size: Write buffer size in bytes.

        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Validate mode
        if mode not in ("wb", "ab"):
            msg = f"Invalid mode '{mode}'. Must be 'wb' or 'ab'."
            raise ValueError(msg)

        with (
            gzip.open(path, mode) as gz_file,
            io.BufferedWriter(gz_file, buffer_size=buffer_size) as f,  # type: ignore[arg-type]
        ):
            for atoms in data:
                # Pickle to bytes first
                obj_bytes = pickle.dumps(atoms)
                size = len(obj_bytes)

                if size > CONSTANTS.max_object_size:
                    msg = (
                        f"Object size {size} bytes exceeds limit of "
                        f"{CONSTANTS.max_object_size} bytes. Skipping save."
                    )
                    self.logger.error(msg)
                    continue

                header = struct.pack(">Q", size)

                # Write header and payload
                f.write(header)
                f.write(obj_bytes)

        if calculate_checksum:
            # Always calculate checksum from the file on disk to ensure consistency
            # with verify_checksum which reads the compressed file.
            # The streaming hasher previously hashed uncompressed data, causing mismatch.
            from pyacemaker.core.utils import calculate_checksum as calc_checksum

            checksum = calc_checksum(path)
            path.with_suffix(path.suffix + ".sha256").write_text(checksum)
