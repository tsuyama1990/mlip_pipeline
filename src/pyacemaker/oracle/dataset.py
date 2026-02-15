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
from typing import IO, Any

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS
from pyacemaker.core.utils import verify_checksum

# Size limits for objects to prevent OOM
MAX_OBJECT_SIZE_BYTES = 128 * 1024 * 1024  # 128 MB
DEFAULT_BUFFER_SIZE = 10 * 1024 * 1024  # 10 MB


class DatasetManager:
    """Manages reading and writing of datasets (lists of Atoms).

    Uses a Framed Pickle format:
    [8-byte size][pickled_bytes][8-byte size][pickled_bytes]...
    """

    def __init__(self) -> None:
        """Initialize the Dataset Manager."""
        self.logger = logger.bind(name="DatasetManager")

    def _read_frame(self, f: IO[bytes], path: Path) -> bytes | None:
        """Read a single frame from the stream."""
        # Read 8-byte size header
        size_bytes = f.read(8)
        if not size_bytes:
            return None  # EOF

        try:
            size = struct.unpack(">Q", size_bytes)[0]
        except struct.error:
            self.logger.exception(f"Corrupted size header in {path}")
            return None

        if size > MAX_OBJECT_SIZE_BYTES:
            msg = (
                f"Object size {size} bytes exceeds limit of "
                f"{MAX_OBJECT_SIZE_BYTES} bytes. Potential OOM risk."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Read object bytes
        obj_bytes = f.read(size)
        if len(obj_bytes) != size:
            self.logger.error(f"Incomplete read: expected {size}, got {len(obj_bytes)}")
            return None

        return obj_bytes

    def load_iter(
        self,
        path: Path,
        verify: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
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
        with gzip.open(path, "rb") as gz_file:
            # Wrap in BufferedReader for performance if not already buffered enough
            with io.BufferedReader(gz_file, buffer_size=buffer_size) as f:  # type: ignore[arg-type]
                current_idx = 0
                while True:
                    # Peek or read frame header to determine size
                    # We can't easily peek size without reading logic in _read_frame
                    # But _read_frame reads size then data.
                    # To skip, we read size, then skip data.

                    # Manual frame reading to support skipping
                    size_bytes = f.read(8)
                    if not size_bytes:
                        break  # EOF

                    try:
                        size = struct.unpack(">Q", size_bytes)[0]
                    except struct.error:
                        self.logger.exception(f"Corrupted size header in {path}")
                        break

                    if size > MAX_OBJECT_SIZE_BYTES:
                        msg = (
                            f"Object size {size} bytes exceeds limit of "
                            f"{MAX_OBJECT_SIZE_BYTES} bytes. Potential OOM risk."
                        )
                        self.logger.error(msg)
                        raise ValueError(msg)

                    # Optimization: Skip bytes if before start_index
                    if current_idx < start_index:
                        # Seek forward if possible, or read and discard
                        # gzip stream supports forward seek via read
                        try:
                            f.seek(size, 1)
                        except (OSError, AttributeError, io.UnsupportedOperation):
                            # Fallback to read
                            f.read(size)
                        current_idx += 1
                        continue

                    # Read object bytes
                    obj_bytes = f.read(size)
                    if len(obj_bytes) != size:
                        self.logger.error(f"Incomplete read: expected {size}, got {len(obj_bytes)}")
                        break

                    try:
                        obj = pickle.loads(obj_bytes)  # noqa: S301
                        if isinstance(obj, list):
                            msg = (
                                "Encountered a list object in stream. "
                                "This likely indicates legacy format usage. "
                                "Please convert dataset to framed format."
                            )
                            self.logger.error(msg)
                            raise TypeError(msg)
                        elif isinstance(obj, Atoms):
                            yield obj
                    except pickle.UnpicklingError:
                        self.logger.exception(f"Corrupted record found in {path}. Stop reading.")
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
        buffer_size: int = DEFAULT_BUFFER_SIZE,
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

        # Setup streaming checksum
        hasher = hashlib.sha256() if calculate_checksum and mode == "wb" else None

        # Note: If mode is 'ab', we cannot easily calculate the *full* new checksum
        # without reading the existing file.
        # For now, we only support efficient streaming checksum for 'wb'.
        # If 'ab' and calculate_checksum is True, we warn or perform the expensive check later?
        # The audit asked for streaming checksum. We implement it for 'wb'.
        # For 'ab', we skip it here and do it at the end if strictly required (which is O(N)),
        # or we just don't do it to save time (since we disabled it in orchestrator anyway).

        with gzip.open(path, mode) as gz_file:
            # Buffer the gzip stream
            with io.BufferedWriter(gz_file, buffer_size=buffer_size) as f:  # type: ignore[arg-type]
                for atoms in data:
                    # Pickle to bytes first
                    obj_bytes = pickle.dumps(atoms)
                    size = len(obj_bytes)

                    if size > MAX_OBJECT_SIZE_BYTES:
                        msg = (
                            f"Object size {size} bytes exceeds limit of "
                            f"{MAX_OBJECT_SIZE_BYTES} bytes. Skipping save."
                        )
                        self.logger.error(msg)
                        continue

                    header = struct.pack(">Q", size)

                    # Update hash if active
                    if hasher:
                        hasher.update(header)
                        hasher.update(obj_bytes)

                    # Write header and payload
                    f.write(header)
                    f.write(obj_bytes)

        if calculate_checksum:
            if mode == "wb" and hasher:
                checksum = hasher.hexdigest()
                path.with_suffix(path.suffix + ".sha256").write_text(checksum)
            elif mode == "ab":
                # Fallback to full read for append if requested (expensive)
                # Ideally caller avoids this for large files in append mode.
                from pyacemaker.core.utils import calculate_checksum as calc_checksum
                checksum = calc_checksum(path)
                path.with_suffix(path.suffix + ".sha256").write_text(checksum)
