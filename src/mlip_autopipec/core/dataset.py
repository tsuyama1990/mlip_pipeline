import json
import logging
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, cast

from mlip_autopipec.domain_models.config import DEFAULT_BUFFER_SIZE
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class Dataset:
    """
    Manages the persistent dataset of atomic structures.
    Uses JSONL format for streaming read/write access.
    """

    def __init__(self, path: Path, root_dir: Path | None = None) -> None:
        """
        Initialize Dataset with security checks.

        Args:
            path: Path to the dataset file.
            root_dir: Optional root directory to confine the dataset path.
                      Defaults to the parent of path if not provided (less strict),
                      or CWD. For security, should be explicit.
        """
        # Security: Resolve absolute path
        try:
            self.path = path.resolve(strict=False)
        except OSError as e:
            msg = f"Invalid path: {path}"
            raise ValueError(msg) from e

        # Check for null bytes
        if "\0" in str(self.path):
            msg = "File path contains null byte"
            raise ValueError(msg)

        # Security: Path Confinement
        if root_dir:
            try:
                root = root_dir.resolve(strict=True)
            except OSError as e:
                msg = f"Invalid root directory: {root_dir}"
                raise ValueError(msg) from e

            if not str(self.path).startswith(str(root)):
                msg = f"Path {self.path} is outside the allowed root directory {root}"
                raise ValueError(msg)

        self.meta_path = self.path.with_suffix(".meta.json")
        self._ensure_exists()

    def __repr__(self) -> str:
        try:
            count = len(self)
        except Exception:
            count = -1
        return f"<Dataset(path={self.path}, count={count})>"

    def _ensure_exists(self) -> None:
        try:
            if not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)

            if not self.path.exists():
                self.path.touch()
                self._save_meta({"count": 0})

            if not self.meta_path.exists():
                self._save_meta({"count": 0})
        except OSError:
            logger.exception(f"Failed to initialize dataset at {self.path}")
            raise

    def _load_meta(self) -> dict[str, Any]:
        try:
            with self.meta_path.open("r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    msg = "Metadata file must contain a JSON object"
                    raise TypeError(msg)
                return cast(dict[str, Any], data)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load dataset metadata")
            raise

    def _save_meta(self, meta: dict[str, Any]) -> None:
        try:
            with self.meta_path.open("w") as f:
                json.dump(meta, f)
        except OSError:
            logger.exception("Failed to save dataset metadata")
            raise

    def __len__(self) -> int:
        meta = self._load_meta()
        count = meta.get("count", 0)
        if not isinstance(count, int):
            msg = "Metadata 'count' must be an integer"
            raise TypeError(msg)
        return count

    def append(
        self, structures: Iterable[Structure], buffer_size: int = DEFAULT_BUFFER_SIZE
    ) -> None:
        """
        Append structures to the dataset.
        Uses buffered writing for I/O efficiency.
        Validates that structures are labeled before appending.

        Args:
            structures: Iterable of Structure objects.
            buffer_size: Number of lines to buffer before writing to disk.
        """
        count = len(self)
        added = 0
        buffer: list[str] = []

        try:
            with self.path.open("a") as f:
                for s in structures:
                    # Enforce data integrity
                    s.validate_labeled()
                    buffer.append(s.model_dump_json() + "\n")
                    count += 1
                    added += 1

                    if len(buffer) >= buffer_size:
                        f.writelines(buffer)
                        buffer.clear()

                # Flush remaining
                if buffer:
                    f.writelines(buffer)
                    buffer.clear()

            self._save_meta({"count": count})
            logger.info(f"Successfully appended {added} structures to dataset")

        except OSError:
            logger.exception("Failed to append structures to dataset")
            raise

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over structures lazily, skipping invalid lines."""
        if not self.path.exists():
            return

        try:
            with self.path.open("r") as f:
                # Standard iteration over file object is buffered and memory efficient
                for i, raw_line in enumerate(f):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        yield Structure.model_validate_json(line)
                    except Exception:
                        logger.warning(
                            f"Skipping malformed line {i + 1} in dataset: {line[:50]}..."
                        )
        except OSError:
            logger.exception("Failed to iterate over dataset")
            raise

    def iter_batches(self, batch_size: int = 100) -> Iterator[list[Structure]]:
        """
        Iterate over the dataset in batches.
        Useful for efficient loading into training processes or batched calculations.

        Args:
            batch_size: Number of structures per batch.

        Yields:
            List of Structure objects.
        """
        iterator = iter(self)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch
