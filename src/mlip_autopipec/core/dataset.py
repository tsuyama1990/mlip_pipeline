import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from mlip_autopipec.domain_models.config import DEFAULT_BUFFER_SIZE
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, path: Path) -> None:
        # Security: Resolve absolute path and prevent traversal
        try:
            # resolve() handles symlinks and '..' components
            # If path does not exist, resolve works if parents exist or on recent Python versions for non-existing.
            # strict=False allows the file itself to be missing.
            self.path = path.resolve(strict=False)

            # Security: Ensure we are not traversing out of expected root?
            # Without a passed root, we can't strictly enforce confinement to a "jail".
            # But resolving prevents '..' attacks hiding the true path.

        except OSError as e:
            msg = f"Invalid path: {path}"
            raise ValueError(msg) from e

        # Additional check for null bytes
        if "\0" in str(self.path):
            msg = "File path contains null byte"
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
