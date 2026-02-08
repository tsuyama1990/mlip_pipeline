import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, List

from mlip_autopipec.domain_models import Structure

logger = logging.getLogger(__name__)


class Dataset:
    """
    File-based dataset using JSONL format.
    Maintains a sidecar metadata file for O(1) counting.
    """

    def __init__(self, path: Path, root_dir: Optional[Path] = None) -> None:
        """
        Args:
            path: Path to the dataset file.
            root_dir: Optional root directory to restrict file operations to.
                      If provided, 'path' must be within 'root_dir'.
        """
        # Resolve paths to handle symlinks and relative segments
        # strict=False because the dataset file might not exist yet
        self.path = path.resolve(strict=False)
        self.root_dir: Optional[Path] = None

        if root_dir:
            # Root directory must exist for security check
            self.root_dir = root_dir.resolve(strict=True)

            # Explicit security check
            # is_relative_to is the correct way to check containment in Python 3.9+
            if not self.path.is_relative_to(self.root_dir):
                msg = f"Dataset path {self.path} is outside root directory {self.root_dir}"
                raise ValueError(msg)

        self.meta_path = self.path.with_suffix(".meta.json")
        self._count = 0

        # Load metadata if exists
        if self.meta_path.exists():
            try:
                with self.meta_path.open("r") as f:
                    meta = json.load(f)
                    self._count = meta.get("count", 0)
            except OSError as e:
                msg = f"Failed to load dataset metadata: {e}"
                raise RuntimeError(msg) from e
        else:
            # Ensure file exists or will be created on append
            # If path doesn't exist, we start with 0 count
            self._save_meta()

    def _save_meta(self) -> None:
        try:
            # Ensure parent directory exists before saving
            if not self.meta_path.parent.exists():
                # Security check: verify parent is safe if root_dir is set
                if self.root_dir:
                    resolved_parent = self.meta_path.parent.resolve(strict=False)
                    if not resolved_parent.is_relative_to(self.root_dir):
                         msg = f"Dataset parent {resolved_parent} is outside root {self.root_dir}"
                         raise ValueError(msg)

                self.meta_path.parent.mkdir(parents=True, exist_ok=True)

            with self.meta_path.open("w") as f:
                json.dump({"count": self._count}, f)
        except OSError as e:
            msg = f"Failed to save dataset metadata: {e}"
            raise RuntimeError(msg) from e

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Structure]:
        """
        Iterates over the dataset.
        Streams data line-by-line to minimize memory usage.
        """
        if not self.path.exists():
            return

        try:
            with self.path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        yield Structure(**data)
                    except json.JSONDecodeError:
                        logger.exception("Corrupt dataset line")
                        continue
        except OSError as e:
            msg = f"Failed to read dataset: {e}"
            raise RuntimeError(msg) from e

    def append(self, structures: Iterable[Structure], batch_size: int = 1000) -> int:
        """
        Appends structures to the dataset.
        Validates that structures are labeled.
        Writes to disk in batches to reduce I/O overhead.

        Args:
            structures: Iterable of structures to append.
            batch_size: Number of structures to buffer before writing. Default 1000.

        Returns:
            Number of structures added.
        """
        added = 0
        # Ensure parent directory exists
        if not self.path.parent.exists():
            if self.root_dir:
                resolved_parent = self.path.parent.resolve(strict=False)
                if not resolved_parent.is_relative_to(self.root_dir):
                     msg = f"Dataset parent {resolved_parent} is outside root {self.root_dir}"
                     raise ValueError(msg)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                msg = f"Failed to create dataset directory: {e}"
                raise RuntimeError(msg) from e

        buffer: List[str] = []
        try:
            with self.path.open("a") as f:
                for structure in structures:
                    # Enforce labeled structure
                    structure.validate_labeled()

                    # Convert to JSON-compatible dict (lists instead of numpy arrays)
                    data = structure.model_dump(mode="json")
                    json_line = json.dumps(data)
                    buffer.append(json_line)
                    added += 1

                    if len(buffer) >= batch_size:
                        f.write("\n".join(buffer) + "\n")
                        buffer.clear()

                # Write remaining items
                if buffer:
                    f.write("\n".join(buffer) + "\n")

        except OSError as e:
            msg = f"Failed to append to dataset: {e}"
            raise RuntimeError(msg) from e
        else:
            self._count += added
            self._save_meta()
            return added
