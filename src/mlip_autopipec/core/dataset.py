import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from mlip_autopipec.domain_models import Structure

logger = logging.getLogger(__name__)


class Dataset:
    """
    File-based dataset using JSONL format.
    Maintains a sidecar metadata file for O(1) counting.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
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
                self.meta_path.parent.mkdir(parents=True, exist_ok=True)

            with self.meta_path.open("w") as f:
                json.dump({"count": self._count}, f)
        except OSError as e:
            msg = f"Failed to save dataset metadata: {e}"
            raise RuntimeError(msg) from e

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Structure]:
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

    def append(self, structures: Iterable[Structure]) -> int:
        """
        Appends structures to the dataset.
        Validates that structures are labeled.
        Streams item-by-item to disk.
        """
        added = 0
        # Ensure parent directory exists
        if not self.path.parent.exists():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                msg = f"Failed to create dataset directory: {e}"
                raise RuntimeError(msg) from e

        try:
            with self.path.open("a") as f:
                for structure in structures:
                    # Enforce labeled structure
                    structure.validate_labeled()

                    # Convert to JSON-compatible dict (lists instead of numpy arrays)
                    data = structure.model_dump(mode="json")
                    json_line = json.dumps(data)
                    f.write(json_line + "\n")
                    added += 1

        except OSError as e:
            msg = f"Failed to append to dataset: {e}"
            raise RuntimeError(msg) from e
        else:
            self._count += added
            self._save_meta()
            return added
