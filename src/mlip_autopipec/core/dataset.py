import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self.path.touch()
            self._save_meta({"count": 0})

        if not self.meta_path.exists():
            self._save_meta({"count": 0})

    def _load_meta(self) -> dict[str, Any]:
        with self.meta_path.open("r") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_meta(self, meta: dict[str, Any]) -> None:
        with self.meta_path.open("w") as f:
            json.dump(meta, f)

    def __len__(self) -> int:
        meta = self._load_meta()
        return int(meta.get("count", 0))

    def append(self, structures: Iterable[Structure], batch_size: int | None = None) -> None:
        """
        Append structures to the dataset.

        Args:
            structures: Iterable of Structure objects.
            batch_size: Deprecated. Writing is now streamed line-by-line to avoid memory issues.
        """
        count = len(self)

        with self.path.open("a") as f:
            for s in structures:
                f.write(s.model_dump_json() + "\n")
                count += 1

        self._save_meta({"count": count})

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over structures lazily, skipping invalid lines."""
        if not self.path.exists():
            return

        with self.path.open("r") as f:
            for i, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield Structure.model_validate_json(line)
                except Exception:
                    logger.warning(f"Skipping malformed line {i + 1} in dataset: {line[:50]}...")
