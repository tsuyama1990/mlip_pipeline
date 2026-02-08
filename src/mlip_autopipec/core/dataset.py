import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


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

    def append(self, structures: Iterable[Structure]) -> None:
        count = len(self)
        with self.path.open("a") as f:
            for s in structures:
                f.write(s.model_dump_json() + "\n")
                count += 1
        self._save_meta({"count": count})

    def __iter__(self) -> Iterator[Structure]:
        with self.path.open("r") as f:
            for line in f:
                if line.strip():
                    yield Structure.model_validate_json(line)
