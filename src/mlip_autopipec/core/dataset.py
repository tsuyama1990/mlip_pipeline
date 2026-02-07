import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from mlip_autopipec.domain_models import Structure


class Dataset:
    """
    Manages a collection of structures stored on disk with a sidecar metadata file.
    Uses JSONL format for structure storage and JSON for metadata.
    """
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.meta_filepath = filepath.with_suffix('.meta.json')
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.filepath.exists():
            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.filepath.touch()

        if not self.meta_filepath.exists():
            # Create metadata file with initial count
            self._write_meta({'count': 0})

    def _write_meta(self, data: dict[str, Any]) -> None:
        """Writes metadata to disk atomically."""
        temp_meta = self.meta_filepath.with_suffix('.tmp')
        with temp_meta.open('w') as f:
            json.dump(data, f)
        temp_meta.replace(self.meta_filepath)

    def _read_meta(self) -> dict[str, Any]:
        """Reads metadata from disk."""
        if not self.meta_filepath.exists():
            return {'count': 0}
        try:
            with self.meta_filepath.open('r') as f:
                data = json.load(f)
                return cast(dict[str, Any], data)
        except json.JSONDecodeError:
            return {'count': 0}

    def append(self, structures: Iterable[Structure]) -> int:
        """
        Appends structures to the dataset file and updates metadata.
        Returns the number of structures added.
        """
        added_count = 0
        with self.filepath.open('a') as f:
            for s in structures:
                # Validate before saving (domain constraint: must be labeled)
                s.validate_labeled()
                f.write(s.model_dump_json() + '\n')
                added_count += 1

        # Update metadata count
        meta = self._read_meta()
        meta['count'] = meta.get('count', 0) + added_count
        self._write_meta(meta)

        return added_count

    def __iter__(self) -> Iterator[Structure]:
        """Iterates over structures in the dataset file using buffered reading."""
        if not self.filepath.exists():
            return

        # Standard file iteration is buffered by default in Python (typically 8KB chunks)
        # To be explicit about "streaming", we rely on this behavior.
        with self.filepath.open('r') as f:
            for line in f:
                if line.strip():
                    yield Structure.model_validate_json(line)

    def count(self) -> int:
        """Returns the number of structures in the dataset from metadata (O(1))."""
        meta = self._read_meta()
        return int(meta.get('count', 0))
