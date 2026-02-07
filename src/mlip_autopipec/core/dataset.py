from collections.abc import Iterable, Iterator
from pathlib import Path

from mlip_autopipec.domain_models import Structure


class Dataset:
    """
    Manages a collection of structures stored on disk.
    Currently uses JSONL format for simplicity.
    """
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.filepath.exists():
            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.filepath.touch()

    def append(self, structures: Iterable[Structure]) -> None:
        """Appends structures to the dataset file."""
        with self.filepath.open('a') as f:
            for s in structures:
                # Validate before saving (domain constraint: must be labeled)
                s.validate_labeled()
                f.write(s.model_dump_json() + '\n')

    def __iter__(self) -> Iterator[Structure]:
        """Iterates over structures in the dataset file."""
        if not self.filepath.exists():
            return
        with self.filepath.open('r') as f:
            for line in f:
                if line.strip():
                    yield Structure.model_validate_json(line)

    def count(self) -> int:
        """Returns the number of structures in the dataset."""
        if not self.filepath.exists():
            return 0
        count = 0
        with self.filepath.open('r') as f:
            for _ in f:
                count += 1
        return count
