from typing import Iterator, Any
from pathlib import Path
from abc import ABC, abstractmethod
import json

class Structure:
    """
    Represents an atomic structure.
    In Cycle 01, this is a placeholder/wrapper.
    In future cycles, it will wrap ASE Atoms or similar.
    """
    def __init__(self, data: Any = None):
        self.data = data

    def to_dict(self) -> dict:
        return {"data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        return cls(data=d.get("data"))

    def __repr__(self) -> str:
        return f"Structure(data={self.data})"


class Dataset(ABC):
    """Abstract base class for dataset management."""

    @abstractmethod
    def __iter__(self) -> Iterator[Structure]:
        """Iterate over structures in the dataset."""
        pass

    # Removed __len__ from interface to strictly enforce streaming/lazy behavior

    @abstractmethod
    def append(self, structure: Structure) -> None:
        """Append a structure to the dataset."""
        pass

class StreamingDataset(Dataset):
    """
    A dataset that reads/writes to a file (JSONL) line-by-line.
    This ensures O(1) memory usage regardless of dataset size.
    """
    def __init__(self, filepath: Path, mode: str = "r"):
        self.filepath = Path(filepath)
        if mode == "w" and not self.filepath.exists():
             self.filepath.touch()
        elif mode == "r" and not self.filepath.exists():
             raise FileNotFoundError(f"Dataset file not found: {self.filepath}")

    def __iter__(self) -> Iterator[Structure]:
        with open(self.filepath, "r") as f:
            for line in f:
                if line.strip():
                    yield Structure.from_dict(json.loads(line))

    def append(self, structure: Structure) -> None:
        with open(self.filepath, "a") as f:
            f.write(json.dumps(structure.to_dict()) + "\n")

    def count(self) -> int:
        """Efficiently count items without loading them."""
        # Use a generator to count lines without loading content
        with open(self.filepath, "r") as f:
            return sum(1 for _ in f)
