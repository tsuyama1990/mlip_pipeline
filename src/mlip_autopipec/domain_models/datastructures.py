from typing import Iterator, Any, List
from abc import ABC, abstractmethod

# Use Any for Structure for now until a proper Structure model is defined
Structure = Any

class Dataset(ABC):
    """Abstract base class for dataset management."""

    @abstractmethod
    def __iter__(self) -> Iterator[Structure]:
        """Iterate over structures in the dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        pass

    @abstractmethod
    def append(self, structure: Structure) -> None:
        """Append a structure to the dataset."""
        pass

class InMemoryDataset(Dataset):
    """Simple in-memory dataset implementation for Cycle 01/Testing."""

    def __init__(self, structures: List[Structure] = None):
        self._data = structures if structures is not None else []

    def __iter__(self) -> Iterator[Structure]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def append(self, structure: Structure) -> None:
        self._data.append(structure)
