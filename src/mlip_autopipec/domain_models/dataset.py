from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .structures import StructureMetadata


class Dataset(BaseModel):
    """
    Abstraction for a collection of labeled structures.
    Designed to support streaming access to avoid loading entire datasets into memory,
    though currently backed by an in-memory list for Cycle 01.
    """

    _structures: list[StructureMetadata] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def add_batch(self, structures: list[StructureMetadata]) -> None:
        """Add a batch of structures to the dataset."""
        self._structures.extend(structures)

    def stream(self) -> Iterator[StructureMetadata]:
        """Stream structures one by one."""
        yield from self._structures

    def __len__(self) -> int:
        return len(self._structures)
