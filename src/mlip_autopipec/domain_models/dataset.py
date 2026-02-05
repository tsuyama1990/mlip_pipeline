from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, PrivateAttr

from mlip_autopipec.domain_models.structures import StructureMetadata


class Dataset(BaseModel):
    """
    Abstraction for a collection of structures.
    Designed to support lazy loading in future cycles to meet scalability constraints.
    """
    model_config = ConfigDict(extra='forbid')

    name: str
    _structures: list[StructureMetadata] = PrivateAttr(default_factory=list)

    def add(self, structure: StructureMetadata) -> None:
        """Add a structure to the dataset."""
        self._structures.append(structure)

    def __iter__(self) -> Iterator[StructureMetadata]: # type: ignore[override]
        """Iterate over structures. In future cycles, this will stream from disk."""
        return iter(self._structures)

    def __len__(self) -> int:
        return len(self._structures)
