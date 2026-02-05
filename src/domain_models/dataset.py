from pydantic import BaseModel, ConfigDict, Field

from .structures import StructureMetadata


class Dataset(BaseModel):
    """
    A collection of atomic structures with metadata.
    """

    model_config = ConfigDict(extra="forbid")

    structures: list[StructureMetadata] = Field(
        default_factory=list, description="List of labeled structures"
    )

    def __len__(self) -> int:
        return len(self.structures)

    def add(self, structure: StructureMetadata) -> None:
        self.structures.append(structure)
