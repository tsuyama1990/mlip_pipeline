from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


class Dataset(BaseModel):
    """
    A collection of Structures representing a training dataset.
    """
    structures: list[Structure] = Field(default_factory=list)
    description: str = Field(default="", description="Description of the dataset")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __len__(self) -> int:
        return len(self.structures)

    def add(self, structure: Structure) -> None:
        self.structures.append(structure)
