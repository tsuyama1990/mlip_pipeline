from ase import Atoms
from pydantic import BaseModel, ConfigDict


class StructureRecord(BaseModel):
    """Represents a single structure record in the database."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    atoms: Atoms
    config_type: str
    source: str
    surrogate_energy: float | None = None
