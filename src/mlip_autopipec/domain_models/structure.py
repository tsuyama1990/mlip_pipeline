from typing import Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class Structure(BaseModel):
    """
    Represents an atomic configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    properties: dict[str, Any] = Field(default_factory=dict)
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None

    @field_validator("positions", "cell", "pbc", "forces", "stress", mode="before")
    @classmethod
    def convert_to_numpy(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v)
        return v

    @field_serializer("positions", "cell", "pbc", "forces", "stress")
    def serialize_numpy(self, v: np.ndarray | None, _info: Any) -> list[Any] | None:
        if v is None:
            return None
        return v.tolist()  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def validate_shapes(self) -> "Structure":
        n_atoms = len(self.symbols)

        # Validate positions shape (N, 3)
        if self.positions.shape != (n_atoms, 3):
            msg = f"Positions shape {self.positions.shape} does not match number of atoms {n_atoms} (expected ({n_atoms}, 3))"
            raise ValueError(msg)

        # Validate cell shape (3, 3)
        if self.cell.shape != (3, 3):
            msg = f"Cell shape {self.cell.shape} must be (3, 3)"
            raise ValueError(msg)

        # Validate pbc shape (3,)
        if self.pbc.shape != (3,):
            msg = f"PBC shape {self.pbc.shape} must be (3,)"
            raise ValueError(msg)

        # Validate forces if present (N, 3)
        if self.forces is not None and self.forces.shape != (n_atoms, 3):
            msg = f"Forces shape {self.forces.shape} does not match number of atoms {n_atoms}"
            raise ValueError(msg)

        # Validate stress if present (3, 3) or (6,) depending on convention.
        if self.stress is not None and self.stress.shape not in [(3, 3), (6,)]:
            msg = f"Stress shape {self.stress.shape} must be (3, 3) or (6,)"
            raise ValueError(msg)

        return self
