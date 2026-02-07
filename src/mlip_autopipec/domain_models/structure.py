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

    @field_serializer("positions", "cell", "pbc")
    def serialize_numpy(self, v: np.ndarray, _info: Any) -> list[Any]:
        return v.tolist()  # type: ignore[no-any-return]

    @field_validator("positions", mode="before")
    @classmethod
    def validate_positions(cls, v: Any) -> np.ndarray:
        return np.array(v, dtype=float)

    @field_validator("cell", mode="before")
    @classmethod
    def validate_cell(cls, v: Any) -> np.ndarray:
        return np.array(v, dtype=float)

    @field_validator("pbc", mode="before")
    @classmethod
    def validate_pbc(cls, v: Any) -> np.ndarray:
        return np.array(v, dtype=bool)

    @field_validator("positions")
    @classmethod
    def check_positions_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"Positions must be (N, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @field_validator("cell")
    @classmethod
    def check_cell_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.shape != (3, 3):
            msg = f"Cell must be (3, 3), got {v.shape}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> "Structure":
        n_atoms = len(self.symbols)
        if self.positions.shape[0] != n_atoms:
            msg = (
                f"Number of positions ({self.positions.shape[0]}) "
                f"does not match number of symbols ({n_atoms})"
            )
            raise ValueError(msg)
        return self
