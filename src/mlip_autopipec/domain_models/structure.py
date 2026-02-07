
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class Structure(BaseModel):
    """
    Represents an atomic structure with positions, cell, and optional properties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: list[list[float]] | np.ndarray
    cell: list[list[float]] | np.ndarray
    species: list[str]
    energy: float | None = None
    forces: list[list[float]] | np.ndarray | None = None
    stress: list[list[float]] | np.ndarray | None = None

    @field_validator("positions")
    @classmethod
    def validate_positions(
        cls, v: list[list[float]] | np.ndarray
    ) -> list[list[float]] | np.ndarray:
        if isinstance(v, np.ndarray):
            if v.ndim != 2 or v.shape[1] != 3:
                msg = "Positions must be (N, 3)"
                raise ValueError(msg)
        elif isinstance(v, list) and not all(len(x) == 3 for x in v):
            # Basic list validation
            msg = "Positions must be (N, 3)"
            raise ValueError(msg)
        return v

    @field_validator("cell")
    @classmethod
    def validate_cell(
        cls, v: list[list[float]] | np.ndarray
    ) -> list[list[float]] | np.ndarray:
        if isinstance(v, np.ndarray):
            if v.shape != (3, 3):
                msg = "Cell must be (3, 3)"
                raise ValueError(msg)
        elif isinstance(v, list) and (len(v) != 3 or not all(len(x) == 3 for x in v)):
            msg = "Cell must be (3, 3)"
            raise ValueError(msg)
        return v
