from typing import Annotated, Any, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, PlainSerializer, BeforeValidator, model_validator


def to_numpy(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def numpy_to_list(v: np.ndarray) -> list:
    return v.tolist()


NumpyArray = Annotated[
    np.ndarray,
    PlainSerializer(numpy_to_list, return_type=list, when_used="json"),
    BeforeValidator(to_numpy),
]


class Structure(BaseModel):
    """
    A wrapper around atomic data (positions, numbers, cell, pbc) and optional properties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positions: NumpyArray
    atomic_numbers: NumpyArray
    cell: NumpyArray
    pbc: Tuple[bool, bool, bool] = (True, True, True)

    # Optional labels/properties
    energy: Optional[float] = None
    forces: Optional[NumpyArray] = None
    stress: Optional[NumpyArray] = None
    properties: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_shapes(self) -> "Structure":
        """
        Validates that positions and forces match the number of atoms (atomic_numbers),
        and that cell is 3x3.
        """
        n_atoms = len(self.atomic_numbers)

        if self.positions.shape != (n_atoms, 3):
            msg = f"positions shape {self.positions.shape} does not match ({n_atoms}, 3)"
            raise ValueError(msg)

        if self.cell.shape != (3, 3):
            msg = f"cell shape {self.cell.shape} must be (3, 3)"
            raise ValueError(msg)

        if self.forces is not None and self.forces.shape != (n_atoms, 3):
            msg = f"forces shape {self.forces.shape} does not match ({n_atoms}, 3)"
            raise ValueError(msg)

        if self.stress is not None and self.stress.shape not in [(3, 3), (6,)]:
            # Allow both Voigt (6,) and full tensor (3, 3)
            msg = f"stress shape {self.stress.shape} must be (3, 3) or (6,)"
            raise ValueError(msg)

        return self

    def validate_labeled(self) -> None:
        """
        Ensures that the structure has all required labels: energy, forces, and stress.
        Raises ValueError if any are missing.
        """
        if self.energy is None:
            msg = "Structure is missing energy label."
            raise ValueError(msg)
        if self.forces is None:
            msg = "Structure is missing forces label."
            raise ValueError(msg)
        if self.stress is None:
            msg = "Structure is missing stress label."
            raise ValueError(msg)
