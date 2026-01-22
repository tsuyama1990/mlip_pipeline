from typing import Annotated, Any

from ase import Atoms
from pydantic import BeforeValidator


def validate_ase_atoms(v: Any) -> Atoms:
    if isinstance(v, Atoms):
        # Basic integrity check
        if len(v) > 0 and v.positions.shape != (len(v), 3):
             raise ValueError("Malformed Atoms object: positions shape mismatch")
        return v
    # Duck typing fallback if needed, or strict check
    if hasattr(v, "get_positions") and hasattr(v, "get_cell"):
        return v
    raise ValueError("Value must be an ASE Atoms object")

ASEAtoms = Annotated[Any, BeforeValidator(validate_ase_atoms)]
