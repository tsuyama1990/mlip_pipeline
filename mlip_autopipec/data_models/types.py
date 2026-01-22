from typing import Annotated, Any

from ase import Atoms
from pydantic import BeforeValidator


def validate_ase_atoms(v: Any) -> Atoms:
    """
    Validates that the input is a valid ASE Atoms object.
    Checks for required attributes: 'positions', 'cell', 'numbers'.
    """
    if isinstance(v, Atoms):
        # Basic integrity check
        if len(v) > 0 and v.positions.shape != (len(v), 3):
             raise ValueError("Malformed Atoms object: positions shape mismatch")
        return v

    # Duck typing fallback
    # Must have array-like positions and cell
    required_attrs = ["positions", "cell", "numbers"]
    missing = [attr for attr in required_attrs if not hasattr(v, attr)]

    if missing:
        # Check for getter methods as fallback (some ASE-like objects might use methods)
        missing_methods = [f"get_{attr}" for attr in missing if not hasattr(v, f"get_{attr}")]
        if missing_methods:
             raise ValueError(f"Value must be an ASE Atoms object. Missing attributes: {missing}")

    return v

ASEAtoms = Annotated[Any, BeforeValidator(validate_ase_atoms)]
