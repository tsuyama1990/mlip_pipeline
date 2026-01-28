from typing import Annotated, Any

import numpy as np
from ase import Atoms
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


def validate_ase_atoms(v: Any) -> Atoms:
    """
    Validates that the input is a valid ASE Atoms object.
    Checks for required attributes: 'positions', 'cell', 'numbers'.
    """
    # Strict Type Check
    if not isinstance(v, Atoms):
        msg = f"Expected ase.Atoms object, got {type(v).__name__}."
        raise TypeError(msg)

    # Validate shape integrity
    try:
        # Standard ASE uses .positions directly
        positions = v.positions

        # Check N_atoms
        n_atoms = len(v)

        # Check positions shape (N, 3)
        pos_arr = np.array(positions)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
             msg = f"Invalid positions shape: {pos_arr.shape}. Expected (N, 3)."
             raise ValueError(msg)

        if pos_arr.shape[0] != n_atoms:
             msg = f"Positions count ({pos_arr.shape[0]}) does not match len(atoms) ({n_atoms})."
             raise ValueError(msg)

        # Check cell shape (3, 3)
        cell = v.cell
        cell_arr = np.array(cell)
        if cell_arr.shape != (3, 3):
             # ASE allows (3,) for orthorhombic but usually expands to 3x3.
             if cell_arr.shape == (3,):
                 pass
             else:
                 msg = f"Invalid cell shape: {cell_arr.shape}. Expected (3, 3) or (3,)."
                 raise ValueError(msg)

        # Check numbers (atomic numbers) length
        numbers = v.numbers
        if len(numbers) != n_atoms:
             msg = f"Atomic numbers count ({len(numbers)}) does not match len(atoms) ({n_atoms})."
             raise ValueError(msg)

    except Exception as e:
        msg = f"Atoms validation failed: {e}"
        raise ValueError(msg) from e

    return v

def serialize_ase_atoms(v: Atoms) -> dict[str, Any]:
    """Serializes ASE Atoms object for Pydantic/JSON."""
    return {
        "symbols": str(v.symbols),
        "positions": v.get_positions().tolist(),
        "cell": v.get_cell().tolist(),
        "pbc": v.get_pbc().tolist()
    }

ASEAtoms = Annotated[
    Any,
    BeforeValidator(validate_ase_atoms),
    PlainSerializer(serialize_ase_atoms),
    WithJsonSchema({"type": "object", "properties": {"symbols": {"type": "string"}}})
]
